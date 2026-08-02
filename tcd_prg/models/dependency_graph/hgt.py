"""Edge-aware heterogeneous graph transformer ending at TASK_GRASP."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from ..common import masked_softmax


class EdgeAwareHGTLayer(nn.Module):
    def __init__(self, dim: int, relation_types: int, heads: int = 4) -> None:
        super().__init__()
        if dim % heads:
            raise ValueError("dim must be divisible by heads")
        self.heads, self.head_dim = heads, dim // heads
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.relation_key = nn.Parameter(torch.empty(relation_types, heads, self.head_dim, self.head_dim))
        self.relation_value = nn.Parameter(torch.empty(relation_types, heads, self.head_dim, self.head_dim))
        nn.init.xavier_uniform_(self.relation_key)
        nn.init.xavier_uniform_(self.relation_value)
        self.output = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))

    def forward(
        self, nodes: Tensor, node_mask: Tensor, relation_weight: Tensor, edge_mask: Tensor
    ) -> Tensor:
        b, n, dim = nodes.shape
        q = self.q(nodes).view(b, n, self.heads, self.head_dim)
        k = self.k(nodes).view(b, n, self.heads, self.head_dim)
        v = self.v(nodes).view(b, n, self.heads, self.head_dim)
        rk = torch.einsum("bijr,rhde->bijhde", relation_weight, self.relation_key)
        rv = torch.einsum("bijr,rhde->bijhde", relation_weight, self.relation_value)
        source_k = k[:, None].expand(-1, n, -1, -1, -1)
        source_v = v[:, None].expand(-1, n, -1, -1, -1)
        transformed_k = torch.einsum("bijhde,bijhe->bijhd", rk, source_k)
        transformed_v = torch.einsum("bijhde,bijhe->bijhd", rv, source_v)
        logits = torch.einsum("bihd,bijhd->bihj", q, transformed_k) * self.head_dim**-0.5
        full_mask = edge_mask & node_mask[:, :, None] & node_mask[:, None, :]
        attention = masked_softmax(logits, full_mask[:, :, None, :], dim=-1)
        message = torch.einsum("bihj,bijhd->bihd", attention, transformed_v).flatten(-2)
        nodes = self.norm1(nodes + self.output(message))
        nodes = self.norm2(nodes + self.ffn(nodes))
        return nodes * node_mask.unsqueeze(-1)


@dataclass(slots=True)
class DependencyGraphOutput:
    node_features: Tensor
    physical_edge_logits: Tensor
    task_edge_logits: Tensor
    derived_direct_mask: Tensor
    derived_indirect_mask: Tensor
    derived_dependency_mask: Tensor
    derived_actionable_mask: Tensor
    dependency_prior: Tensor


def derive_dependency_masks(
    physical_edge_logits: Tensor,
    task_edge_logits: Tensor,
    object_mask: Tensor,
    target_object: Tensor | None = None,
    threshold: float = 0.5,
    use_indirect_reasoning: bool = True,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Derive directed blockers and executable frontier from predicted edges.

    Canonical physical channels are ``near, contact, support, press,
    occlude``.  A prerequisite edge points from a dependent object to an
    object above it through ``support``; the reverse ``press`` channel is used
    as an equivalent directed observation.  No label/oracle graph is consumed.
    """

    physical = torch.sigmoid(physical_edge_logits) >= threshold
    direct = (torch.sigmoid(task_edge_logits) >= threshold).any(-1) & object_mask
    del target_object
    prerequisite = physical[..., 2] | physical[..., 3].transpose(1, 2)
    prerequisite &= object_mask[:, :, None] & object_mask[:, None, :]
    dependency = direct.clone()
    if use_indirect_reasoning:
        for _ in range(object_mask.shape[1]):
            expanded = (dependency[:, :, None] & prerequisite).any(1) & object_mask
            updated = dependency | expanded
            if torch.equal(updated, dependency):
                break
            dependency = updated
    indirect = dependency & ~direct
    has_active_prerequisite = (prerequisite & dependency[:, None, :]).any(-1)
    actionable = dependency & ~has_active_prerequisite & object_mask
    return direct, indirect, dependency, actionable


def derive_dependency_prior(
    physical_edge_logits: Tensor, task_edge_logits: Tensor, object_mask: Tensor,
) -> Tensor:
    """Continuous max-product analogue of the deterministic dependency closure."""

    physical = torch.sigmoid(physical_edge_logits)
    direct = torch.sigmoid(task_edge_logits).amax(-1) * object_mask
    prerequisite = torch.maximum(physical[..., 2], physical[..., 3].transpose(1, 2))
    prerequisite = prerequisite * (
        object_mask[:, :, None] & object_mask[:, None, :]
    ).to(prerequisite.dtype)
    dependency = direct
    for _ in range(object_mask.shape[1]):
        expanded = (dependency[:, :, None] * prerequisite).amax(1)
        updated = torch.maximum(dependency, expanded) * object_mask
        if torch.allclose(updated, dependency):
            break
        dependency = updated
    blocker = (prerequisite * dependency[:, None, :]).amax(-1)
    return dependency * (1.0 - blocker).clamp_min(0.0) * object_mask


class TaskConditionedDependencyGraph(nn.Module):
    """Graph with O object nodes plus one TASK_GRASP node."""

    def __init__(self, dim: int = 256, physical_relations: int = 5,
                 task_relations: int = 3, layers: int = 3, heads: int = 4,
                 edge_threshold: float = 0.5) -> None:
        super().__init__()
        self.physical_relations = physical_relations
        self.task_relations = task_relations
        self.edge_threshold = edge_threshold
        self.task_edge = nn.Sequential(nn.Linear(4 * dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, task_relations))
        self.physical_edge = nn.Sequential(nn.Linear(4 * dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, physical_relations))
        self.layers = nn.ModuleList(
            EdgeAwareHGTLayer(dim, physical_relations + 2 * task_relations + 1, heads)
            for _ in range(layers)
        )

    @staticmethod
    def _pairs(nodes: Tensor) -> Tensor:
        source = nodes[:, None].expand(-1, nodes.shape[1], -1, -1)
        destination = nodes[:, :, None].expand(-1, -1, nodes.shape[1], -1)
        return torch.cat((source, destination, source - destination, source * destination), -1)

    def forward(
        self,
        object_tokens: Tensor,
        object_mask: Tensor,
        task_token: Tensor,
        target_object: Tensor | None = None,
        relation_graph: Tensor | None = None,
        use_indirect_reasoning: bool = True,
    ) -> DependencyGraphOutput:
        task_node = task_token[:, None]
        nodes = torch.cat((object_tokens, task_node), 1)
        node_mask = torch.cat((object_mask, torch.ones_like(object_mask[:, :1])), 1)
        object_pairs = self._pairs(object_tokens)
        physical_logits = self.physical_edge(object_pairs)
        task_pair = torch.cat(
            (
                object_tokens,
                task_node.expand_as(object_tokens),
                object_tokens - task_node,
                object_tokens * task_node,
            ),
            -1,
        )
        task_logits = self.task_edge(task_pair)
        b, total_nodes = nodes.shape[:2]
        relation_count = self.physical_relations + 2 * self.task_relations + 1
        relation_weight = torch.zeros(
            (b, total_nodes, total_nodes, relation_count),
            dtype=nodes.dtype,
            device=nodes.device,
        )
        # Graph reasoning always consumes predicted relations. ``relation_graph``
        # is accepted only for API compatibility and supervision is applied to
        # ``physical_logits`` outside this module; Oracle edges never enter the policy.
        del relation_graph
        relation_weight[:, :-1, :-1, : self.physical_relations] = torch.sigmoid(
            physical_logits
        )
        task_probability = torch.sigmoid(task_logits)
        forward = slice(self.physical_relations, self.physical_relations + self.task_relations)
        reverse = slice(
            self.physical_relations + self.task_relations,
            self.physical_relations + 2 * self.task_relations,
        )
        # object -> TASK_GRASP is the causal blocking edge.  Reverse context
        # messages use separate relation parameters and therefore do not
        # symmetrize the learned edge semantics.
        relation_weight[:, -1, :-1, forward] = task_probability
        relation_weight[:, :-1, -1, reverse] = task_probability
        diagonal = torch.arange(total_nodes, device=nodes.device)
        relation_weight[:, diagonal, diagonal, -1] = 1.0
        edge_mask = node_mask[:, :, None] & node_mask[:, None, :]
        for layer in self.layers:
            nodes = layer(nodes, node_mask, relation_weight, edge_mask)
        derived = derive_dependency_masks(
            physical_logits, task_logits, object_mask,
            target_object=target_object,
            threshold=self.edge_threshold,
            use_indirect_reasoning=use_indirect_reasoning,
        )
        dependency_prior = derive_dependency_prior(
            physical_logits, task_logits, object_mask
        )
        return DependencyGraphOutput(
            nodes, physical_logits, task_logits, *derived, dependency_prior,
        )
