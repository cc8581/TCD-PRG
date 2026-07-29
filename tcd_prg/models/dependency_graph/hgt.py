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
    direct_blocker_logits: Tensor
    indirect_blocker_logits: Tensor
    actionable_blocker_logits: Tensor
    topology_order_logits: Tensor


class TaskConditionedDependencyGraph(nn.Module):
    """Graph with O object nodes plus one TASK_GRASP node."""

    def __init__(self, dim: int = 256, physical_relations: int = 5,
                 task_relations: int = 3, layers: int = 3, heads: int = 4) -> None:
        super().__init__()
        self.physical_relations = physical_relations
        self.task_relations = task_relations
        self.task_edge = nn.Sequential(nn.Linear(4 * dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, task_relations))
        self.physical_edge = nn.Sequential(nn.Linear(4 * dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, physical_relations))
        self.layers = nn.ModuleList(
            EdgeAwareHGTLayer(dim, physical_relations + task_relations + 1, heads)
            for _ in range(layers)
        )
        self.direct = nn.Linear(dim, 1)
        self.indirect = nn.Linear(dim, 1)
        self.actionable = nn.Linear(dim, 1)
        self.order_score = nn.Linear(dim, 1)

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
        relation_count = self.physical_relations + self.task_relations + 1
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
        relation_weight[:, -1, :-1, self.physical_relations : -1] = task_probability
        relation_weight[:, :-1, -1, self.physical_relations : -1] = task_probability
        diagonal = torch.arange(total_nodes, device=nodes.device)
        relation_weight[:, diagonal, diagonal, -1] = 1.0
        edge_mask = node_mask[:, :, None] & node_mask[:, None, :]
        for layer in self.layers:
            nodes = layer(nodes, node_mask, relation_weight, edge_mask)
        objects = nodes[:, :-1]
        direct = self.direct(objects).squeeze(-1).masked_fill(~object_mask, -30.0)
        indirect = self.indirect(objects).squeeze(-1).masked_fill(~object_mask, -30.0)
        if not use_indirect_reasoning:
            indirect = torch.full_like(indirect, -30.0)
        actionable = self.actionable(objects).squeeze(-1).masked_fill(~object_mask, -30.0)
        order_score = self.order_score(objects).squeeze(-1)
        topology_order = order_score[:, :, None] - order_score[:, None, :]
        return DependencyGraphOutput(
            nodes, physical_logits, task_logits, direct, indirect, actionable, topology_order
        )
