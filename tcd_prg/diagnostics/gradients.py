"""Per-objective shared-parameter gradient diagnostics."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from torch import Tensor


def _norm(grads: Sequence[Tensor | None], reference: Tensor) -> Tensor:
    squared = reference.new_zeros((), dtype=torch.float32)
    for gradient in grads:
        if gradient is not None:
            squared = squared + gradient.detach().float().square().sum()
    return squared.sqrt()


def family_gradient_norms(
    family_losses: Mapping[str, Tensor], parameters: Sequence[Tensor], total_loss: Tensor,
) -> dict[str, float]:
    """Measure each weighted objective on the same shared parameter set.

    This deliberately performs one ``autograd.grad`` per family and therefore
    belongs in an offline audit, never in every formal training step.
    """

    trainable = tuple(parameter for parameter in parameters if parameter.requires_grad)
    if not trainable:
        raise ValueError("Gradient audit received no trainable parameters")
    result: dict[str, float] = {}
    for name, loss in family_losses.items():
        if not loss.requires_grad:
            result[name] = 0.0
            continue
        gradients = torch.autograd.grad(
            loss, trainable, retain_graph=True, allow_unused=True
        )
        result[name] = float(_norm(gradients, total_loss))
    total_gradients = torch.autograd.grad(
        total_loss, trainable, retain_graph=False, allow_unused=True
    )
    result["total"] = float(_norm(total_gradients, total_loss))
    return result
