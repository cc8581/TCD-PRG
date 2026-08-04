"""Offline diagnostics that are too expensive for the normal training loop."""

from .gradients import family_gradient_norms

__all__ = ["family_gradient_norms"]
