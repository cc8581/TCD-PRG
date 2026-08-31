"""Common complete physical action contract for logged and sampled PUSHes."""
from dataclasses import dataclass
import torch
from torch import Tensor


@dataclass(frozen=True)
class PushActions:
    batch_index: Tensor
    object: Tensor
    contact_world: Tensor
    direction_world: Tensor
    push_distance: Tensor

    def validate(self, batch_size: int, objects: int) -> "PushActions":
        k = len(self.batch_index)
        if self.batch_index.dtype != torch.long or self.object.dtype != torch.long:
            raise ValueError("PUSH indices must be int64")
        if self.batch_index.shape != (k,) or self.object.shape != (k,) or self.push_distance.shape != (k,):
            raise ValueError("Invalid PUSH vector shape")
        if self.contact_world.shape != (k, 3) or self.direction_world.shape != (k, 3):
            raise ValueError("PUSH contact/direction must be [K,3]")
        values = (self.contact_world, self.direction_world, self.push_distance)
        if any(v.device != self.batch_index.device for v in (*values, self.object)):
            raise ValueError("PUSH tensors must share a device")
        if any(not bool(torch.isfinite(v).all()) for v in values):
            raise ValueError("PUSH actions must be finite")
        if bool(((self.batch_index < 0) | (self.batch_index >= batch_size)).any()):
            raise ValueError("PUSH scene outside batch")
        if bool(((self.object < 0) | (self.object >= objects)).any()):
            raise ValueError("PUSH object outside condition slots")
        if bool((self.push_distance <= 0).any()):
            raise ValueError("PUSH distance must be positive")
        if not torch.allclose(self.direction_world.norm(dim=-1), torch.ones_like(self.push_distance), atol=1e-4):
            raise ValueError("PUSH directions must be unit vectors")
        if bool((self.direction_world[:, 2].abs() > 1e-5).any()):
            raise ValueError("PUSH directions must be horizontal")
        return self

    def select(self, index: Tensor) -> "PushActions":
        return PushActions(*(getattr(self, name)[index] for name in self.__dataclass_fields__))

    @classmethod
    def empty(cls, xyz: Tensor) -> "PushActions":
        return cls(torch.empty(0, dtype=torch.long, device=xyz.device),
                   torch.empty(0, dtype=torch.long, device=xyz.device),
                   xyz.new_empty((0, 3)), xyz.new_empty((0, 3)), xyz.new_empty(0))
