"""Deterministic renderer interface and process-pool observation provider."""

from __future__ import annotations

from concurrent.futures import Future, ProcessPoolExecutor
from typing import Protocol

from .base import ObservationProvider, ObservationRequest, PointObservation


class StateRenderer(Protocol):
    """Renderer implementations must be picklable for process workers."""

    def render(self, request: ObservationRequest) -> PointObservation: ...


class RenderedObservationProvider(ObservationProvider):
    """Synchronous facade; use ``submit`` outside the GPU training loop."""

    def __init__(self, renderer: StateRenderer, workers: int = 0):
        self.renderer = renderer
        self.pool = ProcessPoolExecutor(max_workers=workers) if workers > 0 else None

    def submit(self, request: ObservationRequest) -> Future[PointObservation]:
        if self.pool is None:
            future: Future[PointObservation] = Future()
            try:
                future.set_result(self.renderer.render(request))
            except BaseException as error:
                future.set_exception(error)
            return future
        return self.pool.submit(self.renderer.render, request)

    def get(self, request: ObservationRequest) -> PointObservation:
        return self.renderer.render(request)

    def close(self) -> None:
        if self.pool is not None:
            self.pool.shutdown(wait=True, cancel_futures=True)

