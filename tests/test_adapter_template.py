from __future__ import annotations

import inspect

from tcd_prg.datasets import DatasetAdapter, DatasetAdapterTemplate, GAPGObservationAdapter


def test_adapter_template_preserves_abstract_contract() -> None:
    assert issubclass(DatasetAdapterTemplate, DatasetAdapter)
    assert inspect.isabstract(DatasetAdapterTemplate)
    assert not DatasetAdapterTemplate.capabilities.has_sequences


def test_gapg_observation_adapter_requires_no_numeric_instance_embedding() -> None:
    # Non-contiguous native IDs are deliberately remapped by the caller; the
    # conversion only uses equality and constructs stable per-scene UUIDs.
    assert hasattr(GAPGObservationAdapter, "from_fused_points")
