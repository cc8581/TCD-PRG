"""Read category-to-functional-region semantics from training label catalogs."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import h5py


@lru_cache(maxsize=4)
def allowed_regions(dataset_root: str) -> dict[int, tuple[int, ...]]:
    root = Path(dataset_root) / "task_positive_multistep_sequences" / "scene_labels"
    result: dict[int, set[int]] = {}
    for path in sorted(root.glob("scene_*.h5")):
        with h5py.File(path, "r") as handle:
            scene = next(iter(handle.values()))
            catalog = scene["catalog"]
            categories = catalog["object_category_id"][...]
            objects = catalog["task_object_index"][...]
            labels = catalog["task_label"][...]
            for object_index, label in zip(objects, labels, strict=True):
                result.setdefault(int(categories[int(object_index)]), set()).add(int(label))
        if len(result) >= 20:
            break
    return {key: tuple(sorted(values)) for key, values in result.items()}


def dataset_root_from_app_config(config) -> str:
    import yaml
    paths = yaml.safe_load(config.resolve(
        config.raw["tcd_prg"]["paths_config"]).read_text(encoding="utf-8")) or {}
    return str(paths["dataset_root"])
