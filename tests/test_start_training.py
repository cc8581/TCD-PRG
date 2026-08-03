from __future__ import annotations

from argparse import Namespace

from scripts.start_training import DEFAULT_OVERRIDES, _training_arguments


def test_formal_launcher_defaults_and_user_override_order(tmp_path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text("name: test\n", encoding="utf-8")
    arguments = _training_arguments(Namespace(
        config=config,
        output_dir=tmp_path / "output",
        resume=None,
        initialize=None,
        overrides=["training.batch_size=2"],
    ))
    assert "backbone.backend=point_transformer_v3" in DEFAULT_OVERRIDES
    assert "backbone.enable_flash_attention=false" in arguments
    assert "dataset.scene_points=16384" in arguments
    assert arguments[-1] == "training.batch_size=2"
    assert not any("dry" in argument.lower() for argument in arguments)
