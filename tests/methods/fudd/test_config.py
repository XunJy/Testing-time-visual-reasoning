from __future__ import annotations

from pathlib import Path

import pytest

from ttvr.methods.fudd.config import OFFICIAL_CLIP_MODEL, FuDDConfig


def test_config_normalises_paths_and_serialises_to_json_safe_values(tmp_path: Path) -> None:
    config = FuDDConfig(
        data_root=tmp_path / "data",
        prompt_root=str(tmp_path / "prompts"),
        cache_dir=tmp_path / "cache",
        seed=2026,
    )

    assert isinstance(config.data_root, Path)
    assert isinstance(config.prompt_root, Path)
    assert isinstance(config.cache_dir, Path)
    assert config.to_dict()["data_root"] == str(tmp_path / "data")
    assert config.to_dict()["cache_dir"] == str(tmp_path / "cache")
    assert config.to_dict()["seed"] == 2026
    assert config.to_dict()["precision"] == "fp32"
    assert config.is_official_clip_reproduction


def test_config_marks_changed_protocol_as_non_official(tmp_path: Path) -> None:
    changed_top_k = FuDDConfig(tmp_path, tmp_path, top_k=20)
    changed_backbone = FuDDConfig(tmp_path, tmp_path, model_name="ViT-B/32")
    changed_precision = FuDDConfig(
        tmp_path,
        tmp_path,
        precision="native",
    )

    assert changed_top_k.model_name == OFFICIAL_CLIP_MODEL
    assert not changed_top_k.is_official_clip_reproduction
    assert not changed_backbone.is_official_clip_reproduction
    assert not changed_precision.is_official_clip_reproduction


@pytest.mark.parametrize("top_k", [0, 4, 201])
def test_config_rejects_invalid_candidate_count(tmp_path: Path, top_k: int) -> None:
    with pytest.raises(ValueError, match="top_k"):
        FuDDConfig(tmp_path, tmp_path, top_k=top_k)
