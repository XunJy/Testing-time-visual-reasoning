from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from scripts.feature_adapter import run_clip_multi_bird as runner
from ttvr.data.bird_manifest import BirdSample, BirdTaxon
from ttvr.methods.feature_adapter import FeatureTask, PreparedFeatureTask
from ttvr.methods.feature_adapter.tasks import stable_taxon_partition
from ttvr.models.clip import OpenAIClipCheckpoint, OpenAIClipInstallation


class _SyntheticCub:
    def __init__(self, count: int) -> None:
        self.fingerprint = "synthetic-cub-fingerprint"
        self.targets = tuple(index % 2 for index in range(count))

    def __len__(self) -> int:
        return len(self.targets)


def _backend() -> SimpleNamespace:
    return SimpleNamespace(
        cache_identity="synthetic-backend",
        model_name="synthetic-model",
        precision="fp32",
        feature_dtype_name="torch.float32",
    )


def _cub_payload(dataset: _SyntheticCub) -> dict[str, object]:
    count = len(dataset)
    features = torch.zeros((count, 768), dtype=torch.float32)
    features[torch.arange(count), torch.arange(count) % 768] = 1.0
    backend = _backend()
    return {
        "format": 2,
        "cache_identity": backend.cache_identity,
        "model_name": backend.model_name,
        "precision": backend.precision,
        "dtype": backend.feature_dtype_name,
        "cache_tag": f"cub:test:{dataset.fingerprint}:{backend.cache_identity}",
        "sample_count": count,
        "features": features,
        "labels": torch.tensor(dataset.targets, dtype=torch.long),
        "sample_indices": torch.arange(count, dtype=torch.long),
    }


def test_source_digest_covers_usgs_and_nm_preparation_dependencies() -> None:
    project_root = Path(__file__).resolve().parents[3]
    paths = runner._source_digest_paths(
        project_root,
        (
            "usgs-aerial-avian-2023-publisher-crops",
            "nm-uas-waterfowl-expert-consensus-v1",
        ),
    )
    relative = {path.relative_to(project_root).as_posix() for path in paths}

    assert {
        "scripts/feature_adapter/cache_clip_manifest_features.py",
        "scripts/feature_adapter/verify_clip_runtime.py",
        "src/ttvr/metrics.py",
        "src/ttvr/data/bird_crops.py",
        "src/ttvr/data/bird_source_archive.py",
        "src/ttvr/data/birdnet_lock.py",
        "src/ttvr/data/usgs_aerial_avian.py",
        "src/ttvr/data/nm_uas_waterfowl.py",
    } <= relative


def test_model_runtime_config_records_consuming_interpreter_identity(
    tmp_path: Path,
) -> None:
    installation = OpenAIClipInstallation(
        distribution="clip",
        version="1.0",
        repository_url="https://github.com/openai/CLIP.git",
        vcs="git",
        commit_id="a" * 40,
    )
    checkpoint = OpenAIClipCheckpoint(
        model_name="ViT-L/14@336px",
        path=tmp_path / "ViT-L-14-336px.pt",
        sha256="b" * 64,
        size_bytes=123,
    )
    backend = SimpleNamespace(
        cache_identity=f"openai-clip:ViT-L/14@336px@{'a' * 40}",
        openai_clip_installation=installation,
        openai_clip_checkpoint=checkpoint,
    )

    value = runner._model_runtime_config(backend)  # type: ignore[arg-type]

    assert value["clip_commit"] == "a" * 40
    assert value["checkpoint_sha256"] == "b" * 64
    assert value["cache_identity"].endswith("@" + "a" * 40)


def test_cub_target_cache_validator_locks_metadata_alignment_and_norms(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runner, "CUB_TEST_COUNT", 4)
    dataset = _SyntheticCub(4)
    payload = _cub_payload(dataset)

    features, labels = runner._validated_cub_target_cache(payload, dataset, _backend())

    assert features.shape == (4, 768)
    assert labels.tolist() == list(dataset.targets)


@pytest.mark.parametrize("corruption", ["tag", "indices", "labels", "norm", "finite"])
def test_cub_target_cache_validator_rejects_corruption(
    monkeypatch: pytest.MonkeyPatch,
    corruption: str,
) -> None:
    monkeypatch.setattr(runner, "CUB_TEST_COUNT", 4)
    dataset = _SyntheticCub(4)
    payload = _cub_payload(dataset)
    if corruption == "tag":
        payload["cache_tag"] = f"cub:test:{dataset.fingerprint}:4:synthetic-backend"
    elif corruption == "indices":
        payload["sample_indices"] = torch.tensor([0, 2, 1, 3], dtype=torch.long)
    elif corruption == "labels":
        payload["labels"] = torch.tensor([1, 0, 1, 0], dtype=torch.long)
    elif corruption == "norm":
        assert isinstance(payload["features"], torch.Tensor)
        payload["features"][0].mul_(2.0)
    else:
        assert isinstance(payload["features"], torch.Tensor)
        payload["features"][0, 0] = torch.nan

    with pytest.raises(RuntimeError):
        runner._validated_cub_target_cache(payload, dataset, _backend())


def test_source_task_membership_records_taxa_and_stable_index_digest() -> None:
    task = FeatureTask(
        name="source:selection-train",
        features=torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]]),
        labels=torch.tensor([0, 1, 0], dtype=torch.long),
        text_prototypes=torch.eye(2),
    )
    prepared = PreparedFeatureTask(
        task=task,
        taxon_ids=("taxon:a", "taxon:b"),
        source_indices=torch.tensor([1, 3, 5], dtype=torch.long),
    )
    segments = ({"split": "train", "start": 0, "stop": 6, "cache_path": "/x.pt"},)

    row = runner._source_task_membership_row(
        prepared,
        dataset_id="source",
        role="selection_train",
        source_index_segments=segments,
    )

    assert row["taxon_ids"] == ("taxon:a", "taxon:b")
    assert row["source_indices_count"] == 3
    assert row["source_indices_sha256"] == runner._source_indices_sha256(prepared.source_indices)
    assert row["source_index_space_size"] == 6


def test_source_config_uses_one_global_validation_fraction(tmp_path: Path) -> None:
    path = tmp_path / "sources.json"
    path.write_text(
        json.dumps(
            {
                "validation_taxon_fraction": 0.2,
                "sources": [
                    {
                        "dataset_id": "first",
                        "root": "data/first",
                        "samples": "first-samples.jsonl",
                        "taxa": "first-taxa.jsonl",
                        "train_cache": "first-train.pt",
                    },
                    {
                        "dataset_id": "second",
                        "root": "data/second",
                        "samples": "second-samples.jsonl",
                        "taxa": "second-taxa.jsonl",
                        "train_cache": "second-train.pt",
                        "validation_cache": "second-validation.pt",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    specs, fraction = runner._source_specs(path, tmp_path)

    assert fraction == 0.2
    assert specs[0].validation_cache is None
    assert specs[1].validation_cache == tmp_path / "second-validation.pt"


def _sample(
    dataset_id: str,
    sample_id: str,
    taxon_id: str,
    *,
    split: str = "train",
    sha256: str | None = None,
    phash: str = "0000000000000000",
) -> BirdSample:
    return BirdSample(
        dataset_id=dataset_id,
        source_sample_id=sample_id,
        source_split=split,
        relative_path=f"images/{sample_id}.png",
        image_uri=f"https://example.invalid/{sample_id}",
        group_id=sample_id,
        raw_label=taxon_id,
        taxon_id=taxon_id,
        sha256=sha256 or f"{int(sample_id.encode().hex(), 16) % (1 << 256):064x}",
        phash=phash,
        license="CC0",
        author="test",
        source="synthetic",
    )


class _SplitDataset:
    def __init__(self, samples: tuple[BirdSample, ...]) -> None:
        self.samples = samples
        self.taxon_ids = tuple(sorted({sample.taxon_id for sample in samples}))

    def __len__(self) -> int:
        return len(self.samples)


def _cached_split(
    samples: tuple[BirdSample, ...],
    path: Path,
    *,
    feature_dim: int = 4,
) -> runner.CachedSplit:
    dataset = _SplitDataset(samples)
    local_label = {taxon_id: index for index, taxon_id in enumerate(dataset.taxon_ids)}
    labels = torch.tensor(
        [local_label[sample.taxon_id] for sample in samples],
        dtype=torch.long,
    )
    features = torch.zeros((len(samples), feature_dim), dtype=torch.float32)
    features[torch.arange(len(samples)), torch.arange(len(samples)) % feature_dim] = 1.0
    return runner.CachedSplit(  # type: ignore[arg-type]
        dataset=dataset,
        features=features,
        labels=labels,
        cache_path=path,
    )


def _loaded_source(
    dataset_id: str,
    train_samples: tuple[BirdSample, ...],
    validation_samples: tuple[BirdSample, ...] = (),
    extra_manifest_samples: tuple[BirdSample, ...] = (),
) -> runner.LoadedSource:
    all_samples = train_samples + validation_samples + extra_manifest_samples
    taxon_ids = tuple(sorted({sample.taxon_id for sample in all_samples}))
    taxa = tuple(
        BirdTaxon(
            taxon_id=taxon_id,
            scientific_name=f"Species {index}",
            common_name=f"Bird {index}",
            taxonomy_source="synthetic",
            taxonomy_version="1",
        )
        for index, taxon_id in enumerate(taxon_ids)
    )
    root = Path("/synthetic") / dataset_id
    return runner.LoadedSource(
        spec=runner.SourceSpec(
            dataset_id=dataset_id,
            root=root,
            samples=root / "samples.jsonl",
            taxa=root / "taxa.jsonl",
            train_cache=root / "train.pt",
            validation_cache=(root / "validation.pt" if validation_samples else None),
        ),
        samples=all_samples,
        taxa=taxa,
        train=_cached_split(train_samples, root / "train.pt"),
        validation=(
            _cached_split(validation_samples, root / "validation.pt")
            if validation_samples
            else None
        ),
    )


def test_source_config_ignores_disabled_placeholders_and_locks_duplicate_policy(
    tmp_path: Path,
) -> None:
    path = tmp_path / "sources.json"
    path.write_text(
        json.dumps(
            {
                "validation_taxon_fraction": 0.1,
                "duplicate_audit": {
                    "exact_sha256_policy": "drop-later-source",
                    "perceptual_hash_policy": "report-only",
                    "perceptual_hamming_threshold": 4,
                },
                "sources": [
                    {
                        "dataset_id": "enabled",
                        "root": "data/enabled",
                        "samples": "samples.jsonl",
                        "taxa": "taxa.jsonl",
                        "train_cache": "train.pt",
                    },
                    {
                        "enabled": False,
                        "dataset_id": "future-source",
                        "reason": "cache not ready",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    specs, _ = runner._source_specs(path, tmp_path)
    duplicate = runner._duplicate_audit_config(path)

    assert [spec.dataset_id for spec in specs] == ["enabled"]
    assert duplicate.exact_sha256_policy == "drop-later-source"
    assert duplicate.perceptual_hamming_threshold == 4
    assert runner._disabled_source_rows(path) == [
        {"dataset_id": "future-source", "reason": "cache not ready"}
    ]


def test_duplicate_audit_drops_only_later_active_exact_rows_and_reports_phash() -> None:
    exact_sha = "a" * 64
    first_train = _sample(
        "first",
        "first-train",
        "taxon:a",
        sha256=exact_sha,
        phash="0000000000000000",
    )
    first_test = _sample(
        "first",
        "first-test",
        "taxon:a",
        split="test",
        sha256=exact_sha,
        phash="0000000000000000",
    )
    second_train = _sample(
        "second",
        "second-train",
        "taxon:b",
        sha256=exact_sha,
        phash="0000000000000001",
    )
    sources = (
        _loaded_source("first", (first_train,), extra_manifest_samples=(first_test,)),
        _loaded_source("second", (second_train,)),
    )

    audit, dropped = runner._audit_cross_source_duplicates(
        sources,
        runner.DuplicateAuditConfig(
            exact_sha256_policy="drop-later-source",
            perceptual_hash_policy="report-only",
            perceptual_hamming_threshold=1,
        ),
    )

    assert dropped == {"first": set(), "second": {"second-train"}}
    exact = audit["exact_sha256_cross_source_groups"][0]
    assert exact["canonical_active_occurrence"]["source_sample_id"] == "first-train"
    assert exact["taxon_conflict"] is True
    assert audit["near_phash_cross_source_pairs"] == [
        {
            "phash_a": "0000000000000000",
            "phash_b": "0000000000000001",
            "hamming_distance": 1,
            "dataset_ids_a": ["first"],
            "dataset_ids_b": ["second"],
            "occurrence_count_a": 2,
            "occurrence_count_b": 1,
            "action": "report-only",
        }
    ]
    assert audit["summary"]["perceptual_duplicate_rows_dropped_total"] == 0


def test_refit_combines_different_split_vocabularies_with_explicit_remap() -> None:
    train = (
        _sample("mixed", "train-a", "taxon:a"),
        _sample("mixed", "train-b", "taxon:b"),
    )
    validation = (
        _sample("mixed", "validation-b", "taxon:b", split="validation"),
        _sample("mixed", "validation-c", "taxon:c", split="validation"),
    )
    source = _loaded_source("mixed", train, validation)

    features, labels, taxa, segments, excluded = runner._combined_refit_rows(
        source,
        dropped_source_sample_ids={"validation-b"},
    )

    assert features.shape == (4, 4)
    assert taxa == ("taxon:a", "taxon:b", "taxon:c")
    assert labels.tolist() == [0, 1, 1, 2]
    assert [(row["split"], row["start"], row["stop"]) for row in segments] == [
        ("train", 0, 2),
        ("validation", 2, 4),
    ]
    assert excluded == {2}


def test_small_source_validation_is_audited_and_skipped_without_breaking_global_task(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    taxon_ids = tuple(f"taxon:{index:02d}" for index in range(33))
    _, held_out = stable_taxon_partition(
        taxon_ids,
        validation_fraction=0.1,
        salt="ttvr-birdmix-global-source-species-validation-v1",
    )
    training_ids = tuple(taxon_id for taxon_id in taxon_ids if taxon_id not in held_out)
    small_ids = (held_out[0], training_ids[0], training_ids[1])
    large_ids = tuple(taxon_id for taxon_id in taxon_ids if taxon_id not in small_ids)

    def split_samples(dataset_id: str, ids: tuple[str, ...], split: str) -> tuple[BirdSample, ...]:
        return tuple(
            _sample(
                dataset_id,
                f"{dataset_id}-{split}-{index}",
                taxon_id,
                split=split,
            )
            for index, taxon_id in enumerate(ids)
        )

    large = _loaded_source(
        "large",
        split_samples("large", large_ids, "train"),
        split_samples("large", large_ids, "validation"),
    )
    small_test = _sample(
        "small",
        "small-test",
        small_ids[0],
        split="test",
    )
    small = _loaded_source(
        "small",
        split_samples("small", small_ids, "train"),
        split_samples("small", small_ids, "validation"),
        (small_test,),
    )

    def synthetic_prototypes(
        backend: object,
        taxa_by_id: dict[str, BirdTaxon],
    ) -> tuple[dict[str, torch.Tensor], list[dict[str, str]]]:
        del backend
        prototypes = {
            taxon_id: torch.eye(4)[index % 4] for index, taxon_id in enumerate(sorted(taxa_by_id))
        }
        return prototypes, []

    monkeypatch.setattr(runner, "_prototype_map", synthetic_prototypes)
    prepared = runner._prepare_source_tasks(
        (large, small),
        SimpleNamespace(),  # type: ignore[arg-type]
        excluded_taxa=set(),
        validation_taxon_fraction=0.1,
    )
    train_tasks, validation_tasks, refit_tasks = prepared[:3]
    membership_rows = prepared[7]
    partition_rows = prepared[8]

    assert len(train_tasks) == 2
    assert len(validation_tasks) == 1
    assert len(refit_tasks) == 2
    assert all(task.name.startswith("large:") for task in validation_tasks)
    assert not any(
        row["dataset_id"] == "small" and row["role"] == "unseen_validation"
        for row in membership_rows
    )
    small_audit = next(row for row in partition_rows if row["dataset_id"] == "small")
    assert small_audit["unseen_validation"]["status"] == ("omitted-fewer-than-two-eligible-taxa")
    assert small_audit["unseen_validation"]["eligible_taxon_count"] == 1
    assert small_audit["official_test_manifest_samples"] == 1
    assert small_audit["official_test_samples_used"] == 0
