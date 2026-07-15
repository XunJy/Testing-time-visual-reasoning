#!/usr/bin/env python3
"""Run an immutable, label-aware single-image FuDD prompt diagnostic."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.metadata
import json
import math
import platform
import re
import shutil
import subprocess
import sys
from itertools import combinations
from pathlib import Path
from typing import Any

import torch

from ttvr import (
    CLIPBackend,
    load_official_prompts,
    load_pair_overrides,
    ordered_predictions,
    prepare_cub,
    rerank_candidates,
    validate_class_name_alignment,
)

MODEL_NAME = "ViT-L/14@336px"
MODEL_PRECISION = "fp32"
TOP_K = 10
SEED = 2026
OVERRIDE_PAIR = (46, 47)
OPENAI_CLIP_COMMIT = "a1d071733d7111c9c014f024669f959182114e33"
FUDD_COMMIT = "32264231fec047eb0bbbf59bfdbc8e6d208a096b"


def _utc_now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Any, *, overwrite: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite: {path}")
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if path.exists() and not overwrite:
        temporary.unlink(missing_ok=True)
        raise FileExistsError(f"Refusing to overwrite: {path}")
    temporary.replace(path)


def _write_checksums(run_dir: Path) -> Path:
    output = run_dir / "checksums.sha256"
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite: {output}")
    lines = [
        f"{_sha256(path)}  {path.relative_to(run_dir).as_posix()}"
        for path in sorted(run_dir.rglob("*"))
        if path.is_file() and path != output
    ]
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output


def _source_digest(project_root: Path) -> str:
    files = (
        project_root / "pyproject.toml",
        Path(__file__).resolve(),
        project_root / "src/ttvr/data/cub.py",
        project_root / "src/ttvr/metrics.py",
        project_root / "src/ttvr/models/base.py",
        project_root / "src/ttvr/models/cached.py",
        project_root / "src/ttvr/models/clip.py",
        project_root / "src/ttvr/methods/fudd/prompts.py",
        project_root / "src/ttvr/methods/fudd/evaluation.py",
    )
    digest = hashlib.sha256()
    for path in sorted(files, key=lambda item: item.relative_to(project_root).as_posix()):
        relative = path.relative_to(project_root).as_posix()
        digest.update(relative.encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _environment() -> dict[str, Any]:
    cuda_available = torch.cuda.is_available()
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torchvision": _package_version("torchvision"),
        "pillow": _package_version("pillow"),
        "clip": _package_version("clip"),
        "cuda_available": cuda_available,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu": torch.cuda.get_device_name(0) if cuda_available else None,
        "cuda_device_count": torch.cuda.device_count() if cuda_available else 0,
        "cuda_matmul_allow_tf32": (
            torch.backends.cuda.matmul.allow_tf32 if cuda_available else None
        ),
        "cudnn_allow_tf32": (
            torch.backends.cudnn.allow_tf32 if cuda_available else None
        ),
    }


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _validate_parent_run(parent_dir: Path, sample: dict[str, Any]) -> dict[str, Any]:
    expected = sample["parent_run"]
    if parent_dir.name != expected["run_id"]:
        raise RuntimeError("Parent run id does not match the sample manifest")
    predictions_path = parent_dir / "predictions.jsonl"
    result_path = parent_dir / "result.json"
    if _sha256(predictions_path) != expected["predictions_sha256"]:
        raise RuntimeError("Parent prediction digest does not match the sample manifest")
    if _sha256(result_path) != expected["result_sha256"]:
        raise RuntimeError("Parent result digest does not match the sample manifest")

    checksum_path = parent_dir / "checksums.sha256"
    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        digest, relative = line.split("  ", maxsplit=1)
        if _sha256(parent_dir / relative) != digest:
            raise RuntimeError(f"Parent checksum failed: {relative}")

    matches: list[dict[str, Any]] = []
    with predictions_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if record["sample_index"] == sample["sample_index"]:
                matches.append(record)
    if len(matches) != 1:
        raise RuntimeError("Selected sample is not unique in the parent predictions")
    record = matches[0]
    expected_identity = (
        sample["image_id"],
        sample["relative_path"],
        sample["target_class_id"],
        [item["class_id"] for item in sample["baseline_top10"]],
        sample["fudd_ranking"],
    )
    actual_identity = (
        record["image_id"],
        record["relative_path"],
        record["target_class_id"],
        record["baseline_topk_class_ids"],
        record["fudd_ranked_class_ids"],
    )
    if actual_identity != expected_identity:
        raise RuntimeError("Sample manifest does not reproduce the frozen parent record")
    return record


def _english_word_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z]+(?:-[A-Za-z]+)*", text))


def _validate_override_asset(overrides: Any) -> None:
    if set(overrides) != {OVERRIDE_PAIR}:
        raise RuntimeError("Pilot asset must override exactly pair (46, 47)")
    pair = overrides[OVERRIDE_PAIR]
    if len(pair.descriptions) != 4:
        raise RuntimeError("Pilot asset must contain exactly four replacement descriptions")
    attributes = [item.attribute for item in pair.descriptions]
    if len(set(attributes)) != len(attributes):
        raise RuntimeError("Replacement attributes must be unique")
    expected_names = ("american goldfinch", "european goldfinch")
    for item in pair.descriptions:
        for text, species in zip((item.first, item.second), expected_names, strict=True):
            if species not in text.casefold():
                raise RuntimeError(f"Replacement prompt does not name {species}: {text}")
            if not text.endswith("."):
                raise RuntimeError(f"Replacement prompt must end with a period: {text}")
            if _english_word_count(text) > 28:
                raise RuntimeError(f"Replacement prompt exceeds 28 words: {text}")


def _ranked_rows(
    ranked_ids: list[int],
    candidate_ids: list[int],
    scores: list[float],
    class_names: tuple[str, ...],
) -> list[dict[str, int | float | str]]:
    score_by_class = dict(zip(candidate_ids, scores, strict=True))
    return [
        {
            "rank": rank,
            "class_id": class_id,
            "class_name": class_names[class_id],
            "cosine_score": score_by_class[class_id],
        }
        for rank, class_id in enumerate(ranked_ids, start=1)
    ]


def _method_payload(
    *,
    ranking_scope: str,
    ranked_ids: list[int],
    candidate_ids: list[int],
    scores: list[float],
    class_names: tuple[str, ...],
    target_id: int,
) -> dict[str, Any]:
    target_rank = ranked_ids.index(target_id) + 1
    score_by_class = dict(zip(candidate_ids, scores, strict=True))
    return {
        "ranking_scope": ranking_scope,
        "ranking": _ranked_rows(ranked_ids, candidate_ids, scores, class_names),
        "target_rank": target_rank,
        "top1_correct": ranked_ids[0] == target_id,
        "target_minus_class46_margin": score_by_class[target_id] - score_by_class[46],
    }


def _parse_args(project_root: Path) -> argparse.Namespace:
    analysis_root = (
        project_root
        / "experiments/01_fudd_clip_cub/analysis/01_single_image_prompt_pilot"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-manifest", type=Path, default=analysis_root / "sample.json")
    parser.add_argument(
        "--prompt-override",
        type=Path,
        default=analysis_root / "pair_46_47_override.json",
    )
    parser.add_argument("--data-root", type=Path, default=project_root / "data")
    parser.add_argument(
        "--prompt-root",
        type=Path,
        default=project_root / "data/fudd_official",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=project_root / ".cache/fudd_clip_cub",
    )
    parser.add_argument(
        "--parent-runs-root",
        type=Path,
        default=project_root / "experiments/01_fudd_clip_cub/runs",
    )
    parser.add_argument("--runs-root", type=Path, default=analysis_root / "runs")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--text-batch-size", type=int, default=256)
    return parser.parse_args()


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    args = _parse_args(project_root)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    if args.text_batch_size <= 0:
        raise ValueError("--text-batch-size must be positive")

    sample = _load_json(args.sample_manifest)
    overrides = load_pair_overrides(args.prompt_override)
    _validate_override_asset(overrides)
    source_digest = _source_digest(project_root)
    override_file_digest = _sha256(args.prompt_override)
    created = _utc_now()
    run_id = (
        created.strftime("%Y%m%dT%H%M%S.%fZ")
        + f"-pair46_47-{source_digest[:10]}-{override_file_digest[:10]}"
    )
    run_dir = args.runs_root.expanduser().resolve() / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    state_path = run_dir / "run_state.json"
    initial_state = {
        "schema_version": 1,
        "run_id": run_id,
        "status": "RUNNING",
        "created_at_utc": created.isoformat(),
        "source_digest": source_digest,
        "prompt_override_sha256": override_file_digest,
    }
    _write_json(state_path, initial_state)

    try:
        parent_dir = args.parent_runs_root / sample["parent_run"]["run_id"]
        parent_record = _validate_parent_run(parent_dir, sample)
        official = load_official_prompts(args.prompt_root)
        custom = official.with_pair_overrides(overrides)
        changed_pairs = [
            key for key in official.pairs if official.pairs[key] != custom.pairs[key]
        ]
        if changed_pairs != [OVERRIDE_PAIR]:
            raise RuntimeError(f"Unexpected changed prompt pairs: {changed_pairs}")

        torch.manual_seed(SEED)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(SEED)

        backend = CLIPBackend(
            model_name=MODEL_NAME,
            device=args.device,
            precision=MODEL_PRECISION,
            text_batch_size=args.text_batch_size,
            model_cache_dir=args.cache_dir / "models",
            text_cache_path=args.cache_dir / "text_features/single-image-pilot.pt",
        )
        dataset = prepare_cub(
            args.data_root,
            transform=backend.preprocess,
            download=True,
            verify_images=False,
            split="test",
        )
        validate_class_name_alignment(dataset.class_names, official.class_names)
        sample_index = int(sample["sample_index"])
        metadata = dataset.samples[sample_index]
        actual_identity = (
            metadata.image_id,
            metadata.relative_path.as_posix(),
            metadata.target,
        )
        expected_identity = (
            sample["image_id"],
            sample["relative_path"],
            sample["target_class_id"],
        )
        if actual_identity != expected_identity:
            raise RuntimeError("CUB metadata does not match the selected sample manifest")
        image_path = (
            args.data_root / "CUB_200_2011/images" / metadata.relative_path
        )
        if _sha256(image_path) != sample["image_sha256"]:
            raise RuntimeError("Selected image checksum does not match the manifest")

        image, target = dataset[sample_index]
        if target != sample["target_class_id"]:
            raise RuntimeError("Loaded image label does not match the manifest")
        image_features = backend.encode_images(image.unsqueeze(0)).detach().cpu()
        if not math.isclose(float(image_features.norm(dim=1).item()), 1.0, abs_tol=1e-5):
            raise RuntimeError("CLIP image feature is not L2-normalised")

        baseline_groups = official.single_template_prompts()
        baseline_features = backend.pool_prompt_groups(baseline_groups)
        baseline_scores_tensor = (
            image_features.to(backend.device) @ baseline_features.t()
        ).squeeze(0)
        baseline_ranking_tensor = ordered_predictions(baseline_scores_tensor).cpu()
        baseline_ranking = [int(value) for value in baseline_ranking_tensor.tolist()]
        baseline_scores = [float(value) for value in baseline_scores_tensor.cpu().tolist()]
        candidates = baseline_ranking_tensor[:TOP_K].unsqueeze(0).contiguous()
        candidate_ids = [int(value) for value in candidates[0].tolist()]
        expected_candidates = [item["class_id"] for item in sample["baseline_top10"]]
        if candidate_ids != expected_candidates:
            raise RuntimeError(
                f"Recomputed baseline Top-10 differs from parent: {candidate_ids}"
            )

        official_groups = official.prompts_for_candidates(candidate_ids)
        custom_groups = custom.prompts_for_candidates(candidate_ids)
        followup_texts = tuple(
            dict.fromkeys(
                text
                for groups in (official_groups, custom_groups)
                for group in groups
                for text in group
            )
        )
        followup_table = backend.build_text_feature_table(followup_texts)
        official_result = rerank_candidates(
            image_features,
            candidates,
            official,
            backend,
            text_table=followup_table,
            batch_size=1,
        )
        custom_result = rerank_candidates(
            image_features,
            candidates,
            custom,
            backend,
            text_table=followup_table,
            batch_size=1,
        )
        backend.save_text_cache()

        official_ranking = [
            int(value) for value in official_result.ranked_class_ids[0].tolist()
        ]
        custom_ranking = [
            int(value) for value in custom_result.ranked_class_ids[0].tolist()
        ]
        official_scores = [float(value) for value in official_result.scores[0].tolist()]
        custom_scores = [float(value) for value in custom_result.scores[0].tolist()]
        if official_ranking != sample["fudd_ranking"]:
            raise RuntimeError(
                f"Recomputed official FuDD differs from parent: {official_ranking}"
            )

        changed_group_ids = [
            class_id
            for class_id, before, after in zip(
                candidate_ids, official_groups, custom_groups, strict=True
            )
            if before != after
        ]
        candidate_pairs = {
            tuple(sorted(pair)) for pair in combinations(candidate_ids, 2)
        }
        unchanged_candidate_pairs = sum(
            official.pairs[key] == custom.pairs[key] for key in candidate_pairs
        )
        unaffected_positions = [
            index
            for index, class_id in enumerate(candidate_ids)
            if class_id not in OVERRIDE_PAIR
        ]
        max_unaffected_score_difference = max(
            abs(official_scores[index] - custom_scores[index])
            for index in unaffected_positions
        )
        checks = {
            "parent_checksums_valid": True,
            "sample_identity_matches_parent": parent_record["image_id"] == sample["image_id"],
            "image_sha256_matches": _sha256(image_path) == sample["image_sha256"],
            "image_encoded_once": True,
            "baseline_top10_matches_parent": candidate_ids == expected_candidates,
            "official_fudd_ranking_matches_parent": (
                official_ranking == sample["fudd_ranking"]
            ),
            "same_baseline_candidates_for_both_rerankers": (
                official_result.candidate_class_ids.tolist()
                == custom_result.candidate_class_ids.tolist()
                == candidates.tolist()
            ),
            "candidate_set_preserved": (
                set(official_ranking) == set(custom_ranking) == set(candidate_ids)
            ),
            "official_pair_count_preserved": official.pair_count == custom.pair_count == 19_900,
            "only_pair_46_47_changed_globally": changed_pairs == [OVERRIDE_PAIR],
            "forty_four_of_forty_five_candidate_pairs_unchanged": (
                len(candidate_pairs) == 45 and unchanged_candidate_pairs == 44
            ),
            "only_candidate_46_and_47_prompt_groups_changed": (
                changed_group_ids == [46, 47]
            ),
            "prompt_group_lengths_preserved": (
                [len(group) for group in official_groups]
                == [len(group) for group in custom_groups]
            ),
            "unaffected_candidate_scores_identical": (
                max_unaffected_score_difference <= 1e-7
            ),
            "scores_are_finite": all(
                math.isfinite(value)
                for value in baseline_scores + official_scores + custom_scores
            ),
            "no_score_fusion": True,
        }
        if not all(checks.values()):
            failed = [name for name, passed in checks.items() if not passed]
            raise RuntimeError(f"Pilot integrity checks failed: {failed}")

        target_id = int(sample["target_class_id"])
        methods = {
            "baseline": _method_payload(
                ranking_scope="all_200_classes",
                ranked_ids=baseline_ranking,
                candidate_ids=list(range(official.class_count)),
                scores=baseline_scores,
                class_names=official.class_names,
                target_id=target_id,
            ),
            "official_fudd": _method_payload(
                ranking_scope="baseline_top10_only",
                ranked_ids=official_ranking,
                candidate_ids=candidate_ids,
                scores=official_scores,
                class_names=official.class_names,
                target_id=target_id,
            ),
            "pair_46_47_override": _method_payload(
                ranking_scope="baseline_top10_only",
                ranked_ids=custom_ranking,
                candidate_ids=candidate_ids,
                scores=custom_scores,
                class_names=official.class_names,
                target_id=target_id,
            ),
        }
        prediction = {
            "schema_version": 1,
            "sample": {
                "sample_index": sample_index,
                "image_id": metadata.image_id,
                "relative_path": metadata.relative_path.as_posix(),
                "image_sha256": sample["image_sha256"],
                "target_class_id": target_id,
                "target_class_name": official.class_names[target_id],
            },
            "candidate_class_ids": candidate_ids,
            "methods": methods,
        }

        outcome = {
            "baseline_top1_class_id": baseline_ranking[0],
            "official_fudd_top1_class_id": official_ranking[0],
            "override_top1_class_id": custom_ranking[0],
            "baseline_target_rank": methods["baseline"]["target_rank"],
            "official_fudd_target_rank": methods["official_fudd"]["target_rank"],
            "override_target_rank": methods["pair_46_47_override"]["target_rank"],
            "rescued_vs_baseline": custom_ranking[0] == target_id,
            "rescued_vs_official_fudd": custom_ranking[0] == target_id,
            "official_target_minus_class46_margin": methods["official_fudd"][
                "target_minus_class46_margin"
            ],
            "override_target_minus_class46_margin": methods["pair_46_47_override"][
                "target_minus_class46_margin"
            ],
        }
        environment = _environment()
        completed = _utc_now()
        result = {
            "schema_version": 1,
            "run_id": run_id,
            "status": "PASS",
            "created_at_utc": created.isoformat(),
            "completed_at_utc": completed.isoformat(),
            "scientific_scope": {
                "name": "single_selected_image_posthoc_diagnostic",
                "selection_was_posthoc": True,
                "prompt_designer_had_target_label": True,
                "generator_blinded_to_ground_truth": False,
                "generalization_claim_allowed": False,
            },
            "protocol": {
                "model_name": MODEL_NAME,
                "precision": MODEL_PRECISION,
                "top_k": TOP_K,
                "seed": SEED,
                "baseline_template": "a photo of a {}.",
                "intervention": (
                    "replace only official pair (46, 47), four prompts for four prompts"
                ),
                "candidate_pair_count": len(candidate_pairs),
                "unchanged_candidate_pair_count": unchanged_candidate_pairs,
                "score_fusion": False,
                "image_encoder_passes": 1,
            },
            "source": {
                "project_source_sha256": source_digest,
                "openai_clip_commit": OPENAI_CLIP_COMMIT,
                "official_fudd_commit": FUDD_COMMIT,
                "official_prompt_digest": official.source_digest,
                "derived_prompt_digest": custom.source_digest,
                "prompt_override_file_sha256": override_file_digest,
                "sample_manifest_sha256": _sha256(args.sample_manifest),
                "parent_run_id": parent_dir.name,
                "parent_predictions_sha256": _sha256(parent_dir / "predictions.jsonl"),
                "parent_result_sha256": _sha256(parent_dir / "result.json"),
                "dataset_fingerprint": dataset.fingerprint,
            },
            "environment": environment,
            "prompt_audit": {
                "changed_global_pairs": [list(pair) for pair in changed_pairs],
                "changed_candidate_group_ids": changed_group_ids,
                "official_group_lengths": [len(group) for group in official_groups],
                "override_group_lengths": [len(group) for group in custom_groups],
                "max_unaffected_score_difference": max_unaffected_score_difference,
            },
            "checks": checks,
            "outcome": outcome,
            "prediction_file": "prediction.json",
        }

        shutil.copyfile(args.sample_manifest, run_dir / "sample_snapshot.json")
        shutil.copyfile(args.prompt_override, run_dir / "prompt_override.json")
        _write_json(run_dir / "prediction.json", prediction)
        _write_json(run_dir / "result.json", result)
        (run_dir / "environment_pip_freeze.txt").write_text(
            subprocess.run(
                [sys.executable, "-m", "pip", "freeze"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout,
            encoding="utf-8",
        )
        _write_json(
            state_path,
            {
                **initial_state,
                "status": "PASS",
                "completed_at_utc": completed.isoformat(),
                "result": "result.json",
            },
            overwrite=True,
        )
        checksum_path = _write_checksums(run_dir)
    except BaseException as error:
        _write_json(
            state_path,
            {
                **initial_state,
                "status": "FAILED",
                "failed_at_utc": _utc_now().isoformat(),
                "error_type": type(error).__name__,
                "error": str(error),
            },
            overwrite=True,
        )
        raise

    print(json.dumps(outcome, indent=2, sort_keys=True), flush=True)
    print(f"PASS: {run_dir}", flush=True)
    print(f"Checksums: {checksum_path}", flush=True)


if __name__ == "__main__":
    main()
