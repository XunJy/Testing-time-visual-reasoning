#!/usr/bin/env python3
"""Analyze paired Top-1 predictions without modifying an immutable run."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import torch

from ttvr.metrics import exact_mcnemar_test, paired_bootstrap_accuracy_gain


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="Immutable run containing predictions.jsonl")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--reps", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument("--chunk-size", type=int, default=256)
    return parser.parse_args()


def _read_predictions(path: Path) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    raw = path.read_bytes()
    if not raw:
        raise ValueError(f"Prediction file is empty: {path}")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"Prediction file is not valid UTF-8: {path}") from error

    baseline: list[bool] = []
    comparison: list[bool] = []
    sample_indices: set[int] = set()
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            raise ValueError(f"Blank JSONL record at line {line_number}")
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid JSON at line {line_number}: {error.msg}") from error
        if not isinstance(record, dict):
            raise ValueError(f"JSONL record at line {line_number} must be an object")
        sample_index = record.get("sample_index")
        if isinstance(sample_index, bool) or not isinstance(sample_index, int):
            raise ValueError(f"sample_index at line {line_number} must be an integer")
        if sample_index in sample_indices:
            raise ValueError(f"Duplicate sample_index {sample_index} at line {line_number}")
        sample_indices.add(sample_index)
        for field in ("baseline_top1_correct", "fudd_top1_correct"):
            if type(record.get(field)) is not bool:
                raise ValueError(f"{field} at line {line_number} must be a JSON boolean")
        baseline.append(record["baseline_top1_correct"])
        comparison.append(record["fudd_top1_correct"])

    if not baseline:
        raise ValueError(f"Prediction file has no JSONL records: {path}")
    metadata = {
        "prediction_file": path.name,
        "prediction_sha256": hashlib.sha256(raw).hexdigest(),
        "rows": len(baseline),
        "sample_index_unique": True,
        "sample_index_min": min(sample_indices),
        "sample_index_max": max(sample_indices),
    }
    return (
        torch.tensor(baseline, dtype=torch.bool),
        torch.tensor(comparison, dtype=torch.bool),
        metadata,
    )


def _default_output(run_dir: Path) -> Path:
    if run_dir.parent.name != "runs":
        raise ValueError("Default output requires a run path shaped as <experiment>/runs/<run-id>")
    return run_dir.parent.parent / "analysis" / f"{run_dir.name}.json"


def _write_json_exclusive(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.unlink(missing_ok=True)
    try:
        temporary.write_text(
            json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.link(temporary, path)
    except FileExistsError as error:
        raise FileExistsError(f"Refusing to overwrite analysis: {path}") from error
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    args = _parse_args()
    try:
        run_dir = args.run_dir.expanduser().resolve(strict=True)
        if not run_dir.is_dir():
            raise ValueError(f"Run path is not a directory: {run_dir}")
        predictions_path = run_dir / "predictions.jsonl"
        baseline, comparison, input_metadata = _read_predictions(predictions_path)
        output = (
            args.output.expanduser().absolute()
            if args.output is not None
            else _default_output(run_dir)
        )
        if output == run_dir or run_dir in output.parents:
            raise ValueError("Analysis output must be outside the immutable run directory")

        mcnemar = exact_mcnemar_test(baseline, comparison)
        bootstrap = paired_bootstrap_accuracy_gain(
            baseline,
            comparison,
            confidence_level=args.confidence_level,
            reps=args.reps,
            seed=args.seed,
            chunk_size=args.chunk_size,
        )
        result = {
            "schema_version": 1,
            "run_id": run_dir.name,
            "comparison": {
                "baseline_field": "baseline_top1_correct",
                "comparison_field": "fudd_top1_correct",
                "unit": "percentage_points",
            },
            "input": input_metadata,
            "mcnemar": mcnemar.to_dict(),
            "bootstrap": bootstrap.to_dict(),
            "runtime": {"torch_version": torch.__version__},
        }
        _write_json_exclusive(output, result)
        print(output)
    except (OSError, ValueError) as error:
        raise SystemExit(f"error: {error}") from error


if __name__ == "__main__":
    main()
