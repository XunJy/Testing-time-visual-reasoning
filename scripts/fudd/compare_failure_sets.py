#!/usr/bin/env python3
"""Export readable failure sets from two immutable FuDD prediction runs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

CATEGORIES = ("both_correct", "recovered", "degraded", "both_wrong")
LABEL_PATTERN = re.compile(r"[a-z0-9][a-z0-9_-]*")


@dataclass(frozen=True, slots=True)
class PredictionRun:
    """Validated predictions keyed by the shared CUB sample index."""

    label: str
    run_dir: Path
    predictions_sha256: str
    rows: dict[int, dict[str, Any]]


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _category(record: dict[str, Any]) -> str:
    baseline = record["baseline_top1_correct"]
    fudd = record["fudd_top1_correct"]
    if baseline and fudd:
        return "both_correct"
    if not baseline and fudd:
        return "recovered"
    if baseline and not fudd:
        return "degraded"
    return "both_wrong"


def _required_int(record: dict[str, Any], field: str, line_number: int) -> int:
    value = record.get(field)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} at line {line_number} must be an integer")
    return value


def _required_bool(record: dict[str, Any], field: str, line_number: int) -> bool:
    value = record.get(field)
    if type(value) is not bool:
        raise ValueError(f"{field} at line {line_number} must be a JSON boolean")
    return value


def _ranked_ids(
    record: dict[str, Any],
    field: str,
    line_number: int,
    class_count: int,
) -> list[int]:
    values = record.get(field)
    if not isinstance(values, list) or not values:
        raise ValueError(f"{field} at line {line_number} must be a non-empty list")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < class_count
        for value in values
    ):
        raise ValueError(f"{field} at line {line_number} contains an invalid class id")
    if len(values) != len(set(values)):
        raise ValueError(f"{field} at line {line_number} contains duplicate class ids")
    return values


def _load_run(label: str, run_dir: Path, class_count: int) -> PredictionRun:
    resolved = run_dir.expanduser().resolve(strict=True)
    if not resolved.is_dir():
        raise ValueError(f"Run path is not a directory: {resolved}")
    predictions_path = resolved / "predictions.jsonl"
    raw = predictions_path.read_bytes()
    if not raw:
        raise ValueError(f"Prediction file is empty: {predictions_path}")

    rows: dict[int, dict[str, Any]] = {}
    for line_number, line in enumerate(raw.decode("utf-8").splitlines(), start=1):
        if not line.strip():
            raise ValueError(f"Blank JSONL record at line {line_number} in {label}")
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid JSON at line {line_number} in {label}") from error
        if not isinstance(record, dict):
            raise ValueError(f"Record at line {line_number} in {label} must be an object")

        sample_index = _required_int(record, "sample_index", line_number)
        image_id = _required_int(record, "image_id", line_number)
        target = _required_int(record, "target_class_id", line_number)
        baseline_top1 = _required_int(record, "baseline_top1_class_id", line_number)
        fudd_top1 = _required_int(record, "fudd_top1_class_id", line_number)
        for value, field in (
            (target, "target_class_id"),
            (baseline_top1, "baseline_top1_class_id"),
            (fudd_top1, "fudd_top1_class_id"),
        ):
            if not 0 <= value < class_count:
                raise ValueError(f"{field} at line {line_number} is outside the class table")
        relative_path = record.get("relative_path")
        if not isinstance(relative_path, str) or not relative_path.strip():
            raise ValueError(f"relative_path at line {line_number} must be non-empty")
        baseline_correct = _required_bool(record, "baseline_top1_correct", line_number)
        fudd_correct = _required_bool(record, "fudd_top1_correct", line_number)
        target_in_topk = _required_bool(record, "target_in_baseline_topk", line_number)
        baseline_topk = _ranked_ids(record, "baseline_topk_class_ids", line_number, class_count)
        fudd_ranked = _ranked_ids(record, "fudd_ranked_class_ids", line_number, class_count)

        if baseline_top1 != baseline_topk[0] or fudd_top1 != fudd_ranked[0]:
            raise ValueError(f"Top-1 field disagrees with ranking at line {line_number}")
        if baseline_correct != (baseline_top1 == target):
            raise ValueError(f"baseline correctness is inconsistent at line {line_number}")
        if fudd_correct != (fudd_top1 == target):
            raise ValueError(f"FuDD correctness is inconsistent at line {line_number}")
        if target_in_topk != (target in baseline_topk):
            raise ValueError(f"target_in_baseline_topk is inconsistent at line {line_number}")
        if set(baseline_topk) != set(fudd_ranked):
            raise ValueError(
                f"FuDD candidates differ from baseline candidates at line {line_number}"
            )
        if sample_index in rows:
            raise ValueError(f"Duplicate sample_index {sample_index} in {label}")
        if image_id <= 0:
            raise ValueError(f"image_id at line {line_number} must be positive")
        rows[sample_index] = record

    expected_indices = set(range(len(rows)))
    if set(rows) != expected_indices:
        raise ValueError(f"{label} sample indices must be contiguous from zero")
    return PredictionRun(
        label=label,
        run_dir=resolved,
        predictions_sha256=_sha256_bytes(raw),
        rows=rows,
    )


def _target_rank(record: dict[str, Any], field: str) -> int | str:
    values = record[field]
    target = record["target_class_id"]
    return values.index(target) + 1 if target in values else ""


def _image_row(
    run: PredictionRun,
    record: dict[str, Any],
    class_names: list[str],
) -> dict[str, Any]:
    target = record["target_class_id"]
    baseline = record["baseline_top1_class_id"]
    fudd = record["fudd_top1_class_id"]
    return {
        "model": run.label,
        "category": _category(record),
        "sample_index": record["sample_index"],
        "image_id": record["image_id"],
        "relative_path": record["relative_path"],
        "target_class_id": target,
        "target_class_name": class_names[target],
        "baseline_top1_class_id": baseline,
        "baseline_top1_class_name": class_names[baseline],
        "fudd_top1_class_id": fudd,
        "fudd_top1_class_name": class_names[fudd],
        "baseline_target_rank": _target_rank(record, "baseline_topk_class_ids"),
        "fudd_target_rank": _target_rank(record, "fudd_ranked_class_ids"),
        "target_in_baseline_top10": record["target_in_baseline_topk"],
        "top1_changed_by_fudd": baseline != fudd,
    }


def _top_target_classes(
    records: list[dict[str, Any]],
    class_names: list[str],
    limit: int = 12,
) -> list[dict[str, Any]]:
    counts = Counter(record["target_class_id"] for record in records)
    return [
        {"class_id": class_id, "class_name": class_names[class_id], "count": count}
        for class_id, count in counts.most_common(limit)
    ]


def _run_summary(run: PredictionRun, class_names: list[str]) -> dict[str, Any]:
    category_records = {
        category: [record for record in run.rows.values() if _category(record) == category]
        for category in CATEGORIES
    }
    total = len(run.rows)
    baseline_correct = len(category_records["both_correct"]) + len(category_records["degraded"])
    both_wrong = category_records["both_wrong"]
    degraded = category_records["degraded"]
    counts = {category: len(records) for category, records in category_records.items()}
    return {
        "run_dir": str(run.run_dir),
        "predictions_sha256": run.predictions_sha256,
        "total": total,
        "counts": counts,
        "percentages": {category: 100.0 * count / total for category, count in counts.items()},
        "degraded_percent_of_all": 100.0 * len(degraded) / total,
        "degraded_percent_of_baseline_correct": 100.0 * len(degraded) / baseline_correct,
        "both_wrong_details": {
            "target_in_baseline_top10": sum(
                record["target_in_baseline_topk"] for record in both_wrong
            ),
            "target_outside_baseline_top10": sum(
                not record["target_in_baseline_topk"] for record in both_wrong
            ),
            "top1_changed_by_fudd": sum(
                record["baseline_top1_class_id"] != record["fudd_top1_class_id"]
                for record in both_wrong
            ),
            "same_wrong_top1": sum(
                record["baseline_top1_class_id"] == record["fudd_top1_class_id"]
                for record in both_wrong
            ),
        },
        "top_degraded_target_classes": _top_target_classes(degraded, class_names),
        "top_both_wrong_target_classes": _top_target_classes(both_wrong, class_names),
    }


def _comparison_row(
    first: PredictionRun,
    second: PredictionRun,
    sample_index: int,
    class_names: list[str],
) -> dict[str, Any]:
    first_record = first.rows[sample_index]
    second_record = second.rows[sample_index]
    target = first_record["target_class_id"]
    row: dict[str, Any] = {
        "sample_index": sample_index,
        "image_id": first_record["image_id"],
        "relative_path": first_record["relative_path"],
        "target_class_id": target,
        "target_class_name": class_names[target],
        f"{first.label}_category": _category(first_record),
        f"{second.label}_category": _category(second_record),
    }
    for run, record in ((first, first_record), (second, second_record)):
        baseline = record["baseline_top1_class_id"]
        fudd = record["fudd_top1_class_id"]
        row[f"{run.label}_baseline_top1_class_id"] = baseline
        row[f"{run.label}_baseline_top1_class_name"] = class_names[baseline]
        row[f"{run.label}_fudd_top1_class_id"] = fudd
        row[f"{run.label}_fudd_top1_class_name"] = class_names[fudd]
        row[f"{run.label}_baseline_target_rank"] = _target_rank(record, "baseline_topk_class_ids")
        row[f"{run.label}_fudd_target_rank"] = _target_rank(record, "fudd_ranked_class_ids")
    return row


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write an empty CSV: {path.name}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _readme(summary: dict[str, Any]) -> str:
    first_label, second_label = summary["labels"]
    first = summary["runs"][first_label]
    second = summary["runs"][second_label]
    overlap = summary["cross_model"]["same_category_overlap"]
    lines = [
        "# FuDD 两次实验失败图片清单",
        "",
        "定义：`degraded` 表示 baseline 正确、FuDD 错误；`both_wrong` 表示同一模型的",
        "baseline 和 FuDD 都错误。所有计数均来自完整 5,794 张 CUB 测试集。",
        "",
        "| 模型 | FuDD 改错 | baseline/FuDD 都错 | FuDD 恢复 | 两者都对 |",
        "|---|---:|---:|---:|---:|",
        (
            f"| {first_label} | {first['counts']['degraded']:,} | "
            f"{first['counts']['both_wrong']:,} | {first['counts']['recovered']:,} | "
            f"{first['counts']['both_correct']:,} |"
        ),
        (
            f"| {second_label} | {second['counts']['degraded']:,} | "
            f"{second['counts']['both_wrong']:,} | {second['counts']['recovered']:,} | "
            f"{second['counts']['both_correct']:,} |"
        ),
        "",
        "## 跨模型重合",
        "",
        "| 类别 | 两模型交集 | 仅第一个模型 | 仅第二个模型 | Jaccard |",
        "|---|---:|---:|---:|---:|",
    ]
    for category in ("degraded", "both_wrong"):
        values = overlap[category]
        lines.append(
            f"| {category} | {values['intersection']:,} | {values['first_only']:,} | "
            f"{values['second_only']:,} | {values['jaccard_percent']:.4f}% |"
        )
    cross = summary["cross_model"]
    lines.extend(
        [
            "",
            f"四种配置全部错误的图片共 **{cross['all_four_wrong']:,}** 张；两个模型的 "
            f"baseline 都错为 {cross['both_models_baseline_wrong']:,} 张，两个模型的 FuDD "
            f"都错为 {cross['both_models_fudd_wrong']:,} 张。",
            "",
            "## 清单",
            "",
            f"- [`{first_label}_degraded.csv`]({first_label}_degraded.csv)",
            f"- [`{first_label}_both_wrong.csv`]({first_label}_both_wrong.csv)",
            f"- [`{second_label}_degraded.csv`]({second_label}_degraded.csv)",
            f"- [`{second_label}_both_wrong.csv`]({second_label}_both_wrong.csv)",
            "- [`degraded_in_both_models.csv`](degraded_in_both_models.csv)",
            "- [`both_wrong_in_both_models.csv`](both_wrong_in_both_models.csv)",
            "- [`cross_model_categories.csv`](cross_model_categories.csv)",
            "- [`summary.json`](summary.json)",
            "",
            "CSV 同时给出图片相对路径、真实类别、baseline/FuDD Top-1 类别，以及真实类别",
            "在两个 Top-10 排名中的位置。原始 run 未被修改。",
            "",
        ]
    )
    return "\n".join(lines)


def _build_report(
    runs: tuple[PredictionRun, PredictionRun],
    class_names: list[str],
    output_dir: Path,
    class_names_sha256: str,
) -> None:
    first, second = runs
    first_indices = set(first.rows)
    if first_indices != set(second.rows):
        raise ValueError("The two runs do not contain the same sample indices")
    for sample_index in sorted(first_indices):
        first_record = first.rows[sample_index]
        second_record = second.rows[sample_index]
        identity = ("image_id", "relative_path", "target_class_id")
        if any(first_record[field] != second_record[field] for field in identity):
            raise ValueError(f"Run identity mismatch at sample_index {sample_index}")

    labels = (first.label, second.label)
    run_summaries = {run.label: _run_summary(run, class_names) for run in runs}
    category_sets = {
        run.label: {
            category: {index for index, record in run.rows.items() if _category(record) == category}
            for category in CATEGORIES
        }
        for run in runs
    }
    overlap: dict[str, Any] = {}
    for category in CATEGORIES:
        first_set = category_sets[first.label][category]
        second_set = category_sets[second.label][category]
        union = first_set | second_set
        overlap[category] = {
            "intersection": len(first_set & second_set),
            "first_only": len(first_set - second_set),
            "second_only": len(second_set - first_set),
            "union": len(union),
            "jaccard_percent": 100.0 * len(first_set & second_set) / len(union),
        }

    transition = {
        first_category: {
            second_category: len(
                category_sets[first.label][first_category]
                & category_sets[second.label][second_category]
            )
            for second_category in CATEGORIES
        }
        for first_category in CATEGORIES
    }
    summary = {
        "schema_version": 1,
        "labels": list(labels),
        "class_names_sha256": class_names_sha256,
        "runs": run_summaries,
        "cross_model": {
            "same_category_overlap": overlap,
            "category_transition_matrix": transition,
            "both_models_baseline_wrong": sum(
                not first.rows[index]["baseline_top1_correct"]
                and not second.rows[index]["baseline_top1_correct"]
                for index in first_indices
            ),
            "both_models_fudd_wrong": sum(
                not first.rows[index]["fudd_top1_correct"]
                and not second.rows[index]["fudd_top1_correct"]
                for index in first_indices
            ),
            "all_four_wrong": len(
                category_sets[first.label]["both_wrong"] & category_sets[second.label]["both_wrong"]
            ),
        },
    }

    resolved_output = output_dir.expanduser().absolute()
    if resolved_output.exists():
        raise FileExistsError(f"Refusing to overwrite report: {resolved_output}")
    temporary = resolved_output.with_name(f".{resolved_output.name}.tmp-{os.getpid()}")
    temporary.mkdir(parents=True, exist_ok=False)
    try:
        for run in runs:
            for category in ("degraded", "both_wrong"):
                records = [
                    _image_row(run, run.rows[index], class_names)
                    for index in sorted(category_sets[run.label][category])
                ]
                _write_csv(temporary / f"{run.label}_{category}.csv", records)

        comparison_rows = [
            _comparison_row(first, second, index, class_names) for index in sorted(first_indices)
        ]
        _write_csv(temporary / "cross_model_categories.csv", comparison_rows)
        for category, filename in (
            ("degraded", "degraded_in_both_models.csv"),
            ("both_wrong", "both_wrong_in_both_models.csv"),
        ):
            indices = sorted(
                category_sets[first.label][category] & category_sets[second.label][category]
            )
            _write_csv(
                temporary / filename,
                [comparison_rows[index] for index in indices],
            )

        (temporary / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (temporary / "README.md").write_text(_readme(summary), encoding="utf-8")
        checksum_lines = []
        for path in sorted(temporary.iterdir()):
            if path.is_file():
                checksum_lines.append(f"{_sha256_bytes(path.read_bytes())}  {path.name}")
        (temporary / "checksums.sha256").write_text(
            "\n".join(checksum_lines) + "\n", encoding="utf-8"
        )
        resolved_output.parent.mkdir(parents=True, exist_ok=True)
        temporary.replace(resolved_output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _parse_run_spec(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Run must be formatted as label=/path/to/run")
    label, raw_path = value.split("=", 1)
    if LABEL_PATTERN.fullmatch(label) is None:
        raise argparse.ArgumentTypeError(
            "Run label must contain only lowercase letters, digits, underscores, or hyphens"
        )
    if not raw_path:
        raise argparse.ArgumentTypeError("Run path must not be empty")
    return label, Path(raw_path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        type=_parse_run_spec,
        help="Repeat twice: label=/path/to/immutable/run",
    )
    parser.add_argument("--class-names", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if len(args.run) != 2:
        raise SystemExit("error: exactly two --run arguments are required")
    if args.run[0][0] == args.run[1][0]:
        raise SystemExit("error: run labels must be unique")
    try:
        class_names_raw = args.class_names.expanduser().resolve(strict=True).read_bytes()
        class_names = json.loads(class_names_raw.decode("utf-8"))
        if (
            not isinstance(class_names, list)
            or not class_names
            or any(not isinstance(name, str) or not name.strip() for name in class_names)
        ):
            raise ValueError("Class names must be a non-empty JSON string list")
        runs = tuple(_load_run(label, path, len(class_names)) for label, path in args.run)
        _build_report(
            runs,  # type: ignore[arg-type]
            class_names,
            args.output_dir,
            _sha256_bytes(class_names_raw),
        )
        print(args.output_dir.expanduser().absolute())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise SystemExit(f"error: {error}") from error


if __name__ == "__main__":
    main()
