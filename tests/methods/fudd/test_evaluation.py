from __future__ import annotations

import json
from pathlib import Path

import pytest

from ttvr.methods.fudd.evaluation import (
    EvaluationReport,
    FuDDMetrics,
    ParityReport,
    PredictionRecord,
)
from ttvr.metrics import TopKAccuracy, TransferCounts


def test_evaluation_report_is_json_serialisable() -> None:
    predictions = tuple(
        PredictionRecord(
            sample_index=index,
            image_id=index + 1,
            relative_path=f"class/image_{index}.jpg",
            target_class_id=target,
            baseline_topk_class_ids=baseline,
            fudd_ranked_class_ids=fudd,
        )
        for index, (target, baseline, fudd) in enumerate(
            (
                (0, (0, 1, 2, 3, 4), (0, 2, 1, 3, 4)),
                (1, (0, 1, 2, 3, 4), (1, 0, 2, 3, 4)),
                (2, (2, 1, 3, 4, 5), (2, 1, 3, 4, 5)),
                (9, (5, 6, 7, 8, 9), (5, 6, 7, 8, 9)),
            )
        )
    )
    report = EvaluationReport(
        config={"model_name": "ViT-L/14@336px", "top_k": 10, "seed": 2026},
        num_samples=4,
        baseline=TopKAccuracy(total=4, top1_correct=2, top5_correct=4),
        fudd=FuDDMetrics(
            total=4,
            top1_correct=3,
            top5_correct=4,
            candidate_hits=4,
        ),
        transfers=TransferCounts(
            total=4,
            both_correct=2,
            recovered=1,
            degraded=0,
            both_wrong=1,
        ),
        parity=ParityReport(
            samples=4,
            matching_predictions=4,
            max_abs_prototype_difference=1e-7,
        ),
        prompt_digest="prompt-sha256",
        dataset_fingerprint="dataset-sha256",
        feature_dtype="torch.float32",
        predictions=predictions,
    )

    payload = report.to_dict()

    assert json.loads(json.dumps(payload)) == payload
    assert payload["baseline"]["top1"] == 50.0
    assert payload["fudd"]["top1"] == 75.0
    assert payload["fudd"]["candidate_recall"] == 100.0
    assert payload["transfers"]["recovered"] == 1
    assert payload["parity"]["passed"]
    assert payload["prediction_count"] == 4


def test_prediction_jsonl_is_complete_hashed_and_non_overwriting(
    tmp_path: Path,
) -> None:
    record = PredictionRecord(
        sample_index=0,
        image_id=17,
        relative_path="001.Bird/example.jpg",
        target_class_id=3,
        baseline_topk_class_ids=(4, 3, 2, 1, 0),
        fudd_ranked_class_ids=(3, 4, 2, 1, 0),
    )
    report = EvaluationReport(
        config={"top_k": 5},
        num_samples=1,
        baseline=TopKAccuracy(total=1, top1_correct=0, top5_correct=1),
        fudd=FuDDMetrics(
            total=1,
            top1_correct=1,
            top5_correct=1,
            candidate_hits=1,
        ),
        transfers=TransferCounts(
            total=1,
            both_correct=0,
            recovered=1,
            degraded=0,
            both_wrong=0,
        ),
        parity=ParityReport(
            samples=1,
            matching_predictions=1,
            max_abs_prototype_difference=0.0,
        ),
        prompt_digest="prompt",
        dataset_fingerprint="dataset",
        feature_dtype="torch.float32",
        predictions=(record,),
    )
    output = tmp_path / "predictions.jsonl"

    path, digest = report.write_predictions_jsonl(output)

    assert path == output
    assert len(digest) == 64
    assert json.loads(output.read_text()) == record.to_dict()
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        report.write_predictions_jsonl(output)


def test_fudd_metrics_require_candidate_recall_to_cover_top5() -> None:
    with pytest.raises(ValueError, match="Candidate hits"):
        FuDDMetrics(
            total=10,
            top1_correct=3,
            top5_correct=8,
            candidate_hits=7,
        )
