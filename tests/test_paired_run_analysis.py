from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "fudd" / "analyze_paired_run.py"


def _run_cli(run_dir: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(PROJECT_ROOT / "src")
    return subprocess.run(
        [sys.executable, str(SCRIPT), str(run_dir), *arguments],
        cwd=PROJECT_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )


def test_cli_writes_analysis_next_to_runs_without_mutating_run(tmp_path: Path) -> None:
    run_dir = tmp_path / "experiment" / "runs" / "run-001"
    run_dir.mkdir(parents=True)
    records = [
        {"sample_index": 3, "baseline_top1_correct": True, "fudd_top1_correct": True},
        {"sample_index": 7, "baseline_top1_correct": False, "fudd_top1_correct": True},
        {"sample_index": 9, "baseline_top1_correct": True, "fudd_top1_correct": False},
    ]
    prediction_bytes = b"".join(
        (json.dumps(record, separators=(",", ":")) + "\n").encode() for record in records
    )
    (run_dir / "predictions.jsonl").write_bytes(prediction_bytes)
    before = {path.name: path.read_bytes() for path in run_dir.iterdir()}

    completed = _run_cli(run_dir, "--reps", "31", "--seed", "5", "--chunk-size", "4")

    assert completed.returncode == 0, completed.stderr
    output = tmp_path / "experiment" / "analysis" / "run-001.json"
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["input"]["prediction_sha256"] == hashlib.sha256(prediction_bytes).hexdigest()
    assert result["input"]["rows"] == 3
    assert result["mcnemar"]["recovered"] == 1
    assert result["mcnemar"]["degraded"] == 1
    assert result["bootstrap"]["reps"] == 31
    assert result["bootstrap"]["seed"] == 5
    assert result["runtime"]["torch_version"]
    assert {path.name: path.read_bytes() for path in run_dir.iterdir()} == before


def test_cli_rejects_duplicate_sample_indices_and_non_boolean_fields(tmp_path: Path) -> None:
    run_dir = tmp_path / "experiment" / "runs" / "bad-run"
    run_dir.mkdir(parents=True)
    (run_dir / "predictions.jsonl").write_text(
        "\n".join(
            [
                '{"sample_index":1,"baseline_top1_correct":true,"fudd_top1_correct":false}',
                '{"sample_index":1,"baseline_top1_correct":false,"fudd_top1_correct":true}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    completed = _run_cli(run_dir, "--reps", "2")

    assert completed.returncode != 0
    assert "Duplicate sample_index" in completed.stderr
    assert not (tmp_path / "experiment" / "analysis").exists()

    boolean_run = tmp_path / "experiment" / "runs" / "bad-boolean"
    boolean_run.mkdir()
    (boolean_run / "predictions.jsonl").write_text(
        '{"sample_index":2,"baseline_top1_correct":1,"fudd_top1_correct":true}\n',
        encoding="utf-8",
    )

    boolean_result = _run_cli(boolean_run, "--reps", "2")

    assert boolean_result.returncode != 0
    assert "must be a JSON boolean" in boolean_result.stderr
