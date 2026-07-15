#!/usr/bin/env python3
"""Run the FuDD + OpenAI CLIP combination on CUB into an immutable directory."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch

from ttvr import FuDDConfig, run_clip_cub_experiment

OPENAI_CLIP_COMMIT = "a1d071733d7111c9c014f024669f959182114e33"
FUDD_COMMIT = "32264231fec047eb0bbbf59bfdbc8e6d208a096b"
PAPER_TARGET = {"baseline_top1": 63.48, "fudd_top1": 65.90, "gain_pp": 2.42}
PRIOR_REPRODUCTION = {
    "baseline_top1": 63.36,
    "fudd_top1": 65.74,
    "gain_pp": 2.38,
}


def _utc_now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def _source_digest(project_root: Path) -> str:
    files = [project_root / "pyproject.toml", Path(__file__).resolve()]
    shared_root = project_root / "src" / "ttvr"
    files.extend(
        [
            shared_root / "__init__.py",
            shared_root / "metrics.py",
            shared_root / "models" / "__init__.py",
            shared_root / "models" / "base.py",
            shared_root / "models" / "clip.py",
        ]
    )
    files.extend((shared_root / "data").rglob("*.py"))
    files.extend((shared_root / "methods" / "fudd").rglob("*.py"))
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
        "ftfy": _package_version("ftfy"),
        "cuda_available": cuda_available,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu": torch.cuda.get_device_name(0) if cuda_available else None,
        "cuda_device_count": torch.cuda.device_count() if cuda_available else 0,
        "cuda_matmul_allow_tf32": (
            torch.backends.cuda.matmul.allow_tf32 if cuda_available else None
        ),
        "cudnn_allow_tf32": (torch.backends.cudnn.allow_tf32 if cuda_available else None),
    }


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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


class ProgressPrinter:
    """Print bounded progress updates without requiring a notebook UI."""

    def __init__(self) -> None:
        self._last: dict[tuple[str, int], int] = {}

    def __call__(self, stage: str, completed: int, total: int) -> None:
        key = (stage, total)
        interval = max(1, total // 20) if total else 1
        previous = self._last.get(key, -interval)
        if completed == total or completed - previous >= interval:
            percent = 100.0 if total == 0 else 100.0 * completed / total
            print(
                f"[{stage}] {completed:,}/{total:,} ({percent:5.1f}%)",
                flush=True,
            )
            self._last[key] = completed


def _parse_args(project_root: Path) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Deterministic test prefix for a smoke run; omit for all 5,794 images.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--text-batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--parity-samples", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=project_root / "data",
    )
    parser.add_argument(
        "--prompt-root",
        type=Path,
        default=project_root / "data" / "fudd_official",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=project_root / ".cache" / "fudd_clip_cub",
    )
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=project_root / "experiments" / "01_fudd_clip_cub" / "runs",
    )
    return parser.parse_args()


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    args = _parse_args(project_root)
    if args.max_samples is not None and args.max_samples <= 0:
        raise ValueError("--max-samples must be positive")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")

    source_digest = _source_digest(project_root)
    created = _utc_now()
    run_kind = "full" if args.max_samples is None else f"smoke{args.max_samples}"
    run_id = created.strftime("%Y%m%dT%H%M%S.%fZ") + f"-{run_kind}-{source_digest[:10]}"
    run_dir = args.runs_root.expanduser().resolve() / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    config = FuDDConfig(
        data_root=args.data_root,
        prompt_root=args.prompt_root,
        cache_dir=args.cache_dir,
        model_name="ViT-L/14@336px",
        precision="fp32",
        top_k=10,
        batch_size=args.batch_size,
        text_batch_size=args.text_batch_size,
        num_workers=args.num_workers,
        device=args.device,
        seed=2026,
    )
    state_path = run_dir / "run_state.json"
    initial_state = {
        "run_id": run_id,
        "status": "RUNNING",
        "created_at_utc": created.isoformat(),
        "source_digest": source_digest,
        "config": config.to_dict(),
        "max_samples": args.max_samples,
        "parity_samples": args.parity_samples,
    }
    _write_json(state_path, initial_state)

    try:
        environment = _environment()
        print(json.dumps(environment, indent=2), flush=True)
        report = run_clip_cub_experiment(
            config,
            max_samples=args.max_samples,
            parity_samples=args.parity_samples,
            download_data=True,
            download_prompts=True,
            verify_images=True,
            progress=ProgressPrinter(),
        )
        predictions_path, predictions_sha256 = report.write_predictions_jsonl(
            run_dir / "predictions.jsonl"
        )
        summary = report.to_dict()
        baseline_top1 = float(summary["baseline"]["top1"])
        fudd_top1 = float(summary["fudd"]["top1"])
        gain_pp = fudd_top1 - baseline_top1
        checks = {
            "official_protocol_locked": config.is_official_clip_reproduction,
            "prediction_count_matches": summary["prediction_count"] == summary["num_samples"],
            "official_reference_parity": bool(summary["parity"]["passed"]),
        }
        if args.max_samples is None:
            tolerance_pp = 0.50
            checks.update(
                {
                    "official_test_size": summary["num_samples"] == 5_794,
                    "positive_net_gain": gain_pp > 0.0,
                    "baseline_within_0.50pp_of_prior": abs(
                        baseline_top1 - PRIOR_REPRODUCTION["baseline_top1"]
                    )
                    <= tolerance_pp,
                    "fudd_within_0.50pp_of_prior": abs(fudd_top1 - PRIOR_REPRODUCTION["fudd_top1"])
                    <= tolerance_pp,
                    "gain_within_0.50pp_of_paper": abs(gain_pp - PAPER_TARGET["gain_pp"])
                    <= tolerance_pp,
                }
            )
        else:
            checks["smoke_sample_count"] = summary["num_samples"] == args.max_samples
        status_prefix = "PASS" if all(checks.values()) else "REVIEW"
        status = f"{run_kind.upper()}_{status_prefix}"

        freeze_path = run_dir / "environment_pip_freeze.txt"
        freeze_path.write_text(
            subprocess.run(
                [sys.executable, "-m", "pip", "freeze"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout,
            encoding="utf-8",
        )
        completed = _utc_now()
        result = {
            "schema_version": 1,
            "run_id": run_id,
            "run_kind": run_kind,
            "status": status,
            "created_at_utc": created.isoformat(),
            "completed_at_utc": completed.isoformat(),
            "source": {
                "project_source_sha256": source_digest,
                "openai_clip_commit": OPENAI_CLIP_COMMIT,
                "official_fudd_commit": FUDD_COMMIT,
            },
            "environment": environment,
            "paper_target_percent": PAPER_TARGET,
            "prior_reproduction_percent": PRIOR_REPRODUCTION,
            "gain_pp": gain_pp,
            "checks": checks,
            "predictions": {
                "path": predictions_path.name,
                "sha256": predictions_sha256,
                "rows": summary["prediction_count"],
            },
            "evaluation": summary,
        }
        result_path = run_dir / "result.json"
        _write_json(result_path, result)
        _write_json(
            state_path,
            {
                **initial_state,
                "status": status,
                "completed_at_utc": completed.isoformat(),
                "result": result_path.name,
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

    print(
        f"{status}: baseline={baseline_top1:.4f}% FuDD={fudd_top1:.4f}% gain={gain_pp:+.4f} pp",
        flush=True,
    )
    print(f"Run directory: {run_dir}", flush=True)
    print(f"Checksums: {checksum_path}", flush=True)


if __name__ == "__main__":
    main()
