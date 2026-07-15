#!/usr/bin/env python3
"""Cache frozen OpenAI CLIP image features for both official CUB splits."""

from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path

import torch

from ttvr import CLIPBackend, prepare_cub


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-")


def _cache_path(root: Path, split: str, dataset_fingerprint: str, backend: CLIPBackend) -> Path:
    identity = hashlib.sha256(backend.cache_identity.encode()).hexdigest()[:16]
    model = _slug(backend.model_name)
    precision = _slug(f"{backend.precision}-{backend.feature_dtype_name}")
    return root / (f"cub-{split}-{model}-{identity}-{precision}-{dataset_fingerprint[:16]}-all.pt")


class ProgressPrinter:
    def __init__(self) -> None:
        self._last: dict[tuple[str, int], int] = {}

    def __call__(self, stage: str, completed: int, total: int) -> None:
        key = (stage, total)
        previous = self._last.get(key, 0)
        if completed == total or completed - previous >= 256:
            print(f"[{stage}] {completed:,}/{total:,}", flush=True)
            self._last[key] = completed


def _parse_args(project_root: Path) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=project_root / "data")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=project_root / ".cache/fudd_clip_cub",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    args = _parse_args(project_root)
    torch.manual_seed(2026)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(2026)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    backend = CLIPBackend(
        model_name="ViT-L/14@336px",
        device=args.device,
        precision="fp32",
        text_batch_size=256,
        model_cache_dir=args.cache_dir / "models",
    )
    feature_root = args.cache_dir / "image_features"
    for split in ("train", "test"):
        dataset = prepare_cub(
            args.data_root,
            transform=backend.preprocess,
            download=True,
            verify_images=False,
            split=split,
        )
        path = _cache_path(feature_root, split, dataset.fingerprint, backend)
        result = backend.encode_dataset(
            dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            seed=2026,
            cache_path=path,
            cache_tag=f"cub:{split}:{dataset.fingerprint}:{backend.cache_identity}",
            progress=ProgressPrinter(),
        )
        print(
            f"CACHED split={split} shape={tuple(result.features.shape)} path={path}",
            flush=True,
        )


if __name__ == "__main__":
    main()
