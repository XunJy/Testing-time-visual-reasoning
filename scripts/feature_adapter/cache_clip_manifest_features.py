#!/usr/bin/env python3
"""Cache OpenAI CLIP features for one locked bird manifest dataset."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import re
from pathlib import Path

from ttvr.data.bird_manifest import ManifestBirdDataset, load_samples, load_taxa
from ttvr.gpu_lock import DEFAULT_GPU_LOCK_PATH, GPULockBusyError, exclusive_gpu_lock
from ttvr.models.clip import CLIPBackend


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-")


class ProgressPrinter:
    def __init__(self) -> None:
        self.previous = 0

    def __call__(self, stage: str, completed: int, total: int) -> None:
        if completed == total or completed - self.previous >= 256:
            print(f"[{stage}] {completed:,}/{total:,}", flush=True)
            self.previous = completed


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--taxa", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--model-cache-dir", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", default=["train", "validation"])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument(
        "--shard-size",
        type=int,
        default=2_048,
        help=(
            "Atomically persist this many rows per resumable shard; completed "
            "shards are reused after a runtime interruption"
        ),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip-image-verification", action="store_true")
    parser.add_argument(
        "--gpu-lock",
        type=Path,
        default=DEFAULT_GPU_LOCK_PATH,
        help=(
            "Fail-fast advisory lock shared with the formal BirdMix matrix "
            f"(default: {DEFAULT_GPU_LOCK_PATH})"
        ),
    )
    parser.add_argument(
        "--no-gpu-lock",
        action="store_true",
        help="Unsafe advanced/testing opt-out; never use for formal cache generation",
    )
    return parser.parse_args()


def cache_manifest_features(args: argparse.Namespace) -> list[dict[str, str | int]]:
    """Encode all requested splits after the caller has enforced GPU exclusion."""

    samples = load_samples(args.samples)
    taxa = load_taxa(args.taxa)
    available_splits = {sample.source_split for sample in samples}
    requested = [split for split in args.splits if split in available_splits]
    if not requested:
        raise RuntimeError(f"None of the requested splits exist: {sorted(available_splits)}")

    backend = CLIPBackend(
        device=args.device,
        precision="fp32",
        model_cache_dir=args.model_cache_dir,
    )
    identity = hashlib.sha256(backend.cache_identity.encode()).hexdigest()[:16]
    outputs: list[dict[str, str | int]] = []
    for split in requested:
        dataset = ManifestBirdDataset(
            args.root,
            samples,
            taxa,
            transform=backend.preprocess,
            splits=(split,),
            verify_images=not args.skip_image_verification,
        )
        name = (
            f"{_slug(dataset.dataset_id)}-{_slug(split)}-{_slug(backend.model_name)}-"
            f"{identity}-fp32-{dataset.fingerprint[:16]}.pt"
        )
        cache_path = args.cache_dir / "image_features" / name
        cache_tag = f"{dataset.dataset_id}:{split}:{dataset.fingerprint}:{backend.cache_identity}"
        features = backend.encode_dataset(
            dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            cache_path=cache_path,
            cache_tag=cache_tag,
            cache_shard_size=args.shard_size,
            progress=ProgressPrinter(),
        )
        outputs.append(
            {
                "dataset_id": dataset.dataset_id,
                "split": split,
                "samples": features.size,
                "classes": len(dataset.taxon_ids),
                "dimensions": features.features.shape[1],
                "dataset_fingerprint": dataset.fingerprint,
                "cache_path": str(cache_path),
            }
        )
    return outputs


def main() -> None:
    args = _parse_args()
    lock: contextlib.AbstractContextManager[None]
    if args.no_gpu_lock:
        lock = contextlib.nullcontext()
    else:
        lock = exclusive_gpu_lock(
            args.gpu_lock,
            purpose="CLIP manifest feature caching",
        )
    try:
        with lock:
            outputs = cache_manifest_features(args)
    except GPULockBusyError as error:
        raise SystemExit(str(error)) from error
    print(json.dumps(outputs, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
