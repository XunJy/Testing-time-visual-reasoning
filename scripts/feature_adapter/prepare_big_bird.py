#!/usr/bin/env python3
"""Prepare audited Big Bird species crops without downloading generic-bird images."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ttvr.data.big_bird import (
    audit_big_bird_strict_cub,
    download_big_bird_metadata,
    plan_big_bird,
    prepare_big_bird,
)
from ttvr.data.bird_manifest import load_samples
from ttvr.data.cub_taxonomy import build_cub_birdnet_crosswalk


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/big_bird"))
    parser.add_argument("--birdnet-csv", type=Path, required=True)
    parser.add_argument(
        "--metadata",
        type=Path,
        help="Reuse an official metadata file; otherwise stream it from LILA before images.",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--context-scale", type=float, default=1.25)
    parser.add_argument("--keep-source-images", action="store_true")
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Validate metadata/taxonomy and print counts without downloading source images.",
    )
    parser.add_argument(
        "--cub-class-names",
        type=Path,
        help="Print a non-mutating strict-CUB retention audit after planning/preparation.",
    )
    args = parser.parse_args()

    if args.plan_only:
        metadata = args.metadata or download_big_bird_metadata(args.root)
        plan = plan_big_bird(metadata, birdnet_csv_path=args.birdnet_csv)
        output: dict[str, object] = {
            "plan": {
                "canonical_taxa": len(plan.taxa),
                "generic_bird_crops_omitted": plan.generic_bird_crop_count,
                "metadata_sha256": plan.metadata_sha256,
                "source_categories": plan.specific_source_category_count,
                "source_images_to_download": len(plan.images),
                "specific_crops": len(plan.crops),
            }
        }
        audit_rows = plan.crops
    else:
        report = prepare_big_bird(
            args.root,
            birdnet_csv_path=args.birdnet_csv,
            metadata_path=args.metadata,
            workers=args.workers,
            context_scale=args.context_scale,
            keep_source_images=args.keep_source_images,
            progress=lambda completed, total, image_id: print(
                f"[{completed}/{total}] {image_id}", flush=True
            ),
        )
        output = {"preparation": report.to_dict()}
        audit_rows = load_samples(args.root / "manifests" / "samples.jsonl")

    if args.cub_class_names is not None:
        crosswalk = build_cub_birdnet_crosswalk(args.cub_class_names, args.birdnet_csv)
        excluded = crosswalk.excluded_birdnet_ids(range(len(crosswalk.entries)))
        output["strict_cub_audit"] = audit_big_bird_strict_cub(
            audit_rows,
            excluded,
        ).to_dict()
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
