#!/usr/bin/env python3
"""Prepare the checksum-locked USGS aerial-avian publisher crops."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ttvr.data.bird_manifest import load_samples
from ttvr.data.cub_taxonomy import build_cub_birdnet_crosswalk
from ttvr.data.usgs_aerial_avian import (
    audit_usgs_aerial_archives,
    audit_usgs_aerial_strict_cub,
    download_usgs_aerial_annotations,
    plan_usgs_aerial_avian,
    prepare_usgs_aerial_avian,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/usgs_aerial_avian"))
    parser.add_argument("--birdnet-csv", type=Path, required=True)
    parser.add_argument(
        "--annotations",
        type=Path,
        help="Existing official annotations.zip; otherwise download the 45-KB file",
    )
    parser.add_argument(
        "--images",
        type=Path,
        help="Existing official images.zip; otherwise download it during preparation",
    )
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Audit small annotations/taxonomy without downloading images.zip",
    )
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="Audit both existing archives without extracting images or writing manifests",
    )
    parser.add_argument(
        "--cub-class-names",
        type=Path,
        help="Print a non-mutating strict-CUB taxon retention audit",
    )
    args = parser.parse_args()
    if args.plan_only and args.audit_only:
        parser.error("--plan-only and --audit-only are mutually exclusive")

    annotations = args.annotations or download_usgs_aerial_annotations(args.root)
    plan = plan_usgs_aerial_avian(
        annotations,
        birdnet_csv_path=args.birdnet_csv,
    )
    if args.plan_only:
        output: dict[str, object] = {
            "plan": {
                "annotations_md5": plan.annotations_md5,
                "birdnet_csv_sha256": plan.birdnet_csv_sha256,
                "coarse_crops_omitted": len(plan.all_rows) - len(plan.specific_rows),
                "publisher_rows": len(plan.all_rows),
                "source_groups": len({row.source_frame_id for row in plan.specific_rows}),
                "specific_crops": len(plan.specific_rows),
                "specific_split_counts": plan.specific_split_counts,
                "taxa": len(plan.taxa),
            }
        }
        audit_rows = plan.specific_rows
    elif args.audit_only:
        if args.images is None:
            parser.error("--audit-only requires --images")
        output = {"archive_audit": audit_usgs_aerial_archives(plan, args.images).to_dict()}
        audit_rows = plan.specific_rows
    else:
        report = prepare_usgs_aerial_avian(
            args.root,
            birdnet_csv_path=args.birdnet_csv,
            annotations_path=annotations,
            images_path=args.images,
            progress=lambda completed, total, sample_id: print(
                f"[{completed}/{total}] {sample_id}", flush=True
            ),
        )
        output = {"preparation": report.to_dict()}
        audit_rows = load_samples(args.root / "manifests" / "samples.jsonl")

    if args.cub_class_names is not None:
        crosswalk = build_cub_birdnet_crosswalk(args.cub_class_names, args.birdnet_csv)
        excluded = crosswalk.excluded_birdnet_ids(range(len(crosswalk.entries)))
        output["strict_cub_audit"] = audit_usgs_aerial_strict_cub(
            audit_rows,
            excluded,
        ).to_dict()
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
