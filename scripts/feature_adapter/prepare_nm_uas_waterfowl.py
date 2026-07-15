#!/usr/bin/env python3
"""Prepare audited expert-consensus crops from NM UAS Waterfowl."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ttvr.data.bird_manifest import load_samples
from ttvr.data.cub_taxonomy import build_cub_birdnet_crosswalk
from ttvr.data.nm_uas_waterfowl import (
    audit_nm_uas_archive,
    audit_nm_uas_strict_cub,
    download_nm_uas_metadata,
    plan_nm_uas_waterfowl,
    prepare_nm_uas_waterfowl,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/nm_uas_waterfowl"))
    parser.add_argument("--birdnet-csv", type=Path, required=True)
    parser.add_argument(
        "--metadata",
        type=Path,
        help="Existing expert refined JSON; otherwise fetch only its locked ZIP byte range",
    )
    parser.add_argument(
        "--archive",
        type=Path,
        help="Existing official 322-MB archive; otherwise download it during preparation",
    )
    parser.add_argument("--context-scale", type=float, default=1.25)
    parser.add_argument("--keep-source-images", action="store_true")
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Audit the 224-KB expert metadata/taxonomy without downloading the archive",
    )
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="Audit an existing full archive without extracting crops or writing manifests",
    )
    parser.add_argument(
        "--cub-class-names",
        type=Path,
        help="Print a non-mutating strict-CUB retention audit",
    )
    args = parser.parse_args()
    if args.plan_only and args.audit_only:
        parser.error("--plan-only and --audit-only are mutually exclusive")

    metadata = args.metadata or download_nm_uas_metadata(args.root)
    plan = plan_nm_uas_waterfowl(
        metadata,
        birdnet_csv_path=args.birdnet_csv,
    )
    if args.plan_only:
        output: dict[str, object] = {
            "plan": {
                "birdnet_csv_sha256": plan.birdnet_csv_sha256,
                "clipped_bboxes": plan.clipped_bbox_count,
                "coarse_consensus_boxes_omitted": plan.coarse_crop_count,
                "expert_images": len(plan.images),
                "expert_specific_consensus_crops": len(plan.crops),
                "metadata_sha256": plan.metadata_sha256,
                "taxa": len(plan.taxa),
            }
        }
        audit_rows = plan.crops
    elif args.audit_only:
        if args.archive is None:
            parser.error("--audit-only requires --archive")
        output = {"archive_audit": audit_nm_uas_archive(plan, args.archive).to_dict()}
        audit_rows = plan.crops
    else:
        report = prepare_nm_uas_waterfowl(
            args.root,
            birdnet_csv_path=args.birdnet_csv,
            metadata_path=metadata,
            archive_path=args.archive,
            context_scale=args.context_scale,
            keep_source_images=args.keep_source_images,
            progress=lambda completed, total, sample_id: print(
                f"[{completed}/{total}] {sample_id}", flush=True
            ),
        )
        output = {"preparation": report.to_dict()}
        audit_rows = load_samples(args.root / "manifests" / "samples.jsonl")

    if args.cub_class_names is not None:
        crosswalk = build_cub_birdnet_crosswalk(args.cub_class_names, args.birdnet_csv)
        excluded = crosswalk.excluded_birdnet_ids(range(len(crosswalk.entries)))
        output["strict_cub_audit"] = audit_nm_uas_strict_cub(
            audit_rows,
            excluded,
        ).to_dict()
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
