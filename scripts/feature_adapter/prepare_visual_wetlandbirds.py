#!/usr/bin/env python3
"""Prepare audited stride-10 crops from Visual WetlandBirds v4 videos."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from ttvr.data.visual_wetlandbirds import prepare_visual_wetlandbirds


def _progress(value: dict[str, Any]) -> None:
    print(json.dumps(value, ensure_ascii=False, sort_keys=True), file=sys.stderr, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Verify ROOT/videos.zip, download only the small official metadata "
            "when missing, and create immutable stride-10 bird crops."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("data/visual_wetlandbirds"),
        help="Dataset root containing the manually downloaded videos.zip",
    )
    parser.add_argument("--birdnet-csv", type=Path, required=True)
    parser.add_argument(
        "--no-download-metadata",
        action="store_true",
        help="Require all five small Zenodo metadata files to already exist",
    )
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    args = parser.parse_args()

    report = prepare_visual_wetlandbirds(
        args.root,
        birdnet_csv_path=args.birdnet_csv,
        download_metadata=not args.no_download_metadata,
        ffmpeg_binary=args.ffmpeg,
        ffprobe_binary=args.ffprobe,
        progress=_progress,
    )
    print(json.dumps(report.to_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
