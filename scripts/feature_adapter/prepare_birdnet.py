#!/usr/bin/env python3
"""Prepare the licensed BirdNET+ Taxonomy image source for BirdMix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ttvr.data.birdnet import prepare_birdnet_training_images


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--max-samples", type=int)
    args = parser.parse_args()
    report = prepare_birdnet_training_images(
        args.root,
        workers=args.workers,
        max_samples=args.max_samples,
    )
    print(json.dumps(report.to_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
