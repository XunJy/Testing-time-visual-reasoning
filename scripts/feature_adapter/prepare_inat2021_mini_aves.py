#!/usr/bin/env python3
"""Extract and manifest the official iNaturalist 2021 Mini bird subset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ttvr.data.inat2021 import prepare_inat2021_mini_aves


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--birdnet-csv", type=Path, required=True)
    parser.add_argument("--validation-per-class", type=int, default=5)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    report = prepare_inat2021_mini_aves(
        args.root,
        birdnet_csv_path=args.birdnet_csv,
        validation_per_class=args.validation_per_class,
        workers=args.workers,
    )
    print(json.dumps(report.to_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
