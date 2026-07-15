"""Fail-closed creation of immutable experiment run artifacts."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import torch


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def create_run_directory(
    runs_root: Path | str,
    config: dict[str, Any],
    *,
    kind: str,
) -> Path:
    """Create a timestamped directory whose suffix commits to its config."""

    root = Path(runs_root).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    canonical = json.dumps(config, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode()).hexdigest()[:10]
    timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    output = root / f"{timestamp}-{kind}-{digest}"
    output.mkdir()
    return output


def write_json(path: Path | str, value: Any) -> Path:
    output = Path(path)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite artifact: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f"{output.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    return output


def write_jsonl(path: Path | str, rows: Iterable[dict[str, Any]]) -> Path:
    output = Path(path)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite artifact: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f"{output.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            for row in rows:
                handle.write(
                    json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                    + "\n"
                )
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    return output


def atomic_torch_save(value: Any, path: Path | str) -> Path:
    output = Path(path)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite artifact: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f"{output.name}.tmp-{os.getpid()}")
    try:
        torch.save(value, temporary)
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    return output


def write_checksums(run_directory: Path | str) -> Path:
    root = Path(run_directory)
    output = root / "checksums.sha256"
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite artifact: {output}")
    lines = [
        f"{sha256_file(path)}  {path.relative_to(root).as_posix()}"
        for path in sorted(root.rglob("*"))
        if path.is_file() and path != output
    ]
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output
