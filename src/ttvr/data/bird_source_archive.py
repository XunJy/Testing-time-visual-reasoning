"""Fail-closed archive and publication helpers for audited bird sources.

Dataset modules keep their source-specific schema checks.  This module only
provides the security-sensitive mechanics shared by those modules: verified
HTTPS downloads, canonical ZIP-member validation, no-replace extraction, and
atomic publication of a complete manifest directory.
"""

from __future__ import annotations

import os
import shutil
import stat
import tempfile
import time
import urllib.error
import urllib.request
import zipfile
import zlib
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from .bird_crops import md5_file, validate_relative_posix_path
from .bird_manifest import write_json, write_jsonl


def contained_output_path(root: Path | str, relative_path: str) -> Path:
    """Return a contained output path and reject every existing symlink component."""

    validate_relative_posix_path(relative_path)
    resolved_root = Path(root).expanduser().resolve()
    unresolved = resolved_root / relative_path
    current = resolved_root
    for part in Path(relative_path).parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"output path contains a symlink: {relative_path}")
    candidate = unresolved.resolve()
    try:
        candidate.relative_to(resolved_root)
    except ValueError as error:
        raise ValueError(f"output path escapes dataset root: {relative_path}") from error
    return unresolved


def atomic_publish_file_no_replace(temporary: Path, destination: Path) -> None:
    """Publish one same-filesystem file without replacing a concurrent winner."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(temporary, destination)
    except FileExistsError as error:
        raise FileExistsError(f"Refusing to overwrite file: {destination}") from error


def stream_download_verified(
    url: str,
    destination: Path | str,
    *,
    expected_md5: str,
    expected_size: int,
    allowed_hosts: Iterable[str],
    attempts: int = 4,
    timeout: int = 120,
) -> bool:
    """Materialize one exact publisher file; return ``True`` when it was reused."""

    if attempts <= 0 or timeout <= 0:
        raise ValueError("attempts and timeout must be positive")
    hosts = frozenset(allowed_hosts)
    parsed = urlsplit(url)
    if parsed.scheme != "https" or parsed.hostname not in hosts:
        raise ValueError(f"Refusing download from unexpected URL: {url}")
    if (
        len(expected_md5) != 32
        or any(character not in "0123456789abcdef" for character in expected_md5)
        or isinstance(expected_size, bool)
        or not isinstance(expected_size, int)
        or expected_size <= 0
    ):
        raise ValueError("expected_md5 and expected_size are invalid")
    output = Path(destination).expanduser()
    if output.is_symlink():
        raise ValueError(f"download destination must not be a symlink: {output}")
    if output.exists():
        if not output.is_file():
            raise RuntimeError(f"download destination is not a regular file: {output}")
        actual_md5 = md5_file(output)
        if output.stat().st_size != expected_size or actual_md5 != expected_md5:
            raise RuntimeError(
                f"Existing publisher file mismatch for {output.name}: "
                f"size={output.stat().st_size}, md5={actual_md5}"
            )
        return True

    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{output.name}.part-", dir=output.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        for attempt in range(attempts):
            request = urllib.request.Request(url, headers={"User-Agent": "ttvr-birdmix/1.0"})
            try:
                with urllib.request.urlopen(request, timeout=timeout) as response:
                    final = urlsplit(response.geturl())
                    if final.scheme != "https" or final.hostname not in hosts:
                        raise RuntimeError(
                            f"Publisher redirected to an unapproved host: {response.geturl()}"
                        )
                    with temporary.open("wb") as handle:
                        shutil.copyfileobj(response, handle, length=1024 * 1024)
                break
            except urllib.error.HTTPError as error:
                retryable = error.code in {408, 425, 429} or error.code >= 500
                if not retryable or attempt + 1 == attempts:
                    raise
            except (TimeoutError, urllib.error.URLError):
                if attempt + 1 == attempts:
                    raise
            time.sleep(2**attempt)
        actual_size = temporary.stat().st_size
        actual_md5 = md5_file(temporary)
        if actual_size != expected_size or actual_md5 != expected_md5:
            raise RuntimeError(
                f"Downloaded publisher file mismatch for {output.name}: "
                f"expected size={expected_size}, md5={expected_md5}; "
                f"found size={actual_size}, md5={actual_md5}"
            )
        try:
            atomic_publish_file_no_replace(temporary, output)
        except FileExistsError as error:
            # A concurrent winner is acceptable only if it is the same object.
            if (
                output.is_symlink()
                or not output.is_file()
                or output.stat().st_size != expected_size
                or md5_file(output) != expected_md5
            ):
                raise RuntimeError(
                    f"Concurrent publisher file mismatch: {output}"
                ) from error
        return False
    finally:
        temporary.unlink(missing_ok=True)


def validated_zip_members(
    archive_path: Path | str,
    *,
    expected_files: Iterable[str] | None = None,
) -> dict[str, zipfile.ZipInfo]:
    """Validate every ZIP path/type and optionally require an exact file set."""

    archive = Path(archive_path).expanduser()
    if archive.is_symlink() or not archive.is_file():
        raise ValueError(f"archive must be a regular non-symlink file: {archive}")
    members: dict[str, zipfile.ZipInfo] = {}
    with zipfile.ZipFile(archive) as bundle:
        for info in bundle.infolist():
            raw_name = info.filename
            canonical = raw_name[:-1] if info.is_dir() and raw_name.endswith("/") else raw_name
            validate_relative_posix_path(canonical)
            if raw_name in members:
                raise RuntimeError(f"ZIP contains duplicate member: {raw_name}")
            unix_mode = info.external_attr >> 16
            if stat.S_ISLNK(unix_mode):
                raise RuntimeError(f"ZIP contains a symbolic link: {raw_name}")
            if info.flag_bits & 0x1:
                raise RuntimeError(f"ZIP contains an encrypted member: {raw_name}")
            if not info.is_dir() and info.compress_type not in {
                zipfile.ZIP_STORED,
                zipfile.ZIP_DEFLATED,
            }:
                raise RuntimeError(f"ZIP member uses an unsupported compression method: {raw_name}")
            if info.file_size < 0 or info.compress_size < 0:
                raise RuntimeError(f"ZIP member has a negative size: {raw_name}")
            members[raw_name] = info
    files = {name for name, info in members.items() if not info.is_dir()}
    if expected_files is not None:
        expected = set(expected_files)
        missing = sorted(expected - files)
        extra = sorted(files - expected)
        if missing or extra:
            raise RuntimeError(
                "ZIP file-set mismatch: "
                f"missing={missing[:5]} ({len(missing)}), extra={extra[:5]} ({len(extra)})"
            )
    return members


def _crc32_file(path: Path) -> int:
    value = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value = zlib.crc32(chunk, value)
    return value & 0xFFFFFFFF


def extract_zip_member_no_replace(
    archive_path: Path | str,
    info: zipfile.ZipInfo,
    destination: Path | str,
) -> bool:
    """Extract one validated regular member and return whether it was reused."""

    if info.is_dir():
        raise ValueError("cannot extract a directory as a file")
    output = Path(destination).expanduser()
    if output.is_symlink():
        raise ValueError(f"extraction destination must not be a symlink: {output}")

    def _matches() -> bool:
        return (
            output.is_file()
            and not output.is_symlink()
            and output.stat().st_size == info.file_size
            and _crc32_file(output) == info.CRC
        )

    if output.exists():
        if not _matches():
            raise RuntimeError(f"Existing extracted member mismatch: {output}")
        return True
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{output.name}.part-", dir=output.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with zipfile.ZipFile(Path(archive_path).expanduser()) as bundle:
            current = bundle.getinfo(info.filename)
            if (
                current.CRC != info.CRC
                or current.file_size != info.file_size
                or current.compress_size != info.compress_size
            ):
                raise RuntimeError(f"ZIP member metadata changed: {info.filename}")
            with bundle.open(current) as source, temporary.open("wb") as target:
                shutil.copyfileobj(source, target, length=1024 * 1024)
        if temporary.stat().st_size != info.file_size or _crc32_file(temporary) != info.CRC:
            raise RuntimeError(f"Extracted member checksum mismatch: {info.filename}")
        try:
            atomic_publish_file_no_replace(temporary, output)
        except FileExistsError as error:
            if not _matches():
                raise RuntimeError(
                    f"Concurrent extracted member mismatch: {output}"
                ) from error
        return False
    finally:
        temporary.unlink(missing_ok=True)


def publish_manifest_bundle(
    root: Path | str,
    *,
    jsonl_files: Mapping[str, Iterable[dict[str, Any]]],
    json_files: Mapping[str, Any],
) -> None:
    """Atomically publish a complete, immutable ``manifests`` directory."""

    root_path = Path(root).expanduser().resolve()
    destination = root_path / "manifests"
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Refusing to replace manifest directory: {destination}")
    root_path.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".manifests.stage-", dir=root_path))
    lock = root_path / ".manifests.publish.lock"
    descriptor: int | None = None
    try:
        for name, rows in jsonl_files.items():
            validate_relative_posix_path(name)
            if "/" in name:
                raise ValueError("manifest bundle filenames must not contain directories")
            write_jsonl(stage / name, rows)
        for name, value in json_files.items():
            validate_relative_posix_path(name)
            if "/" in name:
                raise ValueError("manifest bundle filenames must not contain directories")
            write_json(stage / name, value)
        try:
            descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError as error:
            raise FileExistsError(
                f"Another manifest publisher holds the no-replace lock: {lock}"
            ) from error
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(f"Refusing to replace manifest directory: {destination}")
        os.rename(stage, destination)
    finally:
        if descriptor is not None:
            os.close(descriptor)
            lock.unlink(missing_ok=True)
        if stage.exists():
            shutil.rmtree(stage)
