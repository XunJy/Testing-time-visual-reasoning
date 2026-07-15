from __future__ import annotations

import io
import urllib.error
from pathlib import Path

import pytest

from ttvr.data.birdnet import (
    _download,
    _image_provenance,
    _provenance_matches,
    license_permits_noncommercial_training,
)


@pytest.mark.parametrize(
    "value",
    [
        "cc0",
        "CC0",
        "CC BY 4.0",
        "cc-by-nc",
        "cc-by-sa",
        "cc-by-nc-sa",
        "pd",
        "Public domain",
        "gfdl",
    ],
)
def test_noncommercial_license_filter_accepts_supported_licenses(value: str) -> None:
    assert license_permits_noncommercial_training(value)


@pytest.mark.parametrize(
    "value",
    [
        "",
        "© Macaulay Library",
        "cc-by-nd",
        "CC BY-NC-ND 4.0",
        "CC BY No Derivatives",
        "all rights reserved",
        "cc-by-something-else",
    ],
)
def test_noncommercial_license_filter_rejects_unsupported_licenses(value: str) -> None:
    assert not license_permits_noncommercial_training(value)


def test_download_rejects_non_https_and_non_birdnet_hosts(tmp_path: Path) -> None:
    for url in (
        "file:///etc/passwd",
        "http://birdnet.cornell.edu/taxonomy/api/download/csv",
        "https://example.com/image.jpg",
    ):
        with pytest.raises(ValueError, match="non-BirdNET"):
            _download(url, tmp_path / "download")


def test_download_retries_transient_network_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def fake_urlopen(*_args: object, **_kwargs: object) -> io.BytesIO:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise urllib.error.URLError("temporary outage")
        return io.BytesIO(b"downloaded")

    monkeypatch.setattr("ttvr.data.birdnet.urllib.request.urlopen", fake_urlopen)
    monkeypatch.setattr("ttvr.data.birdnet.time.sleep", lambda _seconds: None)
    destination = tmp_path / "image.bin"

    _download(
        "https://birdnet.cornell.edu/taxonomy/api/image/example",
        destination,
    )

    assert calls == 2
    assert destination.read_bytes() == b"downloaded"


def test_provenance_reuse_requires_exact_attribution_metadata(tmp_path: Path) -> None:
    row = {
        "birdnet_id": "BN00001",
        "image_author": "Photographer",
        "image_license": "cc-by",
        "image_source": "iNaturalist",
        "image_url": "https://birdnet.cornell.edu/taxonomy/api/image/example",
    }
    expected = _image_provenance(row)
    path = tmp_path / "source.json"
    path.write_text(
        """{
  "birdnet_id": "BN00001",
  "image_author": "Photographer",
  "image_license": "cc-by",
  "image_source": "iNaturalist",
  "image_url": "https://birdnet.cornell.edu/taxonomy/api/image/example"
}
""",
        encoding="utf-8",
    )

    assert _provenance_matches(path, expected)
    changed = {**expected, "image_license": "cc-by-nc"}
    assert not _provenance_matches(path, changed)
