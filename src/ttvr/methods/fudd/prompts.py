"""FuDD-specific official CUB class names and differential descriptions.

The prompt assets are downloaded from the authors' repository at the commit
corresponding to the paper release and are verified byte-for-byte.  The loader
then validates all 200 classes and every one of the 19,900 unordered pairs.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import urllib.request
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

FUDD_OFFICIAL_COMMIT = "32264231fec047eb0bbbf59bfdbc8e6d208a096b"
FUDD_RAW_ROOT = (
    "https://raw.githubusercontent.com/BatsResearch/fudd/"
    f"{FUDD_OFFICIAL_COMMIT}/differential_descriptions"
)
CUB_CLASS_COUNT = 200
CUB_PAIR_COUNT = CUB_CLASS_COUNT * (CUB_CLASS_COUNT - 1) // 2
DEFAULT_TEMPLATE = "a photo of a {}."

OFFICIAL_ASSET_SHA256: Mapping[str, str] = MappingProxyType(
    {
        "cub_class_names.json": (
            "01eff4094773d49b92210f56efb9cab13860c0cdaf9953ff1e0ef082cf226723"
        ),
        "cub_prompt_pairs.json": (
            "cc4300eaaf7c7bf46e515839ebe03abe827e86dafc3f49f63ea97a6c4c035237"
        ),
    }
)


@dataclass(frozen=True, slots=True)
class DifferentialDescription:
    """One attribute-specific description for each side of a class pair."""

    attribute: str
    first: str
    second: str

    def __post_init__(self) -> None:
        values = (self.attribute, self.first, self.second)
        if any(not isinstance(value, str) or not value.strip() for value in values):
            raise ValueError("Differential-description fields must be non-empty strings")


@dataclass(frozen=True, slots=True)
class ClassPairDescriptions:
    """Differential descriptions for a canonical ``first_id < second_id`` pair."""

    first_id: int
    second_id: int
    descriptions: tuple[DifferentialDescription, ...]

    def __post_init__(self) -> None:
        if not (0 <= self.first_id < self.second_id):
            raise ValueError("Class-pair ids must satisfy 0 <= first_id < second_id")
        if not self.descriptions:
            raise ValueError("A class pair requires at least one differential description")
        if not all(isinstance(item, DifferentialDescription) for item in self.descriptions):
            raise TypeError("descriptions must contain DifferentialDescription values")

    def prompts_for(self, class_id: int) -> tuple[str, ...]:
        """Return raw prompt strings associated with one side of the pair."""

        if class_id == self.first_id:
            return tuple(item.first for item in self.descriptions)
        if class_id == self.second_id:
            return tuple(item.second for item in self.descriptions)
        raise ValueError(f"Class {class_id} is not in pair ({self.first_id}, {self.second_id})")


@dataclass(frozen=True, slots=True)
class CubPromptRepository:
    """Validated, immutable view of CUB FuDD prompt assets."""

    class_names: tuple[str, ...]
    pairs: Mapping[tuple[int, int], ClassPairDescriptions]
    source_digest: str

    @property
    def class_count(self) -> int:
        return len(self.class_names)

    @property
    def pair_count(self) -> int:
        return len(self.pairs)

    def single_template_prompts(
        self,
        template: str = DEFAULT_TEMPLATE,
    ) -> tuple[tuple[str, ...], ...]:
        """Create the paper's single-template baseline prompts.

        ``str.capitalize`` is intentional: it exactly matches the official
        prompt factory before tokenisation.
        """

        if template.count("{}") != 1:
            raise ValueError("template must contain exactly one '{}' placeholder")
        return tuple((template.format(class_name).capitalize(),) for class_name in self.class_names)

    def pair_prompts(
        self,
        class_a: int,
        class_b: int,
    ) -> tuple[tuple[str, ...], tuple[str, ...]]:
        """Return capitalised prompts in the same order as the requested ids."""

        self._validate_class_ids((class_a, class_b), require_unique=True)
        low, high = sorted((class_a, class_b))
        pair = self.pairs[(low, high)]
        prompts_a = tuple(text.capitalize() for text in pair.prompts_for(class_a))
        prompts_b = tuple(text.capitalize() for text in pair.prompts_for(class_b))
        return prompts_a, prompts_b

    def prompts_for_candidates(
        self,
        class_ids: Sequence[int],
    ) -> tuple[tuple[str, ...], ...]:
        """Build FuDD prompt groups for one ordered candidate set.

        The output has one group per input class and preserves candidate order.
        For each class, descriptions against every other candidate are pooled.
        Duplicate strings are removed while retaining their first occurrence.
        """

        candidate_ids = tuple(int(class_id) for class_id in class_ids)
        self._validate_class_ids(candidate_ids, require_unique=True)
        if len(candidate_ids) < 2:
            raise ValueError("FuDD requires at least two candidate classes")

        grouped: list[list[str]] = [[] for _ in candidate_ids]
        for left_position in range(len(candidate_ids) - 1):
            for right_position in range(left_position + 1, len(candidate_ids)):
                left_prompts, right_prompts = self.pair_prompts(
                    candidate_ids[left_position], candidate_ids[right_position]
                )
                grouped[left_position].extend(left_prompts)
                grouped[right_position].extend(right_prompts)

        return tuple(tuple(dict.fromkeys(prompts)) for prompts in grouped)

    def all_differential_texts(self) -> tuple[str, ...]:
        """Return every unique differential prompt in deterministic order."""

        texts: dict[str, None] = {}
        for pair_key in sorted(self.pairs):
            pair = self.pairs[pair_key]
            for description in pair.descriptions:
                texts.setdefault(description.first.capitalize(), None)
                texts.setdefault(description.second.capitalize(), None)
        return tuple(texts)

    def with_pair_overrides(
        self,
        overrides: Mapping[tuple[int, int], ClassPairDescriptions],
    ) -> CubPromptRepository:
        """Derive a repository with explicitly replaced class pairs.

        The original repository remains unchanged.  The derived source digest
        includes both the base digest and a canonical serialization of every
        replacement, so official and experimental prompt caches cannot be
        confused.
        """

        if not isinstance(overrides, Mapping) or not overrides:
            raise ValueError("At least one pair override is required")

        pairs = dict(self.pairs)
        canonical: list[dict[str, Any]] = []
        for key in sorted(overrides):
            if not (
                isinstance(key, tuple)
                and len(key) == 2
                and all(isinstance(value, int) for value in key)
            ):
                raise TypeError("Pair override keys must be (first_id, second_id) tuples")
            pair = overrides[key]
            if not isinstance(pair, ClassPairDescriptions):
                raise TypeError("Pair overrides must contain ClassPairDescriptions values")
            expected_key = (pair.first_id, pair.second_id)
            if key != expected_key:
                raise ValueError(f"Override key {key} does not match pair ids {expected_key}")
            if key not in pairs:
                raise ValueError(f"Pair override is not present in the base repository: {key}")
            if pair.second_id >= self.class_count:
                raise ValueError(f"Pair override class id is out of range: {key}")

            pairs[key] = pair
            canonical.append(
                {
                    "classes": [pair.first_id, pair.second_id],
                    "prompt_pairs": [
                        {
                            "attr_type": item.attribute,
                            "prompt_pair": [item.first, item.second],
                        }
                        for item in pair.descriptions
                    ],
                }
            )

        digest = hashlib.sha256()
        digest.update(b"ttvr-fudd-pair-overrides-v1\0")
        digest.update(self.source_digest.encode())
        digest.update(b"\0")
        digest.update(
            json.dumps(
                canonical,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        )
        return CubPromptRepository(
            class_names=self.class_names,
            pairs=MappingProxyType(pairs),
            source_digest=digest.hexdigest(),
        )

    def _validate_class_ids(
        self,
        class_ids: Sequence[int],
        *,
        require_unique: bool,
    ) -> None:
        for class_id in class_ids:
            if not 0 <= class_id < self.class_count:
                raise ValueError(f"Class id out of range: {class_id}")
        if require_unique and len(set(class_ids)) != len(class_ids):
            raise ValueError("Class ids must be unique")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download_verified(url: str, destination: Path, expected_sha256: str) -> None:
    partial = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "ttvr-fudd-reproduction/1.0"},
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            with partial.open("wb") as output:
                shutil.copyfileobj(response, output, length=1024 * 1024)
        actual_sha256 = _sha256(partial)
        if actual_sha256 != expected_sha256:
            raise RuntimeError(
                f"Prompt asset checksum mismatch for {destination.name}: "
                f"expected {expected_sha256}, found {actual_sha256}"
            )
        partial.replace(destination)
    finally:
        partial.unlink(missing_ok=True)


def download_official_prompts(
    destination: Path | str,
    *,
    overwrite: bool = False,
) -> Path:
    """Download the pinned official CUB prompts and verify SHA-256 hashes.

    Returns the directory accepted by :func:`load_official_prompts`.  A valid
    existing file is reused.  A mismatched file is never silently replaced
    unless ``overwrite=True`` is explicit.
    """

    root = Path(destination).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    for filename, expected_sha256 in OFFICIAL_ASSET_SHA256.items():
        path = root / filename
        if path.exists() and not path.is_file():
            raise RuntimeError(f"Prompt asset path is not a file: {path}")
        if path.is_file():
            actual_sha256 = _sha256(path)
            if actual_sha256 == expected_sha256:
                continue
            if not overwrite:
                raise RuntimeError(
                    f"Existing prompt asset has the wrong checksum: {path}. "
                    "Pass overwrite=True to replace it."
                )
        _download_verified(
            f"{FUDD_RAW_ROOT}/{filename}",
            path,
            expected_sha256,
        )
    return root


def verify_official_prompt_assets(root: Path | str) -> Mapping[str, str]:
    """Verify all pinned prompt files and return their SHA-256 digests."""

    prompt_root = Path(root).expanduser()
    digests: dict[str, str] = {}
    for filename, expected_sha256 in OFFICIAL_ASSET_SHA256.items():
        path = prompt_root / filename
        if not path.is_file():
            raise FileNotFoundError(f"Missing official prompt asset: {path}")
        actual_sha256 = _sha256(path)
        if actual_sha256 != expected_sha256:
            raise RuntimeError(
                f"Prompt checksum mismatch for {filename}: "
                f"expected {expected_sha256}, found {actual_sha256}"
            )
        digests[filename] = actual_sha256
    return MappingProxyType(digests)


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _parse_class_names(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list) or len(value) != CUB_CLASS_COUNT:
        raise RuntimeError("cub_class_names.json must contain exactly 200 names")
    names: list[str] = []
    for index, name in enumerate(value):
        if not isinstance(name, str) or not name.strip():
            raise RuntimeError(f"Invalid class name at index {index}")
        names.append(name.replace("_", " "))
    if len(set(names)) != CUB_CLASS_COUNT:
        raise RuntimeError("CUB class names must be unique")
    return tuple(names)


def _parse_pair(
    key: str,
    value: Any,
) -> tuple[tuple[int, int], ClassPairDescriptions]:
    try:
        first_from_key, second_from_key = (int(part) for part in key.split("_"))
    except (TypeError, ValueError) as error:
        raise RuntimeError(f"Invalid class-pair key: {key!r}") from error
    expected_classes = (first_from_key, second_from_key)
    if not (0 <= first_from_key < second_from_key < CUB_CLASS_COUNT):
        raise RuntimeError(f"Non-canonical class-pair key: {key!r}")
    if not isinstance(value, dict):
        raise RuntimeError(f"Pair {key!r} must contain an object")
    classes = value.get("classes")
    if classes != [first_from_key, second_from_key]:
        raise RuntimeError(f"Pair {key!r} has inconsistent class ids")
    raw_descriptions = value.get("prompt_pairs")
    if not isinstance(raw_descriptions, list) or not raw_descriptions:
        raise RuntimeError(f"Pair {key!r} has no prompt pairs")

    descriptions: list[DifferentialDescription] = []
    for index, raw in enumerate(raw_descriptions):
        if not isinstance(raw, dict):
            raise RuntimeError(f"Invalid description {index} for pair {key!r}")
        attribute = raw.get("attr_type")
        prompt_pair = raw.get("prompt_pair")
        if not isinstance(attribute, str) or not attribute.strip():
            raise RuntimeError(f"Invalid attribute {index} for pair {key!r}")
        if not (
            isinstance(prompt_pair, list)
            and len(prompt_pair) == 2
            and all(isinstance(text, str) and text.strip() for text in prompt_pair)
        ):
            raise RuntimeError(f"Invalid prompt pair {index} for pair {key!r}")
        descriptions.append(
            DifferentialDescription(
                attribute=attribute,
                first=prompt_pair[0],
                second=prompt_pair[1],
            )
        )

    return expected_classes, ClassPairDescriptions(
        first_id=first_from_key,
        second_id=second_from_key,
        descriptions=tuple(descriptions),
    )


def load_official_prompts(root: Path | str) -> CubPromptRepository:
    """Load and strictly validate the official CUB FuDD prompt repository."""

    prompt_root = Path(root).expanduser()
    digests = verify_official_prompt_assets(prompt_root)
    class_names = _parse_class_names(_load_json(prompt_root / "cub_class_names.json"))
    raw_pairs = _load_json(prompt_root / "cub_prompt_pairs.json")
    if not isinstance(raw_pairs, dict) or len(raw_pairs) != CUB_PAIR_COUNT:
        raise RuntimeError("cub_prompt_pairs.json must contain all 19,900 CUB class pairs")

    pairs: dict[tuple[int, int], ClassPairDescriptions] = {}
    for key, value in raw_pairs.items():
        pair_key, pair = _parse_pair(key, value)
        if pair_key in pairs:
            raise RuntimeError(f"Duplicate CUB prompt pair: {pair_key}")
        pairs[pair_key] = pair

    expected_pairs = {
        (first, second)
        for first in range(CUB_CLASS_COUNT - 1)
        for second in range(first + 1, CUB_CLASS_COUNT)
    }
    missing = expected_pairs.difference(pairs)
    extra = set(pairs).difference(expected_pairs)
    if missing or extra:
        raise RuntimeError(
            f"Incomplete CUB pair matrix: missing={len(missing)}, extra={len(extra)}"
        )

    combined_digest = hashlib.sha256()
    for filename in sorted(digests):
        combined_digest.update(filename.encode())
        combined_digest.update(digests[filename].encode("ascii"))

    return CubPromptRepository(
        class_names=class_names,
        pairs=MappingProxyType(pairs),
        source_digest=combined_digest.hexdigest(),
    )


def load_pair_overrides(path: Path | str) -> Mapping[tuple[int, int], ClassPairDescriptions]:
    """Load a versioned, sparse set of class-pair replacements.

    The ``pairs`` object uses the same per-pair schema as the official FuDD
    asset, while allowing experiment metadata to live beside it.
    """

    override_path = Path(path).expanduser()
    raw = _load_json(override_path)
    if not isinstance(raw, dict) or raw.get("schema_version") != 1:
        raise RuntimeError("Pair override file must be an object with schema_version=1")
    raw_pairs = raw.get("pairs")
    if not isinstance(raw_pairs, dict) or not raw_pairs:
        raise RuntimeError("Pair override file must contain a non-empty 'pairs' object")

    overrides: dict[tuple[int, int], ClassPairDescriptions] = {}
    for key, value in raw_pairs.items():
        pair_key, pair = _parse_pair(key, value)
        if pair_key in overrides:
            raise RuntimeError(f"Duplicate pair override: {pair_key}")
        overrides[pair_key] = pair
    return MappingProxyType(overrides)
