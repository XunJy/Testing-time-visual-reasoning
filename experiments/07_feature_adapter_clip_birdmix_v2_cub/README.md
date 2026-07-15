# 07 · Feature adapter + OpenAI CLIP + BirdMix-v2 + CUB

## Goal

Test whether broader, externally trained bird-domain adaptation transfers to
the untouched CUB-200-2011 test set. Experiment 06 remains the frozen
BirdMix-v1 study; this directory never reads, rewrites, or selects its runs.

The model and loss are unchanged: frozen OpenAI CLIP `ViT-L/14@336px` plus the
class-agnostic `768 -> 128 -> 768` residual feature adapter. CUB is used only
once after source-only selection and all-source refit.

## Enabled sources

Source order is protocol data because it breaks exact-byte duplicate ties.

1. iNaturalist 2021 Mini Aves (`train` and `validation` caches);
2. licensed BirdNET Taxonomy images (`train` cache);
3. Big Bird species bounding-box crops (`train` cache);
4. Visual WetlandBirds stride-10 crops (`train` and `validation` caches);
5. USGS aerial avian publisher crops (`train` and `validation` caches);
6. New Mexico UAS expert-consensus waterfowl crops (`train` cache).

USGS contributes 9,955 non-test crops from four species under CC0. New Mexico
contributes 2,171 canonical crops from seven species; the strict target-taxon
filter later removes 1,688 Mallard and five Gadwall crops, leaving 478 crops
from five species. The New Mexico landing page names CC BY-NC 4.0 while the
embedded COCO record names CC BY-NC 2.0; both official statements are retained
in `source.json`, and the stricter non-commercial restriction governs this
research experiment.

## Colab storage contract

Large downloaded archives, decoded videos, source frames, and generated crops
may stay on Colab's ephemeral disk under the project `data/` directories. The
durable feature-cache root is locked to:

```text
/content/drive/MyDrive/Testing-Time Visual Reasoning/cache/clip_birdmix_v2
```

Pass that path as `--cache-dir` to
`scripts/feature_adapter/cache_clip_manifest_features.py`. Its 2,048-row shards
and assembled final caches are both written there, so an interrupted runtime
can resume without re-encoding completed shards. Copy each immutable
`source.json`, `samples.jsonl`, and `taxa.jsonl` to the per-dataset directory
under `/content/drive/MyDrive/Testing-Time Visual Reasoning/manifests/`; the
locked source config reads those Drive copies while image roots remain
ephemeral. Trial outputs go directly to
`/content/drive/MyDrive/Testing-Time Visual Reasoning/runs/07_feature_adapter_clip_birdmix_v2_cub`.

Thus raw media can be discarded with the Colab runtime, while shards, final
feature caches, manifests, and immutable run artifacts survive on Drive.

## Leakage and split rules

- Canonical manifests retain their complete, licensed source records.
- The runner removes every source row whose accepted BirdNET/AviList taxon is
  a CUB target. It also removes the matching training text prototype.
- Only explicitly configured `train` and `validation` feature caches can be
  loaded. A cache tagged as `test` fails metadata validation. Official source
  `test` rows are counted in `source_partition_audit.jsonl`, with used count
  locked to zero.
- Species are held out globally, so a taxon shared across datasets cannot be
  used for selection training in one source and validation in another.
- A source with fewer than two available held-out taxa contributes no local
  validation task; the omission and exact taxa are recorded. The run fails
  unless at least one source still supplies a valid global validation task.
- Refit combines only cached train/validation rows and remaps each split's
  local labels before fitting. It never assumes the two splits have identical
  vocabularies.

## Cross-source duplicate policy

`cross_source_duplicate_audit.json` records all cross-source exact SHA-256
groups, identical 64-bit perceptual-hash groups, and pHash pairs within Hamming
distance four. Exact-byte duplicates among fit-eligible rows keep the first
source occurrence and exclude later occurrences at task construction time;
canonical manifests and feature caches are not changed. Perceptual candidates
are **report-only** because a visually similar crop can be a valid distinct
observation or even a different species. No fuzzy collision is auto-deleted.

## Locked study

The initial matrix is one all-enabled-source trial for seeds 2026, 2027, and
2028. CUB performance cannot select a seed. The JSON contracts are in
`schemas/`, and the launcher fails closed on unknown keys, missing enabled
inputs, ambiguous cache globs, or subprocess failure.

Dry-run performs schema validation and prints the three commands without
requiring datasets to have finished downloading:

```bash
.venv/bin/python scripts/feature_adapter/run_clip_birdmix_v2_study.py \
  --dry-run --python .venv/bin/python
```

After all enabled manifests and feature caches exist, remove `--dry-run` to
run the three trials sequentially. Every result is written under this
experiment's own `runs/` directory with a timestamp and config digest; no
earlier result is replaced.

## Required audit artifacts

In addition to the adapter, validation history, CUB predictions, statistics,
locked source manifests, and checksums, every completed v2 run contains:

- `cross_source_duplicate_audit.json`;
- `source_partition_audit.jsonl`;
- `source_task_membership.jsonl` with an exact source-index digest for every
  included selection, validation, and refit task.

Status: code and locked configuration are ready; dataset preparation, feature
caching, and GPU training are deliberately not started by this setup change.
