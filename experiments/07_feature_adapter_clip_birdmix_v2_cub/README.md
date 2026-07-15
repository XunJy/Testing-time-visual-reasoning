# 07 · Feature adapter + OpenAI CLIP + BirdMix-v2 + CUB

Status: **complete; external bird-domain residual adaptation does not improve
CUB Top-1 in this locked three-seed study**.

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

The formal launcher acquires `/tmp/ttvr-gpu.lock` before preflight and keeps
the lock across all three seeds. The normal
`cache_clip_manifest_features.py` invocation acquires that same lock before it
loads CLIP or reads a manifest. Both commands fail immediately with a clear
busy-lock error; they never wait silently or run concurrently. `--no-gpu-lock`
is an explicit unsafe opt-out reserved for isolated tests or expert-managed
GPU scheduling, not formal cache generation. Invoke the cache script directly;
do not wrap it in a second external `flock` on `/tmp/ttvr-gpu.lock`, because
the built-in lock is now the sole lock owner for that process.

The launcher runs its CLIP preflight with the exact interpreter selected by
`--python`. That interpreter must report the pinned OpenAI CLIP PEP 610 Git
commit and the exact SHA-256 of `ViT-L-14-336px.pt`; any already-existing text
cache must also be compatible. Each model-consuming child verifies the same
installation and checkpoint again; `config.json` records the verified commit,
cache identity, checkpoint path, size, and SHA-256. The cache identity remains
byte-for-byte compatible with existing caches produced by the correct pinned
runtime.

The run's source-code digest also covers the metric implementation and the
manifest feature-cache entry point in addition to the adapter, model, dataset,
and runner implementations.

After all three runs complete, create the fail-closed formal summary with:

```bash
PYTHONPATH=src .venv/bin/python \
  scripts/feature_adapter/summarize_clip_birdmix_v2_study.py
```

The summarizer requires exactly one complete run for every registered
trial/seed cell. Missing, incomplete, or duplicate attempts are errors. It
also verifies checksums, seed-invariant configs and input digests, source and
target text prototypes, the CUB crosswalk, and selected-step/refit-step
agreement. Baseline, adapted, and gain Top-1 are reported per seed and as the
arithmetic mean with an unclipped two-sided 95% Student-t interval over the
three registered seeds (`df=2`). By default, a new timestamp-and-summary-digest
directory is created below the Drive run root's `summaries/` directory; an
existing result is never overwritten.

## Formal result

The locked study completed on a Tesla T4 on 2026-07-15 using code commit
`299beb4`. The frozen baseline is the historical OpenAI CLIP result,
3,671/5,794 (63.3586%) Top-1, and is identical in every seed. All three adapted
runs are worse than that baseline.

| Seed | Frozen Top-1 | Adapted Top-1 | Gain | Recovered / degraded | Best step | Species-cluster bootstrap 95% CI | McNemar p | Strict pass |
|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 2026 | 3,671 (63.3586%) | 3,626 (62.5820%) | -0.7767 pp | 277 / 322 | 250 | [-2.7574, +0.9270] pp | 0.07212 | no |
| 2027 | 3,671 (63.3586%) | 3,590 (61.9606%) | -1.3980 pp | 369 / 450 | 750 | [-4.1095, +0.7611] pp | 0.00515 | no |
| 2028 | 3,671 (63.3586%) | 3,640 (62.8236%) | -0.5350 pp | 350 / 381 | 750 | [-3.0136, +1.3949] pp | 0.26716 | no |

Mean adapted Top-1 is **62.4554%** and mean gain is **-0.9032 percentage
points**. The preregistered three-seed Student-t 95% interval for the mean gain
is **[-2.0091, +0.2027] pp** (`df=2`). It includes zero, while every individual
point estimate is negative. Consequently the study provides no evidence of
positive transfer and does not pass the preregistered strict criterion in any
seed. Seed 2027 also shows a significant paired-image degradation under the
auxiliary exact McNemar test; that test does not replace the species-cluster
criterion.

Top-5 also falls in every seed:

| Seed | Frozen Top-5 | Adapted Top-5 |
|---:|---:|---:|
| 2026 | 5,338 (92.1298%) | 5,288 (91.2668%) |
| 2027 | 5,338 (92.1298%) | 5,277 (91.0770%) |
| 2028 | 5,338 (92.1298%) | 5,294 (91.3704%) |

Each seed used the same six enabled source datasets: 112,838 selection-fit
images, 7,159 globally species-disjoint validation images, and 139,701 images
for the fresh all-source refit. Four sources supplied validation tasks and two
were explicitly omitted because they did not retain enough held-out taxa.
Target filtering excluded 206 accepted target-taxon identifiers; cross-source
exact-deduplication removed zero eligible images. The adapter has 197,504
trainable parameters.

The independent formal summarizer verified 28 input files, all nine Boolean
audit invariants, every one of the 5,794 canonical CUB prediction identities,
and the fixed 3,671-image baseline. All six
`source_partition_audit.jsonl` rows report zero official-source-test samples
used. The immutable
[summary](analysis/20260715T155506.258376Z-summary-aa87ae1873/summary.json) is
copied byte-for-byte from
`summaries/20260715T155506.258376Z-summary-aa87ae1873/summary.json` beneath the
locked Drive run root; its SHA-256 is
`79fac5a8e3da72dacce8dd6085b45244d989d4c62a7c84c96aba2896ef701b6e`.
The complete caches, run artifacts, predictions, checkpoints, and summary are
retained in the [durable Google Drive experiment folder](https://drive.google.com/drive/folders/1vpcygJZ78tEhj40BNp8oYdwNUzSQuWok).

This experiment evaluates a BirdMix-v2-trained residual feature adapter. It is
not FuDD: no differential descriptions or test-time prompt reranking are used.

## Required audit artifacts

In addition to the adapter, validation history, CUB predictions, statistics,
locked source manifests, and checksums, every completed v2 run contains:

- `cross_source_duplicate_audit.json`;
- `source_partition_audit.jsonl`;
- `source_task_membership.jsonl` with an exact source-index digest for every
  included selection, validation, and refit task.

Status: the formal three-seed study and fail-closed summary are complete. The
immutable Drive runs remain separate from the frozen BirdMix-v1 artifacts.
