# 06 · Feature adapter + OpenAI CLIP + Multi-Bird

## Question

Can a small **class-agnostic** map learned from many bird species improve
frozen CLIP on bird species that are absent from adapter training?

This is not another CUB classifier. OpenAI CLIP `ViT-L/14@336px` stays frozen.
The only learned component is a shared `768 -> 128 -> 768` residual map:

```text
x  = normalize(CLIP_image(image))
r  = W_up(GELU(W_down(x)))
x' = normalize(x + r)
score(image, arbitrary_text_class) = 100 * x' @ text_class
```

`W_up` is zero-initialised, so step zero is exactly frozen CLIP. There is no
parameter whose shape depends on the number or identity of training classes.

## Locked BirdMix-v1 sources

| Role | Source | Locked usable subset |
|---|---|---:|
| train/validation | iNaturalist 2021 Mini Aves | 74,300 images, 1,486 source species |
| auxiliary train | BirdNET+ Taxonomy v0.3-Jul2026 | 7,287 licensed images/species |
| untouched target | CUB-200-2011 official test | 5,794 images, 200 classes |

BirdNET images are filtered to CC/CC0/PD/GFDL, excluding every ND image,
empty licence, and Macaulay copyright image. BirdNET is a one-image-per-species
coverage source, not a benchmark.

NABirds is prepared as a later locked source only after the repository owner
personally accepts Cornell's download terms. Birdsnap is excluded because its
official site no longer provides a verifiable package or licence. Semi-Aves is
conditional because its official anonymous image links currently return 404.
These datasets must not silently appear in a BirdMix-v1 run.

## Primary protocol: external-only strict CUB

1. No CUB image is used for fitting, validation, early stopping, or prompt
   design.
2. Map source and CUB labels to the locked BirdNET/AviList taxonomy.
3. Remove all source images and all training text prototypes whose accepted
   species matches any CUB target species. This prevents indirect exposure to
   target names as negative-only logits.
4. Select optimiser settings on source validation only.
5. After locking the adapter, evaluate once on the complete official CUB test
   split with the exact historical prompt `a photo of a {CUB class}.` for both
   frozen and adapted CLIP.

This track asks for transfer to a target dataset and target species that were
both unseen during adapter fitting.

## Secondary protocol: CUB base-to-novel

A deterministic adjacent-pair split assigns one class in every pair to base
and the other to novel, producing 100/100 classes throughout CUB's ordered
vocabulary. Only official CUB train images from base classes may be added to
BirdMix. Every external image and training prototype matching a novel species
is removed. The novel official test images are evaluated after model selection.

## Training

Each source retains its own local candidate vocabulary. For source `d`:

```text
L_d = CE(100 * x' @ T_d.T, y) + lambda * (1 - cosine(x', x))
```

- choose a source with probability proportional to `sqrt(number of images)`;
- choose a species uniformly within that source;
- choose an image uniformly within that species;
- report source validation as an equal-weight macro average over datasets.

The initial trial uses hidden size 128, batch 256, learning rate `3e-4`, weight
decay `1e-4`, identity weight `0.1`, and seed 2026. A later search may use only
source validation and must be saved as separate immutable trials.

Species used for model selection are split once, globally, across all source
datasets. This means that a species shared by BirdNET and iNaturalist cannot
appear in one source's fitting task and another source's validation task. The
validation-selected step count is then locked, the selection checkpoint is
discarded, and a freshly initialised adapter is refit for exactly that many
steps using every retained source species. `selection_adapter.pt` and the
all-source `adapter.pt` are both preserved so the two stages cannot be confused.

The first formal comparison is preregistered for seeds 2026, 2027, and 2028:

1. iNaturalist 2021 Mini Aves alone;
2. BirdMix-v1: iNaturalist 2021 Mini Aves plus licensed BirdNET images.

Every seed is reported. CUB performance is never used to select a seed.

The complete six-run matrix is locked in
`configs/clip_birdmix_preregistered_v1.json`. Once both source feature caches
exist, inspect the exact commands and then launch them sequentially with:

```bash
.venv/bin/python scripts/feature_adapter/run_clip_birdmix_study.py --dry-run \
  --python .venv/bin/python
.venv/bin/python scripts/feature_adapter/run_clip_birdmix_study.py \
  --python .venv/bin/python
```

The launcher checks every shared input and every cache referenced by either
source config before it starts the first GPU process. It stops at the first
failed trial, streams the runner output, and relies on the core runner's
timestamped fail-on-overwrite run directories; an earlier run is never reused
or replaced.

## Text prototypes

Training uses the mean-normalised frozen CLIP embeddings of:

```text
a photo of the bird species called {canonical common name}.
a photo of the bird species {scientific binomial}.
```

CUB historical evaluation keeps the original single template. Frozen and
adapted comparisons always share the same text matrix; prompt changes cannot
be counted as adapter gains.

## Required run artifacts

Every completed run must contain config/environment, locked source manifests,
taxonomy and exclusion manifests, text prototype strings and tensors, adapter
checkpoint, validation history, per-CUB prediction rows, paired statistics,
aggregate results, lifecycle state, and SHA-256 checksums. Existing runs are
never replaced.

## Current status

In progress. Dataset preparation and CLIP feature extraction run per source so
an interrupted Colab can resume without recomputing completed caches.
