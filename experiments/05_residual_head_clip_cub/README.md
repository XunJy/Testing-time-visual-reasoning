# Experiment 05: Residual head + OpenAI CLIP on CUB-200-2011

Status: **complete; supervised residual adaptation reaches 87.1074% Top-1**.

This experiment trains a small supervised classifier on frozen OpenAI CLIP
`ViT-L/14@336px` image features. It does not update the CLIP image or text
encoders.

The parent baseline is immutable experiment 01 run
`20260714T185902.445729Z-full-3b975c99f4` (3,671/5,794 Top-1). The three test
conditions are:

1. the frozen single-template zero-shot CLIP baseline;
2. a supervised linear probe `W x + b`;
3. a residual classifier `zero_shot_logits + alpha * (W x + b)`.

## Locked protocol

Only the official CUB training split is used for fitting and model selection.
Six images per class are deterministically assigned to validation by a stable
hash, producing 4,794 fit and 1,200 validation images. Hyperparameters and
`alpha` are selected on that validation partition, the chosen head is refit on
all 5,994 training images, and the official 5,794-image test split is evaluated
once.

| Component | Locked value |
|---|---|
| Backbone | OpenAI CLIP `ViT-L/14@336px`, frozen FP32 encoder |
| Baseline prompt | `a photo of a {class}.` from the official FuDD prompt assets |
| Frozen feature dimension | 768 |
| Trainable head | affine `W x + b`, zero initialisation, 153,800 parameters |
| Linear probe | `W x + b` |
| Residual condition | `100 * cosine(x,text) + alpha * (W x + b)` |
| Optimiser | AdamW |
| Learning rates | `1e-3`, `3e-3`, `1e-2` |
| Weight decay | `0`, `1e-4`, `1e-3`, `1e-2` |
| Alpha grid | `0`, `0.25`, `0.5`, `0.75`, `1`, `1.5`, `2` |
| Training | batch 256, at most 200 epochs, patience 20 |
| Selection | validation Top-1, then Top-5, then lower cross-entropy |
| Tie order | ascending alpha, learning rate, and weight decay |
| Seed | 2026 |

Because the frozen CLIP logits and the residual are both affine functions of
the same image feature, the residual condition does **not** have greater model
capacity than a linear probe. Its purpose is to test whether retaining CLIP's
zero-shot classifier as an explicit prior improves supervised adaptation.

## Run

From the repository root on the connected GPU:

```bash
PYTHONPATH=src python scripts/residual_head/cache_clip_cub_features.py
PYTHONPATH=src python scripts/residual_head/run_clip_cub.py
```

Each invocation creates a new non-overwriting `runs/<run-id>/` directory. A
complete run contains the deterministic split manifest, every hyperparameter
trial, validation and full-train head weights, all 5,794 per-image predictions,
aggregate paired statistics, environment metadata, lifecycle state, and
SHA-256 checksums.

## Interpretation

This is supervised frozen-feature head tuning, not zero-shot testing-time
reasoning and not full CLIP fine-tuning. Because this project has already
inspected examples from the CUB test set, the run is exploratory rather than a
pristine confirmatory evaluation.

## Complete result

The locked run completed on a Tesla T4 with immutable run id
`20260715T053257.143885Z-full-dd09ff67b6`. Its recalculated zero-shot Top-10
matches all 5,794 parent-run rows exactly, and every saved checksum verifies.

| Condition | Top-1 | Top-5 | Top-1 vs zero-shot |
|---|---:|---:|---:|
| Frozen zero-shot CLIP | 3,671 / 5,794 (63.3586%) | 5,338 / 5,794 (92.1298%) | — |
| Supervised linear probe | 5,037 / 5,794 (86.9348%) | 5,701 / 5,794 (98.3949%) | +1,366 (+23.5761 pp) |
| Zero-shot + residual affine head | **5,047 / 5,794 (87.1074%)** | 5,700 / 5,794 (98.3776%) | **+1,376 (+23.7487 pp)** |

Against zero-shot CLIP, the residual condition recovered 1,547 errors and
degraded 171 correct predictions. Its paired bootstrap 95% interval for the
Top-1 gain is +22.4543 to +25.0604 pp, and exact McNemar
`p = 2.9867e-277`.

The residual condition is only 10 net images (+0.1726 pp) above the linear
probe. Across those two supervised methods, 150 images are recovered and 140
are degraded; the 95% paired-bootstrap interval is -0.3970 to +0.7421 pp and
exact McNemar `p = 0.5972`. Therefore this run strongly establishes the value
of supervised adaptation, but **does not establish that retaining the
zero-shot classifier as a residual prior is better than an ordinary linear
probe**.

Canonical artifacts:

- [`result.json`](runs/20260715T053257.143885Z-full-dd09ff67b6/result.json)
- [`predictions.jsonl`](runs/20260715T053257.143885Z-full-dd09ff67b6/predictions.jsonl)
- [`heads.pt`](runs/20260715T053257.143885Z-full-dd09ff67b6/heads.pt)
- [`checksums.sha256`](runs/20260715T053257.143885Z-full-dd09ff67b6/checksums.sha256)
- [`linear_vs_residual.json`](analysis/20260715T053257.143885Z-full-dd09ff67b6-linear-vs-residual.json)
