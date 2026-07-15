# Experiment 01: FuDD + OpenAI CLIP on CUB-200-2011

This directory owns the independent CUB reproduction. It does not import or
mutate the ignored local legacy archive.

## Locked scientific protocol

- official CUB test split (5,794 images, 200 classes);
- OpenAI CLIP `ViT-L/14@336px`, loaded through the official FuDD FP32 path;
- one first-round prompt per class: `A photo of a {class name}.`;
- first-round Top-10 candidates;
- official FuDD pairwise descriptions, equal-weight embedding mean, and
  candidate-only reranking;
- no training, score fusion, threshold, prompt tuning, or online LLM call.

## Commands

Install the project once, then run an integration smoke test:

```bash
python -m pip install --upgrade pip "setuptools>=69,<81" wheel
python -m pip install --no-build-isolation -e ".[openai-clip,dev]"
python scripts/fudd/run_clip_cub.py --max-samples 32
```

The OpenAI CLIP extra is pinned to upstream commit
`a1d071733d7111c9c014f024669f959182114e33`. Its legacy build imports
`pkg_resources`, so the compatible setuptools range and `--no-build-isolation`
are intentional reproducibility constraints.

Only after the smoke run passes, evaluate the complete test split:

```bash
python scripts/fudd/run_clip_cub.py
```

Each invocation creates a new fail-closed directory under `runs/`; it never
overwrites an earlier invocation.  A completed run contains:

- `result.json`: protocol, environment, aggregate metrics, parity result, and
  paper/prior comparison checks;
- `predictions.jsonl`: one row per image with target, baseline Top-10, and FuDD
  ranking;
- `environment_pip_freeze.txt`: installed Python packages;
- `run_state.json`: lifecycle state;
- `checksums.sha256`: hashes for every other run artifact.

Smoke accuracy is not a paper result.  Only a 5,794-image run may be compared
with the paper's `63.48% → 65.90%` CUB result.

## Completed reproduction

The first complete independent run passed on 2026-07-14:

| Metric | Baseline | FuDD k=10 | Change |
|---|---:|---:|---:|
| Top-1 | 3,671 / 5,794 (63.3586%) | 3,809 / 5,794 (65.7404%) | +138 (+2.3818 pp) |
| Top-5 | 5,338 / 5,794 (92.1298%) | 5,375 / 5,794 (92.7684%) | +37 (+0.6386 pp) |

Paired Top-1 transitions were 404 recovered, 266 degraded, 3,405 correct in
both rounds, and 1,719 wrong in both rounds.  Baseline Top-10 candidate recall
was 96.4101%.  The 32-image official-style reference parity gate matched all 32
predictions; the largest prototype absolute difference was `1.1921e-7`.

Canonical artifacts:

- [`runs/20260714T185902.445729Z-full-3b975c99f4/result.json`](runs/20260714T185902.445729Z-full-3b975c99f4/result.json)
- [`runs/20260714T185902.445729Z-full-3b975c99f4/predictions.jsonl`](runs/20260714T185902.445729Z-full-3b975c99f4/predictions.jsonl)
- [`runs/20260714T185902.445729Z-full-3b975c99f4/checksums.sha256`](runs/20260714T185902.445729Z-full-3b975c99f4/checksums.sha256)

The run used a Tesla T4, Python 3.12.13, PyTorch 2.11.0+cu128, and source digest
`3b975c99f4c0af9d66d0beaf0d23f58d501437871c49b211c959203e31a62e04`.
Every predeclared full-run check is true.  The exact agreement with the earlier
historical numbers is a cross-check only; this run was produced through the
current implementation and did not import the ignored local legacy archive.
