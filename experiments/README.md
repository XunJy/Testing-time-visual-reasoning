# Experiment registry

Every experiment is named by **method + model + dataset**:

```text
NN_<method>_<model>_<dataset>/
```

Examples are `01_fudd_clip_cub`, `02_fudd_siglip_cub`, and
`03_fudd_siglip2_cub`. Reusable method code belongs under
`src/ttvr/methods/`; model-specific encoding belongs under
`src/ttvr/models/`. An experiment directory owns only the locked protocol,
configuration, scientific status, commands, and immutable artifacts for one
concrete combination.

## Registry

| ID | Experiment | Status | Canonical result |
|---:|---|---|---|
| [01](01_fudd_clip_cub/) | FuDD + OpenAI CLIP ViT-L/14@336px + CUB | **Complete** | Top-1 `63.3586% -> 65.7404%` (**+2.3818 pp**) |
| [02](02_fudd_siglip_cub/) | FuDD + SigLIP + CUB | Planned | No implementation or run |
| [03](03_fudd_siglip2_cub/) | FuDD + SigLIP 2 + CUB | Planned | No implementation or run |
| [04](04_fudd_eva02_clip_cub/) | FuDD + EVA02-CLIP-L/14@336 + CUB | **Complete** | Top-1 `69.9344% -> 71.3842%` (**+1.4498 pp**) |
| [05](05_residual_head_clip_cub/) | Supervised linear/residual head + OpenAI CLIP + CUB | **Exploratory complete** | linear `86.9348%`; residual `87.1074%`; residual-vs-linear gain not significant |
| [06](06_feature_adapter_clip_multi_bird/) | BirdMix-v1 feature adapter + OpenAI CLIP -> CUB | **Frozen unfinished** | Preregistered protocol; no formal run |
| [07](07_feature_adapter_clip_birdmix_v2_cub/) | BirdMix-v2 six-source feature adapter + OpenAI CLIP -> CUB | **Complete; no positive transfer supported** | 3-seed mean `62.4554%`, **-0.9032 pp** versus baseline |

The detailed README inside each experiment is authoritative for its protocol
and interpretation. Compact machine-readable result entry points are:

- experiment 01:
  [`result.json`](01_fudd_clip_cub/runs/20260714T185902.445729Z-full-3b975c99f4/result.json)
  and
  [`predictions.jsonl`](01_fudd_clip_cub/runs/20260714T185902.445729Z-full-3b975c99f4/predictions.jsonl);
- experiment 04:
  [`result.json`](04_fudd_eva02_clip_cub/runs/20260715T024435.574646Z-full-752dd8120b/result.json)
  and
  [`predictions.jsonl`](04_fudd_eva02_clip_cub/runs/20260715T024435.574646Z-full-752dd8120b/predictions.jsonl);
- experiment 05:
  [`result.json`](05_residual_head_clip_cub/runs/20260715T053257.143885Z-full-dd09ff67b6/result.json)
  and
  [`linear_vs_residual.json`](05_residual_head_clip_cub/analysis/20260715T053257.143885Z-full-dd09ff67b6-linear-vs-residual.json);
- experiment 07:
  [`summary.json`](07_feature_adapter_clip_birdmix_v2_cub/analysis/20260715T155506.258376Z-summary-aa87ae1873/summary.json).

Experiment 07's full caches, checkpoints, and per-seed runs are intentionally
kept in the permissioned Google Drive folder linked from its README; Git keeps
the strict summary and checksum. Planned experiments 02 and 03 do not inherit
the historical SigLIP 2 numbers archived under `docs/history/`.

Crop and other trained-head experiments receive an ID only after their model,
baseline, data split, and acceptance criteria are fixed. For example,
`08_crop_siglip2_cub` and `09_residual_head_siglip2_cub` are complete names;
`crop_cub` and `residual_head_cub` are not, because the model axis is missing.

## Run contract

Each formal run writes a new fail-closed `runs/<run-id>/` directory. Existing
results are never overwritten or silently replaced. A completed inference run
should include configuration, environment, aggregate metrics, per-sample
predictions, lifecycle state, and checksums. Training-based methods must also
identify the training and validation split, checkpoint, seed, selection rule,
and parent baseline.

Small publishable artifacts belong in Git. Large datasets, feature caches, and
weights may be stored externally, but their experiment README must identify
the storage and permission boundary and retain checksums when available.
