# Experiment registry

Every experiment is named by **method + model + dataset**:

```text
NN_<method>_<model>_<dataset>/
```

Examples are `01_fudd_clip_cub`, `02_fudd_sigclip_cub`, and
`03_fudd_sigclip2_cub`. A method is implemented once under `src/ttvr/methods/`;
a model backend is implemented once under `src/ttvr/models/`. This directory
owns only the configuration, scientific status, commands, and immutable runs
for a concrete combination.

## Registry

| ID | Method | Model | Dataset | Status |
|---|---|---|---|---|
| 01 | FuDD | OpenAI CLIP ViT-L/14@336px | CUB-200-2011 | Complete |
| 02 | FuDD | SigCLIP | CUB-200-2011 | Planned |
| 03 | FuDD | SigCLIP 2 | CUB-200-2011 | Planned |

Crop and trained residual-head experiments receive an ID only after their
model, baseline, data split, and acceptance criteria are fixed. For example,
`04_crop_sigclip2_cub` and `05_residual_head_sigclip2_cub` would be valid names;
`crop_cub` and `residual_head_cub` would be incomplete because the model axis is
missing.

## Run contract

Each completed combination writes a new fail-closed `runs/<run-id>/` directory.
An existing result is never overwritten. A run should include configuration,
environment, aggregate metrics, per-sample predictions, lifecycle state, and
checksums. Training-based methods must additionally identify the training and
validation split, checkpoint, seed, and parent baseline.
