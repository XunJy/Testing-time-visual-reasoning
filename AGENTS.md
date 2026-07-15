# Codex project instructions

## Project scope

This repository is the canonical workspace for Testing-Time Visual Reasoning.
Keep new experiments separate from frozen historical artifacts and never
silently replace an earlier result.

Name each experiment by method, model, and dataset, for example
`01_fudd_clip_cub`. Reusable method code belongs under `src/ttvr/methods/` and
model-specific encoding belongs under `src/ttvr/models/`; do not duplicate a
method implementation for each model combination.

The canonical local path on Xunjie's Mac is:

```text
/Users/xunj/Desktop/Testing-Time Visual Reasoning
```
