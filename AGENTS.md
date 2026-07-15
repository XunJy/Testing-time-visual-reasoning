# Codex project instructions

## Project scope

This repository is the canonical workspace for Testing-Time Visual Reasoning.
Keep new experiments separate from frozen historical artifacts and never
silently replace an earlier result.

Name each experiment by method, model, and dataset, for example
`01_fudd_clip_cub`. Reusable method code belongs under `src/ttvr/methods/` and
model-specific encoding belongs under `src/ttvr/models/`; do not duplicate a
method implementation for each model combination.

## Remote GPU

Before GPU work, read
[`docs/REMOTE_GPU_CONNECTION.md`](docs/REMOTE_GPU_CONNECTION.md) completely.

- A Colab runtime is ephemeral. Never reuse a Tailscale IP, hostname, process,
  environment, or file path from an earlier runtime without checking it.
- Obtain the current `TAILSCALE_IP` from the active notebook, then test SSH and
  `nvidia-smi` before launching an experiment.
- Use the non-root user `codex` and `/home/codex/project` remotely.
- Never store Tailscale login URLs, auth keys, API keys, or Codex credentials in
  this repository, and never copy `~/.codex/auth.json` between machines.
- Copy important results back to the Mac frequently. `tmux` survives an SSH
  disconnect, but not deletion of the Colab runtime.
- Never use `rsync --delete` for this project.

The canonical local path on Xunjie's Mac is:

```text
/Users/xunj/Desktop/Testing-Time Visual Reasoning
```
