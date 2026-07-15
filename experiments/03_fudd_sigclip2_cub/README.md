# Experiment 03: FuDD + SigCLIP 2 on CUB-200-2011

Status: **planned; not yet implemented or run**.

This combination will reuse the validated FuDD method implementation and add a
separate SigCLIP 2 model backend. It is scientifically distinct from both
FuDD + OpenAI CLIP and FuDD + SigCLIP, so it owns independent configuration,
caches, predictions, and immutable run directories.

Before evaluation, lock the exact SigCLIP 2 checkpoint and variant, resolution,
preprocessing, tokenizer, text pooling, normalization, precision, baseline, and
validation protocol. No result should be added until those choices are recorded.
