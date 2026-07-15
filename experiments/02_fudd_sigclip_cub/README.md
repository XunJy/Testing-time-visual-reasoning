# Experiment 02: FuDD + SigCLIP on CUB-200-2011

Status: **planned; not yet implemented or run**.

This combination will reuse the validated FuDD candidate construction and
candidate-only reranking in `src/ttvr/methods/fudd/`, while providing a separate
SigCLIP backend under `src/ttvr/models/`.

Before the first test-set run, lock the exact SigCLIP checkpoint, preprocessing,
text template, embedding normalization, score direction, precision, baseline,
and validation protocol. Results from FuDD + CLIP must not be copied into this
directory or presented as SigCLIP results.
