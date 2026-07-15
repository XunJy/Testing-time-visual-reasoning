# 04 · FuDD + EVA02-CLIP-L/14@336 + CUB-200-2011

Status: **complete; FuDD improves Top-1 by +1.4498 percentage points**.

This experiment asks one paired question: when the paper-described FuDD
reranking is applied to the stronger EVA02-CLIP-L/14@336 encoder, does CUB
Top-1 improve over that same encoder's single-template baseline?

No number from the OpenAI CLIP experiment is used as a pass threshold. A valid
negative or zero FuDD delta is still a completed scientific result.

## Locked protocol

| Component | Locked value |
|---|---|
| Dataset | Official CUB-200-2011 test split, 5,794 images |
| Baseline prompt | `a photo of a {class}.` |
| FuDD descriptions | Official FuDD CUB assets at commit `32264231fec047eb0bbbf59bfdbc8e6d208a096b` |
| Candidate count | Baseline Top-10 (`k=10`) |
| FuDD aggregation | Normalise each text feature → mean per candidate → normalise → candidate-only rerank |
| Architecture | OpenCLIP `EVA02-L-14-336` |
| Pretrained tag | `merged2b_s6b_b61k` |
| Hugging Face mirror | `timm/eva02_large_patch14_clip_336.merged2b_s6b_b61k` |
| HF revision | `4f62907359c8506be7021582f360564693b22c15` |
| Weight file | `open_clip_model.safetensors` (856,239,456 bytes) |
| Weight SHA-256 | `f753bca0e8327f77e8845b0af2510d599c3e4614237007b48078c791f2cf391c` |
| Tokenizer | OpenCLIP `SimpleTokenizer`, OpenAI BPE, context length 77 |
| Image preprocessing | 336px, bicubic, resize shortest side, center crop, OpenAI mean/std |
| T4 inference | FP16 model forward; embeddings cast to FP32 before normalisation, caching, means, and similarity |
| Default batches | 16 images, 256 texts |
| Seed | 2026 |

The OpenCLIP registry identifies this converted checkpoint as originating from
`QuanSun/EVA-CLIP/EVA02_CLIP_L_336_psz14_s6B.pt`. The runner never resolves a
moving `main` revision: it downloads the safetensors file from the exact HF
commit above and verifies byte count and SHA-256 before model construction.

The experiment reuses `src/ttvr/methods/fudd/evaluation.py`; it does not contain
an EVA-specific copy of FuDD. `local_batched_reference_parity` compares this
project's batched implementation against its local per-image formulation. It is
not labelled as bitwise parity with every upstream Python/set iteration order.

## Run

From the repository root:

```bash
python -m pip install -e ".[eva02,dev]"
python scripts/fudd/run_eva02_clip_cub.py --max-samples 32
python scripts/fudd/run_eva02_clip_cub.py
```

The `eva02` extra does not install the legacy OpenAI CLIP package; the two
model backends have independent dependency groups.

The full command must run without `--max-samples`. Each invocation creates a
new, non-overwriting directory under `runs/` containing:

- `run_state.json` with lifecycle state and locked configuration;
- `predictions.jsonl` with one baseline/FuDD record per image;
- `result.json` with aggregate metrics, transitions, environment, and complete
  model/checkpoint provenance;
- `environment_pip_freeze.txt`;
- `checksums.sha256` for all other run files.

`FULL_PASS` means the locked protocol and artifacts are complete. It does not
mean FuDD improved accuracy. The scientific comparison is the paired
`baseline_top1_percent`, `fudd_top1_percent`, `gain_pp`, recovered/degraded
counts, plus uncertainty/significance computed later from `predictions.jsonl`.

Existing runs are frozen. Never rerun into or edit an earlier run directory.

## Complete result

The locked full run finished on a Tesla T4 and passed every protocol-integrity
check. Its immutable run id is
`20260715T024435.574646Z-full-752dd8120b`.

| Metric | EVA02 baseline | EVA02 + FuDD | Difference |
|---|---:|---:|---:|
| Top-1 | 4,052 / 5,794 (69.9344%) | 4,136 / 5,794 (71.3842%) | **+84 (+1.4498 pp)** |
| Top-5 | 5,483 / 5,794 (94.6324%) | 5,518 / 5,794 (95.2365%) | +35 (+0.6041 pp) |

At Top-1, FuDD recovered 257 baseline errors and degraded 173 baseline
successes. The exact two-sided McNemar test gives
`p = 5.9590433535e-05` (430 discordant images). A 10,000-resample paired
nonparametric bootstrap with seed 2026 places the 95% interval for the Top-1
gain at **+0.7421 to +2.1574 pp**. The interval excludes zero, so
the answer to this experiment's paired question is **yes**.

The [full result](runs/20260715T024435.574646Z-full-752dd8120b/result.json),
[paired analysis](analysis/20260715T024435.574646Z-full-752dd8120b.json), and
all 5,794 per-image records are retained in the experiment directory. The
prediction file's SHA-256 is
`b3754d29688dda6b57d68731eb0832b65a4f9cebbbb3c3c94cb86d982e517cb1`.
The dataset fingerprint and FuDD prompt digest exactly match experiment 01.

## Comparison with OpenAI CLIP

| Model | Baseline Top-1 | FuDD Top-1 | FuDD gain |
|---|---:|---:|---:|
| OpenAI CLIP ViT-L/14@336 | 63.3586% | 65.7404% | +2.3818 pp |
| EVA02-CLIP-L/14@336 | **69.9344%** | **71.3842%** | +1.4498 pp |

EVA02 raises the baseline by 6.5758 pp and the FuDD result by 5.6438 pp versus
the earlier OpenAI CLIP run. FuDD's point-estimate gain is 0.9320 pp smaller
on EVA02, but a paired bootstrap comparison of the two gains has a 95%
interval of -1.9848 to +0.1208 pp and therefore includes zero;
the present data therefore do not establish that the FuDD effect itself is
smaller. They do establish that the paper-described FuDD operation still
provides a positive, statistically supported improvement on this stronger
CLIP-family encoder.

## Failure-set follow-up

The [cross-run failure report](analysis/failure_comparison_with_01/README.md)
exports every image for which FuDD changed a correct baseline prediction into
an error, plus every image that remained wrong before and after reranking. It
also retains the 30 images degraded in both model experiments and the 1,090
images missed by all four model/method configurations. The CSV files include
true and predicted class names, class ids, and target ranks inside the Top-10
candidate set; neither immutable run was modified.
