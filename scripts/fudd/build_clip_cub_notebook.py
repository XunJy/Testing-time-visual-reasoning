#!/usr/bin/env python3
"""Build the self-contained Colab notebook for the FuDD/CLIP reproduction.

The notebook is intentionally a thin orchestration layer.  The experiment lives
in ``src/ttvr``; this builder embeds a compressed snapshot of that package
so the notebook can start from a blank Colab runtime without relying on an
unpublished GitHub repository.

Run from anywhere with::

    python scripts/fudd/build_clip_cub_notebook.py

``nbformat`` is a build-time dependency only.  It is not needed inside Colab.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import textwrap
import zipfile
from pathlib import Path

import nbformat
from nbformat import NotebookNode

DEFAULT_OUTPUT = Path("notebooks/fudd/clip_cub_reproduction.ipynb")


def _project_files(project_root: Path) -> list[Path]:
    """Return the minimal installable project snapshot in stable order."""

    required = [project_root / "pyproject.toml", project_root / "src" / "ttvr"]
    missing = [path for path in required if not path.exists()]
    if missing:
        joined = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"Cannot build the Colab bundle; missing: {joined}")

    files = [project_root / "pyproject.toml"]
    files.extend(
        path
        for path in (project_root / "src" / "ttvr").rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
    )
    for optional_name in ("README.md", "LICENSE", "LICENSE.md"):
        optional_path = project_root / optional_name
        if optional_path.is_file():
            files.append(optional_path)
    return sorted(set(files), key=lambda path: path.relative_to(project_root).as_posix())


def _build_source_bundle(project_root: Path) -> tuple[str, str, list[str]]:
    """Create a deterministic ZIP snapshot and return base64, SHA256, and files."""

    buffer = io.BytesIO()
    bundled_files: list[str] = []
    with zipfile.ZipFile(
        buffer,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as archive:
        for path in _project_files(project_root):
            relative = path.relative_to(project_root).as_posix()
            info = zipfile.ZipInfo(relative)
            # A fixed timestamp makes the generated notebook stable when source
            # contents have not changed.
            info.date_time = (2026, 1, 1, 0, 0, 0)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, path.read_bytes())
            bundled_files.append(relative)

    bundle = buffer.getvalue()
    return (
        base64.b64encode(bundle).decode("ascii"),
        hashlib.sha256(bundle).hexdigest(),
        bundled_files,
    )


def _markdown(source: str) -> NotebookNode:
    return nbformat.v4.new_markdown_cell(textwrap.dedent(source).strip() + "\n")


def _code(source: str, *, hidden: bool = False) -> NotebookNode:
    metadata: dict[str, object] = {}
    if hidden:
        metadata = {
            "collapsed": True,
            "cellView": "form",
            "jupyter": {"source_hidden": True},
            "tags": ["hide-input"],
        }
    return nbformat.v4.new_code_cell(
        textwrap.dedent(source).strip() + "\n",
        metadata=metadata,
    )


def _bundle_install_cell(bundle_b64: str, bundle_sha256: str) -> NotebookNode:
    wrapped = "\n".join(f'    "{line}"' for line in textwrap.wrap(bundle_b64, 88))
    template = r"""
        # @title 1. Install the embedded project snapshot (generated; leave unchanged)
        import base64
        import hashlib
        import io
        import shutil
        import subprocess
        import sys
        import zipfile
        from pathlib import Path

        OPENAI_CLIP_COMMIT = "a1d071733d7111c9c014f024669f959182114e33"
        PROJECT_BUNDLE_SHA256 = "__BUNDLE_SHA256__"
        PROJECT_BUNDLE_B64 = "".join((
        __BUNDLE_B64__
        ))

        bundle_bytes = base64.b64decode(PROJECT_BUNDLE_B64)
        actual_bundle_sha256 = hashlib.sha256(bundle_bytes).hexdigest()
        assert actual_bundle_sha256 == PROJECT_BUNDLE_SHA256, (
            f"Embedded source bundle checksum mismatch: {actual_bundle_sha256}"
        )

        project_dir = Path("/content/ttvr_project")
        if project_dir.exists():
            shutil.rmtree(project_dir)
        project_dir.mkdir(parents=True)
        with zipfile.ZipFile(io.BytesIO(bundle_bytes)) as archive:
            archive.extractall(project_dir)

        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--quiet",
                "--disable-pip-version-check",
                "ftfy==6.3.1",
                "regex==2024.11.6",
                "tqdm==4.67.1",
                f"git+https://github.com/openai/CLIP.git@{OPENAI_CLIP_COMMIT}",
            ],
            check=True,
        )
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--quiet",
                "--disable-pip-version-check",
                "--no-deps",
                "--editable",
                str(project_dir),
            ],
            check=True,
        )
        print(f"Installed ttvr source bundle {PROJECT_BUNDLE_SHA256[:12]}…")
    """
    source = textwrap.dedent(template).replace("__BUNDLE_SHA256__", bundle_sha256)
    source = source.replace("__BUNDLE_B64__", wrapped)
    return _code(source, hidden=True)


def build_notebook(project_root: Path) -> NotebookNode:
    """Construct the canonical FuDD/CLIP reproduction notebook."""

    bundle_b64, bundle_sha256, bundled_files = _build_source_bundle(project_root)

    cells = [
        _markdown(
            r"""
            # Reproducing FuDD on CLIP and CUB-200-2011

            This notebook is the **CLIP positive-control reproduction** for
            *Follow-Up Differential Descriptions* (FuDD, ICLR 2024).  It runs the
            paper's two-stage inference on the official CUB test split:

            1. CLIP ranks all 200 bird classes using one class-name template.
            2. FuDD keeps the top-10 ambiguous classes and reranks them with the
               authors' cached pairwise differential descriptions.

            **Targets (Top-1):** paper `63.48% → 65.90%` (`+2.42 pp`); earlier
            independent reproduction `63.36% → 65.74%` (`+2.38 pp`).  Small
            numerical differences can occur across CUDA/PyTorch versions, so the
            final cell reports both exact values and tolerance-based checks.

            The notebook contains no experiment implementation: it installs a
            checksum-stamped snapshot of `src/ttvr` and only orchestrates its
            public API.  The long generated installation cell is collapsed by
            default.
            """
        ),
        _markdown(
            r"""
            ## Before running

            In Colab select **Runtime → Change runtime type → GPU**.  Then run the
            notebook from top to bottom.  A fresh runtime downloads roughly 1.2 GB
            of CUB data, the OpenAI CLIP checkpoint, and about 22 MB of official
            FuDD descriptions.  Expect the full 5,794-image evaluation to be the
            slow step.

            No API key or LLM call is needed: FuDD's official descriptions are
            already cached by the paper authors.
            """
        ),
        _bundle_install_cell(bundle_b64, bundle_sha256),
        _markdown(
            r"""
            ## Locked evaluation protocol

            These settings intentionally match the CLIP/CUB positive control and
            must not be tuned on the test set:

            - official CUB-200-2011 test split: **5,794 images**;
            - OpenAI CLIP **ViT-L/14@336px** and its native preprocessing;
            - the official CPU-load-then-CUDA **FP32** precision path;
            - the tokenized baseline prompt `A photo of a {class}.`;
            - FuDD top-k **10** and all official CUB pairwise descriptions;
            - a batched-vs-official-loop parity gate before reporting results;
            - inference only—no CUB train images, labels, attributes, or boxes.
            """
        ),
        _code(
            r"""
            # @title 2. GPU and environment diagnostics
            import platform
            import random
            import subprocess
            import sys
            from pathlib import Path

            import numpy as np
            import torch

            SEED = 2026
            random.seed(SEED)
            np.random.seed(SEED)
            torch.manual_seed(SEED)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(SEED)

            assert torch.cuda.is_available(), (
                "GPU not detected. In Colab choose Runtime → Change runtime type → GPU, "
                "then reconnect and rerun."
            )
            DEVICE = "cuda"
            print(f"Python: {sys.version.split()[0]} ({platform.platform()})")
            print(f"PyTorch: {torch.__version__}")
            print(f"CUDA runtime: {torch.version.cuda}")
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=name,driver_version,memory.total",
                    "--format=csv",
                ],
                check=False,
            )
            """
        ),
        _code(
            r"""
            # @title 3. Configure paths and instantiate the locked protocol
            from tqdm.auto import tqdm

            from ttvr import FuDDConfig

            class NotebookProgress:
                # Render one live progress bar per evaluation stage.

                def __init__(self):
                    self._bars = {}
                    self._completed = {}

                def __call__(self, stage, completed, total):
                    if total <= 0:
                        return
                    bar = self._bars.get(stage)
                    if bar is None or bar.total != total:
                        if bar is not None:
                            bar.close()
                        bar = tqdm(total=total, desc=stage, unit="item")
                        self._bars[stage] = bar
                        self._completed[stage] = 0
                    increment = max(0, completed - self._completed[stage])
                    bar.update(increment)
                    self._completed[stage] = completed
                    if completed >= total:
                        bar.close()

            WORK_ROOT = Path("/content/ttvr_runs/fudd_clip_cub")
            DATA_ROOT = WORK_ROOT / "data"
            PROMPT_ROOT = WORK_ROOT / "official_fudd_prompts"
            CACHE_ROOT = WORK_ROOT / "cache"
            RESULTS_ROOT = WORK_ROOT / "results"
            for directory in (DATA_ROOT, PROMPT_ROOT, CACHE_ROOT, RESULTS_ROOT):
                directory.mkdir(parents=True, exist_ok=True)

            config = FuDDConfig(
                data_root=DATA_ROOT,
                prompt_root=PROMPT_ROOT,
                cache_dir=CACHE_ROOT,
                model_name="ViT-L/14@336px",
                precision="fp32",
                top_k=10,
                batch_size=32,
                text_batch_size=256,
                num_workers=2,
                device=DEVICE,
                seed=SEED,
            )
            print(config)
            """
        ),
        _markdown(
            r"""
            ## Download and verify the paper assets

            The project downloader is pinned to FuDD commit
            `32264231fec047eb0bbbf59bfdbc8e6d208a096b` and verifies SHA-256 for:
            `cub_prompt_pairs.json` (19,900 class pairs)
            and `cub_class_names.json` (200 classes).  Re-running this cell reuses
            valid files.
            """
        ),
        _code(
            r"""
            # @title 4. Download official FuDD supplemental assets
            from ttvr import download_official_prompts, load_official_prompts

            downloaded_assets = download_official_prompts(PROMPT_ROOT)
            prompts = load_official_prompts(PROMPT_ROOT)
            print(f"Verified official FuDD assets in: {PROMPT_ROOT}")
            print(f"Classes: {len(prompts.class_names):,}")
            print(f"Differential class pairs: {prompts.pair_count:,}")
            """
        ),
        _markdown(
            r"""
            ## Download CUB and initialize CLIP

            CUB is downloaded from the official CaltechDATA record and checked
            against MD5 `97eceeb196236b17998738112f37df78`.  Dataset metadata and
            image existence are then verified before the official test split is
            exposed.  The model is initialized first so its exact 336px transform
            is passed into the dataset.
            """
        ),
        _code(
            r"""
            # @title 5. Load CLIP and the official CUB test split
            from ttvr import CLIPBackend, prepare_cub

            backend = CLIPBackend(
                model_name=config.model_name,
                device=config.device,
                precision=config.precision,
                text_batch_size=config.text_batch_size,
            )
            cub_test = prepare_cub(
                config.data_root,
                transform=backend.preprocess,
                download=True,
                verify_images=True,
                split="test",
            )
            assert len(cub_test) == 5_794, f"Expected 5,794 test images, got {len(cub_test):,}"
            print(f"CUB test split ready: {len(cub_test):,} images")
            """
        ),
        _markdown(
            r"""
            ## Smoke test

            This exercises the **same** baseline and FuDD path as the full run on
            only 32 test images.  It is an integration check, not a reported
            accuracy estimate.  The full run below always starts at index zero and
            evaluates all 5,794 images.
            """
        ),
        _code(
            r"""
            # @title 6. Run a 32-image end-to-end smoke test
            from ttvr import evaluate_cub

            smoke_report = evaluate_cub(
                cub_test,
                prompts,
                backend,
                config,
                max_samples=32,
                parity_samples=32,
                progress=NotebookProgress(),
            )
            smoke = smoke_report.to_dict()
            assert smoke["num_samples"] == 32
            print(smoke)
            """
        ),
        _markdown(
            r"""
            ## Full baseline + FuDD evaluation

            This cell is the reported experiment.  It computes baseline Top-1/5,
            FuDD Top-1/5, top-10 candidate recall, and paired transition counts
            (`recovered` is wrong→right; `degraded` is right→wrong).
            """
        ),
        _code(
            r"""
            # @title 7. Evaluate all 5,794 official test images
            full_report = evaluate_cub(
                cub_test,
                prompts,
                backend,
                config,
                max_samples=None,
                parity_samples=32,
                progress=NotebookProgress(),
            )
            full = full_report.to_dict()
            assert full["num_samples"] == 5_794
            full
            """
        ),
        _markdown(
            r"""
            ## Validate and save the reproduction

            Structural invariants are asserted.  Numeric comparisons are reported
            as checks rather than hard assertions because CUDA and library versions
            can cause small differences.  A large mismatch should be investigated;
            it must not be silently presented as a successful reproduction.
            """
        ),
        _code(
            r"""
            # @title 8. Compare with targets and write a complete result record
            import datetime as dt
            import hashlib
            import json
            import math

            from IPython.display import Markdown, display

            PAPER_TARGET = {"baseline_top1": 63.48, "fudd_top1": 65.90, "gain_pp": 2.42}
            PRIOR_REPRODUCTION = {"baseline_top1": 63.36, "fudd_top1": 65.74, "gain_pp": 2.38}

            def as_percent(value):
                value = float(value)
                return 100.0 * value if abs(value) <= 1.0 else value

            baseline_top1 = as_percent(full["baseline"]["top1"])
            fudd_top1 = as_percent(full["fudd"]["top1"])
            gain_pp = fudd_top1 - baseline_top1
            transitions = full["transfers"]
            transition_total = sum(
                int(transitions[name])
                for name in ("both_correct", "recovered", "degraded", "both_wrong")
            )

            assert transition_total == full["num_samples"] == 5_794
            assert all(math.isfinite(value) for value in (baseline_top1, fudd_top1, gain_pp))

            tolerance_pp = 0.50
            checks = {
                "official_test_size": full["num_samples"] == 5_794,
                "official_fp32": full["feature_dtype"] == "torch.float32",
                "official_reference_parity": bool(full["parity"]["passed"]),
                "prediction_count_matches": full["prediction_count"] == 5_794,
                "baseline_within_0.50pp_of_prior": (
                    abs(baseline_top1 - PRIOR_REPRODUCTION["baseline_top1"])
                    <= tolerance_pp
                ),
                "fudd_within_0.50pp_of_prior": (
                    abs(fudd_top1 - PRIOR_REPRODUCTION["fudd_top1"])
                    <= tolerance_pp
                ),
                "gain_within_0.50pp_of_paper": (
                    abs(gain_pp - PAPER_TARGET["gain_pp"]) <= tolerance_pp
                ),
                "positive_net_gain": gain_pp > 0.0,
            }
            status = "PASS" if all(checks.values()) else "REVIEW"

            rows = [
                "| Run | Baseline Top-1 | FuDD Top-1 | Gain |",
                "|---|---:|---:|---:|",
                (
                    f"| Paper | {PAPER_TARGET['baseline_top1']:.2f}% | "
                    f"{PAPER_TARGET['fudd_top1']:.2f}% | "
                    f"{PAPER_TARGET['gain_pp']:+.2f} pp |"
                ),
                (
                    "| Prior independent | "
                    f"{PRIOR_REPRODUCTION['baseline_top1']:.2f}% | "
                    f"{PRIOR_REPRODUCTION['fudd_top1']:.2f}% | "
                    f"{PRIOR_REPRODUCTION['gain_pp']:+.2f} pp |"
                ),
                f"| This notebook | {baseline_top1:.2f}% | {fudd_top1:.2f}% | {gain_pp:+.2f} pp |",
            ]
            display(Markdown("\n".join([f"### Reproduction status: **{status}**", "", *rows])))
            print("Checks:")
            for name, passed in checks.items():
                print(f"  {'✓' if passed else '✗'} {name}")

            created_at = dt.datetime.now(dt.timezone.utc)
            run_id = (
                created_at.strftime("%Y%m%dT%H%M%S.%fZ")
                + f"-full-{PROJECT_BUNDLE_SHA256[:10]}"
            )
            run_dir = RESULTS_ROOT / run_id
            run_dir.mkdir(parents=True, exist_ok=False)
            prediction_path, prediction_sha256 = full_report.write_predictions_jsonl(
                run_dir / "predictions.jsonl"
            )
            result_record = {
                "schema_version": 1,
                "run_id": run_id,
                "created_at_utc": created_at.isoformat(),
                "status": status,
                "paper_target_percent": PAPER_TARGET,
                "prior_reproduction_percent": PRIOR_REPRODUCTION,
                "checks": checks,
                "source": {
                    "project_bundle_sha256": PROJECT_BUNDLE_SHA256,
                    "openai_clip_commit": OPENAI_CLIP_COMMIT,
                    "official_fudd_commit": "32264231fec047eb0bbbf59bfdbc8e6d208a096b",
                },
                "environment": {
                    "python": sys.version,
                    "torch": torch.__version__,
                    "cuda": torch.version.cuda,
                    "gpu": torch.cuda.get_device_name(0),
                },
                "predictions": {
                    "path": prediction_path.name,
                    "rows": full["prediction_count"],
                    "sha256": prediction_sha256,
                },
                "smoke_test": smoke,
                "full_evaluation": full,
            }
            result_path = run_dir / "result.json"
            result_path.write_text(json.dumps(result_record, indent=2), encoding="utf-8")

            freeze_path = run_dir / "environment_pip_freeze.txt"
            freeze_path.write_text(
                subprocess.run(
                    [sys.executable, "-m", "pip", "freeze"],
                    check=True,
                    capture_output=True,
                    text=True,
                ).stdout,
                encoding="utf-8",
            )
            checksum_path = run_dir / "checksums.sha256"
            checksum_lines = []
            for path in sorted((prediction_path, result_path, freeze_path)):
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                checksum_lines.append(f"{digest}  {path.name}")
            checksum_path.write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")
            print(f"Saved: {result_path}")
            print(f"Saved immutable run: {run_dir}")
            """
        ),
        _code(
            r"""
            # @title 9. Download the result record and environment snapshot
            import shutil
            from google.colab import files

            archive_base = WORK_ROOT / f"fudd_clip_cub_reproduction-{run_id}"
            expected_archive = archive_base.with_suffix(".zip")
            if expected_archive.exists():
                raise FileExistsError(f"Refusing to overwrite: {expected_archive}")
            archive_path = shutil.make_archive(
                str(archive_base),
                "zip",
                run_dir,
            )
            print(f"Created: {archive_path}")
            files.download(archive_path)
            """
        ),
    ]

    notebook = nbformat.v4.new_notebook(cells=cells)
    notebook.metadata.update(
        {
            "accelerator": "GPU",
            "colab": {
                "gpuType": "T4",
                "provenance": [],
            },
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python"},
            "ttvr_build": {
                "bundled_files": bundled_files,
                "project_bundle_sha256": bundle_sha256,
                "protocol": "CUB test / OpenAI CLIP ViT-L/14@336px / FuDD top-k 10",
            },
        }
    )
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type == "code":
            compile(cell.source, f"<notebook-cell-{index}>", "exec")
    nbformat.validate(notebook)
    return notebook


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Repository root (default: inferred from this script).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output path relative to project root (default: {DEFAULT_OUTPUT}).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    output = args.output
    if not output.is_absolute():
        output = project_root / output
    output.parent.mkdir(parents=True, exist_ok=True)

    notebook = build_notebook(project_root)
    nbformat.write(notebook, output)

    # Read it back through both JSON and nbformat.  This catches truncated writes
    # and schema errors before a generated notebook is handed to Colab.
    json.loads(output.read_text(encoding="utf-8"))
    round_tripped = nbformat.read(output, as_version=4)
    nbformat.validate(round_tripped)
    print(
        f"Wrote {output} ({len(round_tripped.cells)} cells, "
        f"bundle {round_tripped.metadata.ttvr_build.project_bundle_sha256[:12]}…)"
    )


if __name__ == "__main__":
    main()
