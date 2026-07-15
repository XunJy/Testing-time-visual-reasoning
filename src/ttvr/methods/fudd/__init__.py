"""Follow-Up Differential Descriptions (FuDD)."""

from .config import OFFICIAL_CLIP_MODEL, OFFICIAL_CLIP_PRECISION, FuDDConfig
from .evaluation import (
    CandidateRerankResult,
    EvaluationReport,
    FuDDMetrics,
    ParityReport,
    PredictionRecord,
    evaluate_cub,
    rerank_candidates,
    run_clip_cub_experiment,
)
from .prompts import (
    CUB_CLASS_COUNT,
    CUB_PAIR_COUNT,
    DEFAULT_TEMPLATE,
    FUDD_OFFICIAL_COMMIT,
    ClassPairDescriptions,
    CubPromptRepository,
    DifferentialDescription,
    download_official_prompts,
    load_official_prompts,
    load_pair_overrides,
    verify_official_prompt_assets,
)

__all__ = [
    "CUB_CLASS_COUNT",
    "CUB_PAIR_COUNT",
    "CandidateRerankResult",
    "ClassPairDescriptions",
    "CubPromptRepository",
    "DEFAULT_TEMPLATE",
    "DifferentialDescription",
    "EvaluationReport",
    "FuDDConfig",
    "FUDD_OFFICIAL_COMMIT",
    "FuDDMetrics",
    "OFFICIAL_CLIP_MODEL",
    "OFFICIAL_CLIP_PRECISION",
    "ParityReport",
    "PredictionRecord",
    "download_official_prompts",
    "evaluate_cub",
    "load_pair_overrides",
    "load_official_prompts",
    "rerank_candidates",
    "run_clip_cub_experiment",
    "verify_official_prompt_assets",
]
