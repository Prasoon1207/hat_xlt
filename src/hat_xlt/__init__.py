"""HAT-XLT: Hardness Adjusted Transfer Score for Cross-Lingual Evaluation.

Reference implementation for:
"Are Multilingual Models Improving? Revisiting Cross-Lingual Transfer"
Prasoon Bajpai, Eleftheria Briakou, Colin Cherry, Preethi Jyothi, Vihari Piratla
EMNLP 2026 | https://arxiv.org/abs/2606.21954 | https://prasoon1207.github.io/hat_xlt
"""

from hat_xlt.metrics import (
    BootstrapEstimate,
    EvaluationReport,
    TransferProfile,
    bootstrap_ci,
    capped_hat_score,
    compute_pass_rates,
    evaluate_instances,
    hat_score,
    hat_score_krr,
    hat_score_linear_intercept,
    source_accuracy,
    spearman_rank_correlation,
    target_accuracy,
    transfer_profile,
    transfer_score,
    xlt_gap,
)
from hat_xlt.clustering import (
    ClusterFit,
    DualFitResult,
    joint_cross_task_clustering,
    out_of_sample_clustering_validation,
    strict_temporal_split,
    two_stage_model_clustering,
)
from hat_xlt.prompts import (
    ECLEKTIC_JUDGE_PROMPT,
    ECLEKTIC_TASK_PROMPT,
    MGSMV2_JUDGE_PROMPT,
    MGSMV2_TASK_PROMPT,
    MMLU_PROX_LITE_TASK_PROMPT,
    extract_mmlu_choice,
    extract_year_deterministic,
    format_eclektic_judge_prompt,
    format_eclektic_prompt,
    format_mgsmv2_judge_prompt,
    format_mgsmv2_prompt,
    format_mmlu_prox_lite_prompt,
    majority_vote_judges,
    parse_eclektic_judge_grade,
    parse_mgsmv2_judge_grade,
)
from hat_xlt.datasets import (
    DATASET_LANGUAGES,
    LANGUAGE_SCRIPTS,
    load_instances_jsonl,
    partition_by_script_congruence,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    # Metrics
    "BootstrapEstimate",
    "EvaluationReport",
    "TransferProfile",
    "compute_pass_rates",
    "source_accuracy",
    "target_accuracy",
    "xlt_gap",
    "transfer_score",
    "hat_score",
    "hat_score_linear_intercept",
    "hat_score_krr",
    "capped_hat_score",
    "transfer_profile",
    "bootstrap_ci",
    "evaluate_instances",
    "spearman_rank_correlation",
    # Clustering
    "ClusterFit",
    "DualFitResult",
    "two_stage_model_clustering",
    "strict_temporal_split",
    "joint_cross_task_clustering",
    "out_of_sample_clustering_validation",
    # Prompts & Judges
    "ECLEKTIC_TASK_PROMPT",
    "MGSMV2_TASK_PROMPT",
    "MMLU_PROX_LITE_TASK_PROMPT",
    "ECLEKTIC_JUDGE_PROMPT",
    "MGSMV2_JUDGE_PROMPT",
    "format_eclektic_prompt",
    "format_mgsmv2_prompt",
    "format_mmlu_prox_lite_prompt",
    "format_eclektic_judge_prompt",
    "format_mgsmv2_judge_prompt",
    "parse_eclektic_judge_grade",
    "parse_mgsmv2_judge_grade",
    "extract_mmlu_choice",
    "extract_year_deterministic",
    "majority_vote_judges",
    # Datasets
    "DATASET_LANGUAGES",
    "LANGUAGE_SCRIPTS",
    "partition_by_script_congruence",
    "load_instances_jsonl",
]
