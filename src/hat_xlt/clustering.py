"""Temporal progress analysis and Algorithm 1: Two-Stage Method for Model Clustering.

Implements Section 6.3 and Appendices A.3 and A.8 of:
"Are Multilingual Models Improving? Revisiting Cross-Lingual Transfer" (EMNLP 2026).
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class ClusterFit:
    """Linear regression fit and summary metrics for a model cluster."""

    models: Tuple[str, ...]
    slope: float
    intercept: float
    mse: float
    r2: float
    median_hat: Optional[float]


@dataclass(frozen=True)
class DualFitResult:
    """Result of Algorithm 1 (Two-Stage Method for Model Clustering)."""

    cutoff_date: str
    old_cluster: ClusterFit
    new_cluster: ClusterFit
    wmse_total: float
    single_line_mse: float
    single_line_r2: float


def _fit_line(
    points: Sequence[Tuple[float, float]],
    fit_intercept: bool = True,
) -> Tuple[float, float, float, float]:
    """Fit `t = slope * s + intercept` and return `(slope, intercept, mse, r2)`."""
    arr = np.asarray(points, dtype=np.float64)
    s = arr[:, 0]
    t = arr[:, 1]
    if s.size == 0:
        return 0.0, 0.0, 0.0, 0.0
    if s.size == 1 or np.std(s) <= 1e-12:
        slope = float(np.dot(s, t) / max(float(np.dot(s, s)), 1e-12))
        intercept = 0.0
    elif fit_intercept:
        slope, intercept = map(float, np.polyfit(s, t, deg=1))
    else:
        slope = float(np.dot(s, t) / max(float(np.dot(s, s)), 1e-12))
        intercept = 0.0

    pred = slope * s + intercept
    residuals = t - pred
    mse = float(np.mean(residuals ** 2))
    ss_res = float(np.sum(residuals ** 2))
    ss_tot = float(np.sum((t - np.mean(t)) ** 2))
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 1e-15 else 1.0
    return slope, intercept, mse, r2


def two_stage_model_clustering(
    model_scores: Mapping[str, Tuple[float, float]],
    model_cutoffs: Mapping[str, str],
    cutoff_date: str = "2025-01",
    min_new_cluster_size: int = 3,
    fit_intercept: bool = True,
    model_hat_scores: Optional[Mapping[str, float]] = None,
) -> DualFitResult:
    """Run Algorithm 1: Two-Stage Method for Model Clustering (Appendix A.3).

    Stage I (Temporal Initialization):
        C_old      <- {m in M | release(m) < cutoff_date}
        C_base_new <- {m in M | release(m) >= cutoff_date}

    Stage II (MSE Minimization):
        Searches over all subsets `S' subseteq C_base_new` with `|S'| >= min_new_cluster_size`,
        allowing post-cutoff models that remain on the older trajectory to merge into
        `C_candidate_old = C_old U (C_base_new \\ S')`, minimizing weighted MSE:
            WMSE_total = |C_old_cand| * MSE(LinearFit(C_old_cand))
                       + |C_new_cand| * MSE(LinearFit(C_new_cand))

    Args:
        model_scores: Mapping `model_id -> (source_accuracy, target_accuracy)`.
        model_cutoffs: Mapping `model_id -> "YYYY-MM"` knowledge cutoff date.
        cutoff_date: Temporal threshold `"YYYY-MM"` (default `"2025-01"`).
        min_new_cluster_size: Minimum subset size `|S'|` (default `3` per Algorithm 1 line 4).
        fit_intercept: Whether `LinearFit(C)` includes an intercept (default `True` per Figure 4).
        model_hat_scores: Optional mapping `model_id -> hat_score` to report cluster medians.

    Returns:
        `DualFitResult` containing optimal `C*_old` (Red / Fit 1) and `C*_new` (Blue / Fit 2).
    """
    models = list(model_scores.keys())
    c_old_init = [m for m in models if model_cutoffs[m] < cutoff_date]
    c_base_new = [m for m in models if model_cutoffs[m] >= cutoff_date]

    if len(c_old_init) == 0:
        raise ValueError(f"No models have cutoff strictly before {cutoff_date}.")
    if len(c_base_new) < min_new_cluster_size:
        raise ValueError(
            f"Need at least {min_new_cluster_size} post-cutoff models, got {len(c_base_new)}."
        )

    all_pts = [model_scores[m] for m in models]
    _, _, single_mse, single_r2 = _fit_line(all_pts, fit_intercept=fit_intercept)

    best_wmse = float("inf")
    best_old: Tuple[str, ...] = tuple(c_old_init)
    best_new: Tuple[str, ...] = tuple(c_base_new)

    base_new_set = set(c_base_new)
    for k in range(min_new_cluster_size, len(c_base_new) + 1):
        for s_prime in combinations(c_base_new, k):
            s_prime_set = set(s_prime)
            cand_new = list(s_prime)
            cand_old = c_old_init + [m for m in c_base_new if m not in s_prime_set]

            _, _, mse_old, _ = _fit_line(
                [model_scores[m] for m in cand_old], fit_intercept=fit_intercept
            )
            _, _, mse_new, _ = _fit_line(
                [model_scores[m] for m in cand_new], fit_intercept=fit_intercept
            )
            wmse = len(cand_old) * mse_old + len(cand_new) * mse_new

            if wmse < best_wmse:
                best_wmse = wmse
                best_old = tuple(cand_old)
                best_new = tuple(cand_new)

    def _build_cluster(cluster_models: Tuple[str, ...]) -> ClusterFit:
        slope, intercept, mse, r2 = _fit_line(
            [model_scores[m] for m in cluster_models], fit_intercept=fit_intercept
        )
        med_hat = (
            float(np.median([model_hat_scores[m] for m in cluster_models]))
            if model_hat_scores is not None
            else None
        )
        return ClusterFit(
            models=cluster_models,
            slope=slope,
            intercept=intercept,
            mse=mse,
            r2=r2,
            median_hat=med_hat,
        )

    return DualFitResult(
        cutoff_date=cutoff_date,
        old_cluster=_build_cluster(best_old),
        new_cluster=_build_cluster(best_new),
        wmse_total=best_wmse,
        single_line_mse=single_mse,
        single_line_r2=single_r2,
    )


def strict_temporal_split(
    model_scores: Mapping[str, Tuple[float, float]],
    model_cutoffs: Mapping[str, str],
    cutoff_date: str = "2025-01",
    fit_intercept: bool = True,
    model_hat_scores: Optional[Mapping[str, float]] = None,
) -> DualFitResult:
    """Unoptimized strict temporal partition without subset reclassification (Appendix A.8 Table 12)."""
    models = list(model_scores.keys())
    c_old = tuple(m for m in models if model_cutoffs[m] < cutoff_date)
    c_new = tuple(m for m in models if model_cutoffs[m] >= cutoff_date)
    if not c_old or not c_new:
        raise ValueError("Both pre-cutoff and post-cutoff cohorts must be non-empty.")

    all_pts = [model_scores[m] for m in models]
    _, _, single_mse, single_r2 = _fit_line(all_pts, fit_intercept=fit_intercept)

    slope_o, int_o, mse_o, r2_o = _fit_line(
        [model_scores[m] for m in c_old], fit_intercept=fit_intercept
    )
    slope_n, int_n, mse_n, r2_n = _fit_line(
        [model_scores[m] for m in c_new], fit_intercept=fit_intercept
    )

    old_fit = ClusterFit(
        models=c_old,
        slope=slope_o,
        intercept=int_o,
        mse=mse_o,
        r2=r2_o,
        median_hat=float(np.median([model_hat_scores[m] for m in c_old]))
        if model_hat_scores is not None
        else None,
    )
    new_fit = ClusterFit(
        models=c_new,
        slope=slope_n,
        intercept=int_n,
        mse=mse_n,
        r2=r2_n,
        median_hat=float(np.median([model_hat_scores[m] for m in c_new]))
        if model_hat_scores is not None
        else None,
    )
    return DualFitResult(
        cutoff_date=cutoff_date,
        old_cluster=old_fit,
        new_cluster=new_fit,
        wmse_total=len(c_old) * mse_o + len(c_new) * mse_n,
        single_line_mse=single_mse,
        single_line_r2=single_r2,
    )


def joint_cross_task_clustering(
    task_model_scores: Mapping[str, Mapping[str, Tuple[float, float]]],
    model_cutoffs: Mapping[str, str],
    cutoff_date: str = "2025-01",
    min_new_cluster_size: int = 3,
    fit_intercept: bool = True,
) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    """Joint subset selection across multiple tasks simultaneously (Appendix A.8).

    Selects a single subset `S' subseteq C_base_new` only if migrating `C_base_new \\ S'`
    strictly improves or matches WMSE across ALL tasks simultaneously; otherwise retains
    the strict release-date partition.
    """
    tasks = list(task_model_scores.keys())
    if not tasks:
        raise ValueError("`task_model_scores` must contain at least one task.")

    models = list(next(iter(task_model_scores.values())).keys())
    c_old_init = [m for m in models if model_cutoffs[m] < cutoff_date]
    c_base_new = [m for m in models if model_cutoffs[m] >= cutoff_date]

    baseline_wmse: Dict[str, float] = {}
    for task, scores in task_model_scores.items():
        _, _, mse_o, _ = _fit_line([scores[m] for m in c_old_init], fit_intercept=fit_intercept)
        _, _, mse_n, _ = _fit_line([scores[m] for m in c_base_new], fit_intercept=fit_intercept)
        baseline_wmse[task] = len(c_old_init) * mse_o + len(c_base_new) * mse_n

    best_sum_wmse = sum(baseline_wmse.values())
    best_old = tuple(c_old_init)
    best_new = tuple(c_base_new)

    for k in range(min_new_cluster_size, len(c_base_new) + 1):
        for s_prime in combinations(c_base_new, k):
            s_prime_set = set(s_prime)
            cand_new = list(s_prime)
            cand_old = c_old_init + [m for m in c_base_new if m not in s_prime_set]

            task_wmses = {}
            improves_all = True
            for task, scores in task_model_scores.items():
                _, _, mse_o, _ = _fit_line(
                    [scores[m] for m in cand_old], fit_intercept=fit_intercept
                )
                _, _, mse_n, _ = _fit_line(
                    [scores[m] for m in cand_new], fit_intercept=fit_intercept
                )
                w = len(cand_old) * mse_o + len(cand_new) * mse_n
                if w > baseline_wmse[task] + 1e-12:
                    improves_all = False
                    break
                task_wmses[task] = w

            if improves_all and sum(task_wmses.values()) < best_sum_wmse:
                best_sum_wmse = sum(task_wmses.values())
                best_old = tuple(cand_old)
                best_new = tuple(cand_new)

    return best_old, best_new


def out_of_sample_clustering_validation(
    instance_pass_rates: Mapping[str, Tuple[np.ndarray, np.ndarray]],
    model_cutoffs: Mapping[str, str],
    cutoff_date: str = "2025-01",
    train_fraction: float = 0.5,
    seed: int = 42,
) -> Dict[str, object]:
    """Out-of-sample train/validation split validation for Algorithm 1 (Appendix A.8 Table 13).

    Args:
        instance_pass_rates: Mapping `model_id -> (s_array, t_array)` over shared dataset items.
        model_cutoffs: Mapping `model_id -> "YYYY-MM"`.
        cutoff_date: Temporal cutoff date.
        train_fraction: Fraction of items used to select clusters; remaining held out.
        seed: Random seed for item split.

    Returns:
        Dictionary with selected clusters on train split and held-out median HAT scores.
    """
    from hat_xlt.metrics import hat_score

    models = list(instance_pass_rates.keys())
    n_items = instance_pass_rates[models[0]][0].shape[0]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_items)
    n_train = max(1, int(round(train_fraction * n_items)))
    train_idx = perm[:n_train]
    val_idx = perm[n_train:]

    train_scores = {}
    val_hats = {}
    for m in models:
        s, t = instance_pass_rates[m]
        train_scores[m] = (float(np.mean(s[train_idx])), float(np.mean(t[train_idx])))
        val_hats[m] = hat_score(s[val_idx], t[val_idx])

    train_fit = two_stage_model_clustering(
        train_scores, model_cutoffs, cutoff_date=cutoff_date
    )
    fit1_models = train_fit.old_cluster.models
    fit2_models = train_fit.new_cluster.models

    return {
        "fit1_models": fit1_models,
        "fit2_models": fit2_models,
        "val_fit1_median_hat": float(np.median([val_hats[m] for m in fit1_models])),
        "val_fit2_median_hat": float(np.median([val_hats[m] for m in fit2_models])),
        "val_model_hats": val_hats,
    }
