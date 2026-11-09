"""Core cross-lingual transfer (XLT) metrics and Hardness Adjusted Transfer (HAT) Score.

Implements Section 4 and Appendices A.4, A.5, A.7, A.10 of:
"Are Multilingual Models Improving? Revisiting Cross-Lingual Transfer" (EMNLP 2026).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Mapping, Optional, Sequence, Tuple, Union

import numpy as np


ArrayLike = Union[Sequence[float], np.ndarray]


@dataclass(frozen=True)
class BootstrapEstimate:
    """Point estimate paired with a bootstrap confidence interval."""

    estimate: float
    ci_low: float
    ci_high: float
    confidence_level: float = 0.95
    n_bootstrap: int = 1000

    def format(self, decimals: int = 1, scale: float = 100.0) -> str:
        """Format estimate and CI, e.g. '84.6 [82.1, 87.1]'."""
        est = self.estimate * scale
        lo = self.ci_low * scale
        hi = self.ci_high * scale
        return f"{est:.{decimals}f} [{lo:.{decimals}f}, {hi:.{decimals}f}]"


@dataclass(frozen=True)
class TransferProfile:
    """Fitted transfer profile E[T | S = s] and goodness-of-fit diagnostics."""

    estimator: str
    hat_score: float
    slope: Optional[float]
    intercept: float
    r2: float
    mse: float
    grid_s: np.ndarray
    grid_expected_t: np.ndarray


@dataclass(frozen=True)
class EvaluationReport:
    """Complete XLT metric report with 95% bootstrap confidence intervals."""

    n_items: int
    source_accuracy: BootstrapEstimate
    target_accuracy: BootstrapEstimate
    xlt_gap: BootstrapEstimate
    transfer_score: BootstrapEstimate
    hat_score: BootstrapEstimate
    profile: TransferProfile

    def to_dict(self, scale: float = 100.0) -> Dict[str, object]:
        """Serialize report into a JSON-friendly dictionary."""
        def _pack(b: BootstrapEstimate) -> Dict[str, float]:
            return {
                "estimate": round(b.estimate * scale, 4),
                "ci_low": round(b.ci_low * scale, 4),
                "ci_high": round(b.ci_high * scale, 4),
            }

        return {
            "n_items": self.n_items,
            "scale": scale,
            "source_accuracy": _pack(self.source_accuracy),
            "target_accuracy": _pack(self.target_accuracy),
            "xlt_gap": _pack(self.xlt_gap),
            "transfer_score": _pack(self.transfer_score),
            "hat_score": _pack(self.hat_score),
            "profile": {
                "estimator": self.profile.estimator,
                "slope": round(float(self.profile.slope), 6)
                if self.profile.slope is not None
                else None,
                "intercept": round(float(self.profile.intercept), 6),
                "r2": round(float(self.profile.r2), 6),
                "mse": round(float(self.profile.mse), 6),
            },
        }


def _validate_arrays(
    source_pass_rates: ArrayLike,
    target_pass_rates: ArrayLike,
) -> Tuple[np.ndarray, np.ndarray]:
    """Validate and convert source/target pass-rate inputs into 1D float64 arrays.

    `target_pass_rates` may be 1D of shape `(n_items,)` (already averaged across
    target languages) or 2D of shape `(n_items, n_target_languages)`, in which case
    we compute `t^(i) = E_l[t_l^(i)]` across target languages per Equation (3).
    """
    s = np.asarray(source_pass_rates, dtype=np.float64)
    t_raw = np.asarray(target_pass_rates, dtype=np.float64)

    if s.ndim != 1:
        raise ValueError(f"`source_pass_rates` must be 1D, got shape {s.shape}.")

    if t_raw.ndim == 2:
        if t_raw.shape[0] != s.shape[0]:
            raise ValueError(
                f"First dimension mismatch: source has {s.shape[0]} items, "
                f"target has {t_raw.shape[0]} items."
            )
        t = np.nanmean(t_raw, axis=1)
    elif t_raw.ndim == 1:
        if t_raw.shape[0] != s.shape[0]:
            raise ValueError(
                f"Length mismatch: source has {s.shape[0]} items, "
                f"target has {t_raw.shape[0]} items."
            )
        t = t_raw
    else:
        raise ValueError(
            f"`target_pass_rates` must be 1D or 2D, got shape {t_raw.shape}."
        )

    if s.size == 0:
        raise ValueError("Pass-rate arrays must be non-empty.")

    if np.any((s < -1e-9) | (s > 1.0 + 1e-9)) or np.any(
        (t < -1e-9) | (t > 1.0 + 1e-9)
    ):
        raise ValueError("Pass-rates must lie in the interval [0, 1].")

    return np.clip(s, 0.0, 1.0), np.clip(t, 0.0, 1.0)


def compute_pass_rates(
    binary_outcomes: ArrayLike,
) -> np.ndarray:
    """Compute instance-level pass-rates across N stochastic samples.

    Args:
        binary_outcomes: Binary correctness array of shape `(n_items, n_samples)`
            or `(n_items, n_languages, n_samples)` with values in `{0, 1}`.

    Returns:
        Pass-rates averaged over the trailing sample axis (`axis=-1`).
    """
    arr = np.asarray(binary_outcomes, dtype=np.float64)
    if arr.ndim < 2:
        raise ValueError(
            "`binary_outcomes` must have at least 2 dimensions `(n_items, n_samples)`."
        )
    return np.mean(arr, axis=-1)


def source_accuracy(source_pass_rates: ArrayLike) -> float:
    """Average source-language accuracy E_D[S | M]."""
    s = np.asarray(source_pass_rates, dtype=np.float64)
    if s.ndim != 1 or s.size == 0:
        raise ValueError("`source_pass_rates` must be a non-empty 1D array.")
    return float(np.mean(s))


def target_accuracy(target_pass_rates: ArrayLike) -> float:
    """Average target-language accuracy E_{D,l}[T_l | M] (Equation 1)."""
    t = np.asarray(target_pass_rates, dtype=np.float64)
    if t.size == 0:
        raise ValueError("`target_pass_rates` must be non-empty.")
    if t.ndim == 2:
        return float(np.mean(np.nanmean(t, axis=1)))
    if t.ndim == 1:
        return float(np.mean(t))
    raise ValueError(f"`target_pass_rates` must be 1D or 2D, got shape {t.shape}.")


def xlt_gap(
    source_pass_rates: ArrayLike,
    target_pass_rates: ArrayLike,
) -> float:
    """Cross-Lingual Transfer (XLT) Gap E_{D,l}[S - T_l | M] (Section 4)."""
    s, t = _validate_arrays(source_pass_rates, target_pass_rates)
    return float(np.mean(s - t))


def transfer_score(
    source_pass_rates: ArrayLike,
    target_pass_rates: ArrayLike,
    mode: str = "strict_source_correct",
    threshold: float = 1.0,
) -> float:
    """Compute the ECLeKTic Transfer Score (Goldman et al., 2025; Section 4 & Appendix A.10).

    Args:
        source_pass_rates: Source pass-rates `s^(i)` in `[0, 1]`.
        target_pass_rates: Target pass-rates `t^(i)` in `[0, 1]` or `(n_items, n_langs)`.
        mode:
            - `"strict_source_correct"` (default, Appendix Eq. 6): conditional mean of
              target accuracy restricted to items answered strictly correctly in the
              source (`s^(i) >= threshold`, where `threshold=1.0` for discrete/pass@1
              or full pass-rate items; falls back to `"continuous"` if no item reaches
              `threshold`).
            - `"continuous"` (Section 4 theoretical integral):
              `sum_i(s^(i) * t^(i)) / sum_i(s^(i))`.
        threshold: Minimum source pass-rate for `"strict_source_correct"` mode.

    Returns:
        Transfer score in `[0, 1]` (or `0.0` if source accuracy is identically zero).
    """
    s, t = _validate_arrays(source_pass_rates, target_pass_rates)
    if mode == "strict_source_correct":
        mask = s >= (threshold - 1e-12)
        if np.any(mask):
            return float(np.mean(t[mask]))
        # Fallback if all s^(i) < threshold (e.g. noisy multi-sample pass rates < 1.0)
        mode = "continuous"

    if mode == "continuous":
        denom = float(np.sum(s))
        if denom <= 1e-12:
            return 0.0
        return float(np.sum(s * t) / denom)

    raise ValueError(
        f"Unknown mode `{mode}`. Expected 'strict_source_correct' or 'continuous'."
    )


def _r2_and_mse(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
    """Compute coefficient of determination R^2 and mean squared error."""
    residuals = y_true - y_pred
    mse = float(np.mean(residuals ** 2))
    ss_res = float(np.sum(residuals ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot <= 1e-15:
        r2 = 1.0 if ss_res <= 1e-15 else 0.0
    else:
        r2 = 1.0 - (ss_res / ss_tot)
    return r2, mse


def hat_score(
    source_pass_rates: ArrayLike,
    target_pass_rates: ArrayLike,
) -> float:
    """Compute the Hardness Adjusted Transfer (HAT) Score (Equations 3 & 4).

    Fits the constrained zero-intercept linear transfer profile `E[T | S = s] = m * s`
    via ordinary least squares through the origin:

        m_hat = argmin_m sum_i (t^(i) - m * s^(i))^2 = sum_i(s^(i) * t^(i)) / sum_i(s^(i)^2)
        HAT Score = 2 * integral_0^1 (m_hat * s) ds = m_hat

    Args:
        source_pass_rates: Instance-level source pass-rates `s^(i)` in `[0, 1]`.
        target_pass_rates: Instance-level target pass-rates `t^(i)` in `[0, 1]`,
            either 1D `(n_items,)` or 2D `(n_items, n_target_languages)`.

    Returns:
        HAT Score `m_hat` (typically in `[0, 1]`; multiply by 100 for percentage scale).
    """
    s, t = _validate_arrays(source_pass_rates, target_pass_rates)
    denom = float(np.dot(s, s))
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(s, t) / denom)


def capped_hat_score(
    source_pass_rates: ArrayLike,
    target_pass_rates: ArrayLike,
) -> float:
    """Compute target-capped HAT Score with `T' = min(T, S)` (Appendix A.5)."""
    s, t = _validate_arrays(source_pass_rates, target_pass_rates)
    return hat_score(s, np.minimum(t, s))


_trapezoid = getattr(np, "trapezoid", getattr(np, "trapz", None))


def hat_score_linear_intercept(
    source_pass_rates: ArrayLike,
    target_pass_rates: ArrayLike,
    clip_unit_interval: bool = True,
    grid_size: int = 1001,
) -> float:
    """Alternative HAT estimator using unconstrained OLS with intercept (Appendix A.4).

    Fits `f(s) = m * s + b` and evaluates `2 * integral_0^1 f(s) ds`.
    """
    s, t = _validate_arrays(source_pass_rates, target_pass_rates)
    if np.std(s) <= 1e-12:
        return float(2.0 * np.mean(t))
    slope, intercept = np.polyfit(s, t, deg=1)
    if not clip_unit_interval:
        return float(slope + 2.0 * intercept)
    grid = np.linspace(0.0, 1.0, grid_size)
    preds = np.clip(slope * grid + intercept, 0.0, 1.0)
    return float(2.0 * _trapezoid(preds, grid))


def hat_score_krr(
    source_pass_rates: ArrayLike,
    target_pass_rates: ArrayLike,
    alpha: float = 1.0,
    gamma: float = 5.0,
    clip_unit_interval: bool = True,
    grid_size: int = 1001,
) -> float:
    """Alternative HAT estimator using Kernel Ridge Regression with RBF kernel (Appendix A.4).

    Solves `(K + alpha * I) w = t` where `K_ij = exp(-gamma * (s_i - s_j)^2)` and
    numerically integrates `2 * integral_0^1 E[T | S = s] ds`.
    """
    s, t = _validate_arrays(source_pass_rates, target_pass_rates)
    n = s.shape[0]
    sq_dists = (s[:, None] - s[None, :]) ** 2
    k_mat = np.exp(-gamma * sq_dists)
    dual_coef = np.linalg.solve(k_mat + alpha * np.eye(n, dtype=np.float64), t)

    grid = np.linspace(0.0, 1.0, grid_size)
    k_grid = np.exp(-gamma * ((grid[:, None] - s[None, :]) ** 2))
    preds = np.einsum("ij,j->i", k_grid, dual_coef)
    if clip_unit_interval:
        preds = np.clip(preds, 0.0, 1.0)
    return float(2.0 * _trapezoid(preds, grid))


def transfer_profile(
    source_pass_rates: ArrayLike,
    target_pass_rates: ArrayLike,
    estimator: str = "linear_no_intercept",
    alpha: float = 1.0,
    gamma: float = 5.0,
    grid_size: int = 201,
) -> TransferProfile:
    """Fit the transfer profile curve `(s, E[T | S = s])` and return fit statistics.

    Args:
        source_pass_rates: Source pass-rates `s^(i)`.
        target_pass_rates: Target pass-rates `t^(i)`.
        estimator: One of `"linear_no_intercept"` (default, Paper Eq. 3-4),
            `"linear_intercept"`, or `"krr"`.
        alpha: Regularization parameter if `estimator="krr"`.
        gamma: RBF kernel bandwidth if `estimator="krr"`.
        grid_size: Number of points on `[0, 1]` for the curve grid.

    Returns:
        `TransferProfile` containing HAT score, slope/intercept, R^2, MSE, and curve grid.
    """
    s, t = _validate_arrays(source_pass_rates, target_pass_rates)
    grid_s = np.linspace(0.0, 1.0, grid_size)

    if estimator == "linear_no_intercept":
        m_hat = hat_score(s, t)
        y_pred = m_hat * s
        r2, mse = _r2_and_mse(t, y_pred)
        return TransferProfile(
            estimator=estimator,
            hat_score=m_hat,
            slope=m_hat,
            intercept=0.0,
            r2=r2,
            mse=mse,
            grid_s=grid_s,
            grid_expected_t=np.clip(m_hat * grid_s, 0.0, 1.0),
        )

    if estimator == "linear_intercept":
        if np.std(s) <= 1e-12:
            slope, intercept = 0.0, float(np.mean(t))
        else:
            slope, intercept = map(float, np.polyfit(s, t, deg=1))
        y_pred = slope * s + intercept
        r2, mse = _r2_and_mse(t, y_pred)
        grid_t = np.clip(slope * grid_s + intercept, 0.0, 1.0)
        score = float(2.0 * _trapezoid(grid_t, grid_s))
        return TransferProfile(
            estimator=estimator,
            hat_score=score,
            slope=slope,
            intercept=intercept,
            r2=r2,
            mse=mse,
            grid_s=grid_s,
            grid_expected_t=grid_t,
        )

    if estimator == "krr":
        n = s.shape[0]
        k_mat = np.exp(-gamma * ((s[:, None] - s[None, :]) ** 2))
        dual_coef = np.linalg.solve(k_mat + alpha * np.eye(n, dtype=np.float64), t)
        y_pred = np.einsum("ij,j->i", k_mat, dual_coef)
        r2, mse = _r2_and_mse(t, y_pred)
        k_grid = np.exp(-gamma * ((grid_s[:, None] - s[None, :]) ** 2))
        grid_t = np.clip(np.einsum("ij,j->i", k_grid, dual_coef), 0.0, 1.0)
        score = float(2.0 * _trapezoid(grid_t, grid_s))
        return TransferProfile(
            estimator=estimator,
            hat_score=score,
            slope=None,
            intercept=float(grid_t[0]),
            r2=r2,
            mse=mse,
            grid_s=grid_s,
            grid_expected_t=grid_t,
        )

    raise ValueError(
        f"Unknown estimator `{estimator}`. Choose from "
        "'linear_no_intercept', 'linear_intercept', or 'krr'."
    )


def bootstrap_ci(
    source_pass_rates: ArrayLike,
    target_pass_rates: ArrayLike,
    metric_fn: Callable[[np.ndarray, np.ndarray], float] = hat_score,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    seed: Optional[int] = 42,
) -> BootstrapEstimate:
    """Compute paired non-parametric bootstrap confidence intervals (Section 5.4; Efron, 1979).

    Resamples question indices `{0, ..., n - 1}` with replacement `n_bootstrap` times,
    preserving the pairing `(s^(i), t^(i))` across source and target languages.
    """
    s, t = _validate_arrays(source_pass_rates, target_pass_rates)
    point_est = float(metric_fn(s, t))

    rng = np.random.default_rng(seed)
    n = s.shape[0]
    samples = np.empty(n_bootstrap, dtype=np.float64)

    # Fast vectorized path for the primary metrics
    if metric_fn is hat_score:
        idx = rng.integers(0, n, size=(n_bootstrap, n))
        s_boot = s[idx]
        t_boot = t[idx]
        denom = np.sum(s_boot * s_boot, axis=1)
        numer = np.sum(s_boot * t_boot, axis=1)
        samples = np.where(denom > 1e-12, numer / denom, 0.0)
    else:
        for b in range(n_bootstrap):
            idx = rng.integers(0, n, size=n)
            samples[b] = metric_fn(s[idx], t[idx])

    alpha = (1.0 - confidence_level) / 2.0
    ci_low, ci_high = np.percentile(samples, [100.0 * alpha, 100.0 * (1.0 - alpha)])
    return BootstrapEstimate(
        estimate=point_est,
        ci_low=float(ci_low),
        ci_high=float(ci_high),
        confidence_level=confidence_level,
        n_bootstrap=n_bootstrap,
    )


def evaluate_instances(
    source_pass_rates: ArrayLike,
    target_pass_rates: ArrayLike,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    seed: Optional[int] = 42,
    transfer_score_mode: str = "strict_source_correct",
) -> EvaluationReport:
    """Compute all XLT metrics, 95% bootstrap CIs, and the HAT transfer profile."""
    s, t = _validate_arrays(source_pass_rates, target_pass_rates)

    src_est = bootstrap_ci(
        s, t, metric_fn=lambda sb, _tb: float(np.mean(sb)),
        n_bootstrap=n_bootstrap, confidence_level=confidence_level, seed=seed,
    )
    tgt_est = bootstrap_ci(
        s, t, metric_fn=lambda _sb, tb: float(np.mean(tb)),
        n_bootstrap=n_bootstrap, confidence_level=confidence_level, seed=seed,
    )
    gap_est = bootstrap_ci(
        s, t, metric_fn=xlt_gap,
        n_bootstrap=n_bootstrap, confidence_level=confidence_level, seed=seed,
    )
    ts_est = bootstrap_ci(
        s, t, metric_fn=lambda sb, tb: transfer_score(sb, tb, mode=transfer_score_mode),
        n_bootstrap=n_bootstrap, confidence_level=confidence_level, seed=seed,
    )
    hat_est = bootstrap_ci(
        s, t, metric_fn=hat_score,
        n_bootstrap=n_bootstrap, confidence_level=confidence_level, seed=seed,
    )
    prof = transfer_profile(s, t, estimator="linear_no_intercept")

    return EvaluationReport(
        n_items=int(s.shape[0]),
        source_accuracy=src_est,
        target_accuracy=tgt_est,
        xlt_gap=gap_est,
        transfer_score=ts_est,
        hat_score=hat_est,
        profile=prof,
    )


def spearman_rank_correlation(x: ArrayLike, y: ArrayLike) -> float:
    """Compute Spearman rank correlation rho (used in Appendix A.4 Table 8)."""
    xa = np.asarray(x, dtype=np.float64)
    ya = np.asarray(y, dtype=np.float64)
    if xa.shape != ya.shape or xa.ndim != 1 or xa.size < 2:
        raise ValueError("`x` and `y` must be 1D arrays of matching length >= 2.")

    def _ranks(v: np.ndarray) -> np.ndarray:
        order = np.argsort(v, kind="mergesort")
        ranks = np.empty_like(v, dtype=np.float64)
        ranks[order] = np.arange(v.size, dtype=np.float64)
        # Average ranks for ties
        sorted_v = v[order]
        i = 0
        while i < v.size:
            j = i + 1
            while j < v.size and sorted_v[j] == sorted_v[i]:
                j += 1
            if j > i + 1:
                ranks[order[i:j]] = 0.5 * (i + j - 1)
            i = j
        return ranks

    rx = _ranks(xa)
    ry = _ranks(ya)
    rx_c = rx - np.mean(rx)
    ry_c = ry - np.mean(ry)
    denom = float(np.sqrt(np.sum(rx_c ** 2) * np.sum(ry_c ** 2)))
    if denom <= 1e-15:
        return 0.0
    return float(np.sum(rx_c * ry_c) / denom)
