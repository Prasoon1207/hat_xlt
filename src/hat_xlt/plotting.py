"""Publication-quality plotting utilities for HAT transfer profiles and paper figures."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from hat_xlt.clustering import DualFitResult
from hat_xlt.metrics import ArrayLike, _validate_arrays, transfer_profile


def plot_transfer_profile(
    source_pass_rates: ArrayLike,
    target_pass_rates: ArrayLike,
    model_name: str = "Model",
    dataset_name: str = "",
    estimator: str = "linear_no_intercept",
    show_source_histogram: bool = True,
    save_path: Optional[Union[str, Path]] = None,
):
    """Plot a model's HAT Transfer Profile `E[T | S = s]` alongside `y = x` parity.

    Matches the visualization layout of Figure 2 and Appendix A.9 in the paper.
    """
    import matplotlib.pyplot as plt

    s, t = _validate_arrays(source_pass_rates, target_pass_rates)
    prof = transfer_profile(s, t, estimator=estimator)

    fig, ax = plt.subplots(figsize=(5.5, 5.2), dpi=150)

    if show_source_histogram:
        ax_hist = ax.twinx()
        ax_hist.hist(
            s,
            bins=np.linspace(0.0, 1.0, 12),
            color="#3b82f6",
            alpha=0.14,
            edgecolor="#3b82f6",
            linewidth=0.6,
        )
        ax_hist.set_yticks([])
        ax_hist.set_ylim(0, max(10, int(s.size * 1.6)))
        ax_hist.spines["top"].set_visible(False)
        ax_hist.spines["right"].set_visible(False)

    # Empirical binned means for visual clarity alongside raw points
    bins = np.linspace(0.0, 1.0, 12)
    bin_idx = np.digitize(s, bins) - 1
    bx, by = [], []
    for b in range(len(bins) - 1):
        mask = bin_idx == b
        if np.any(mask):
            bx.append(float(np.mean(s[mask])))
            by.append(float(np.mean(t[mask])))
    ax.scatter(
        bx,
        by,
        s=34,
        color="#1e40af",
        alpha=0.8,
        zorder=4,
        label="Data Points (Binned)",
    )

    ax.plot([0, 1], [0, 1], "--", color="#334155", linewidth=1.2, label="Parity (y = x)")
    ax.plot(
        prof.grid_s,
        prof.grid_expected_t,
        color="#dc2626",
        linewidth=2.2,
        zorder=5,
        label=(
            f"HAT Score = {prof.hat_score:.4f}\n"
            f"R² = {prof.r2:.4f}, MSE = {prof.mse:.5f}"
        ),
    )
    ax.fill_between(
        prof.grid_s,
        0.0,
        prof.grid_expected_t,
        color="#fca5a5",
        alpha=0.28,
        zorder=2,
    )

    title = f"{model_name}" + (f" — {dataset_name}" if dataset_name else "")
    ax.set_title(title, fontsize=11.5, fontweight="bold", pad=10)
    ax.set_xlabel("Source Accuracy (s)", fontsize=10.5)
    ax.set_ylabel("Expected Target Accuracy E[T | S = s]", fontsize=10.5)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, linestyle=":", alpha=0.45)
    ax.legend(loc="upper left", fontsize=8.5, frameon=True, facecolor="white", framealpha=0.9)

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")
    return fig, ax


def plot_accuracy_on_the_line(
    benchmark_pairs: Mapping[str, Mapping[str, Tuple[float, float]]],
    save_path: Optional[Union[str, Path]] = None,
):
    """Plot Source Accuracy vs. Target Accuracy across benchmarks (Figure 1)."""
    import matplotlib.pyplot as plt

    n_cols = len(benchmark_pairs)
    fig, axes = plt.subplots(1, n_cols, figsize=(4.8 * n_cols, 4.4), dpi=150)
    if n_cols == 1:
        axes = [axes]

    for ax, (bench_name, pairs) in zip(axes, benchmark_pairs.items()):
        pts = np.asarray(list(pairs.values()), dtype=np.float64)
        s, t = pts[:, 0], pts[:, 1]
        slope, intercept = np.polyfit(s, t, deg=1)
        pred = slope * s + intercept
        ss_res = float(np.sum((t - pred) ** 2))
        ss_tot = float(np.sum((t - np.mean(t)) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0

        grid = np.linspace(0.0, 1.0, 100)
        ax.plot([0, 1], [0, 1], "--", color="#475569", linewidth=1.2, label="Perfect Transfer")
        ax.plot(grid, np.clip(slope * grid + intercept, 0.0, 1.0), color="#dc2626", linewidth=2.0, label="Best Linear Fit")
        ax.scatter(s, t, s=36, color="#2563eb", edgecolor="white", linewidth=0.6, zorder=4)
        ax.text(
            0.05, 0.90, f"R² = {r2:.3f}",
            transform=ax.transAxes,
            fontsize=10,
            fontweight="bold",
            color="#1e293b",
        )
        ax.set_title(bench_name, fontsize=11.5, fontweight="bold")
        ax.set_xlabel("Source Accuracy")
        ax.set_ylabel("Target Accuracy")
        ax.set_xlim(0.0, 1.02)
        ax.set_ylim(0.0, 1.02)
        ax.grid(True, linestyle=":", alpha=0.45)

    axes[0].legend(loc="lower right", fontsize=8.5)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")
    return fig, axes


def plot_dual_fit(
    model_scores: Mapping[str, Tuple[float, float]],
    dual_fit: DualFitResult,
    title: str = "",
    save_path: Optional[Union[str, Path]] = None,
):
    """Plot the Algorithm 1 pair-of-lines temporal fit (Figure 4)."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.2, 4.8), dpi=150)
    grid = np.linspace(0.0, 1.0, 100)

    ax.plot([0, 1], [0, 1], "--", color="#64748b", linewidth=1.1, label="y = x")

    old_pts = np.asarray([model_scores[m] for m in dual_fit.old_cluster.models])
    new_pts = np.asarray([model_scores[m] for m in dual_fit.new_cluster.models])

    ax.scatter(
        old_pts[:, 0], old_pts[:, 1],
        color="#dc2626", marker="^", s=46, alpha=0.88, zorder=4,
        label=f"Fit 1 / Red (R²={dual_fit.old_cluster.r2:.3f})",
    )
    ax.plot(
        grid,
        np.clip(dual_fit.old_cluster.slope * grid + dual_fit.old_cluster.intercept, 0.0, 1.0),
        color="#dc2626", linewidth=1.8,
    )

    ax.scatter(
        new_pts[:, 0], new_pts[:, 1],
        color="#2563eb", marker="s", s=44, alpha=0.88, zorder=5,
        label=f"Fit 2 / Blue (R²={dual_fit.new_cluster.r2:.3f})",
    )
    ax.plot(
        grid,
        np.clip(dual_fit.new_cluster.slope * grid + dual_fit.new_cluster.intercept, 0.0, 1.0),
        color="#2563eb", linewidth=1.8,
    )

    if title:
        ax.set_title(title, fontsize=11.5, fontweight="bold")
    ax.set_xlabel("Source Accuracy")
    ax.set_ylabel("Target Accuracy")
    ax.set_xlim(0.0, 1.02)
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, linestyle=":", alpha=0.45)
    ax.legend(loc="lower right", fontsize=8.5)

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")
    return fig, ax
