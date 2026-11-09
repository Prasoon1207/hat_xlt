#!/usr/bin/env python3
"""Reproduce headline empirical analyses from the EMNLP 2026 HAT paper.

Reproduces:
  1. Accuracy-on-the-Line coupling (Figure 1, R^2 > 0.94 across all 3 benchmarks)
  2. Algorithm 1 Two-Stage Model Clustering (Figure 4 & Table 7)
  3. Strict Pre/Post-Jan 2025 Temporal Cohort Comparison
  4. Gemma-3 Model Size Scaling Ladder (Figure 3)
"""

from pathlib import Path
import argparse

from hat_xlt.clustering import strict_temporal_split, two_stage_model_clustering
from hat_xlt.paper_results import (
    BENCHMARK_TABLES,
    MODEL_CUTOFFS,
    MODEL_METADATA,
    get_model_hats,
    get_source_target_pairs,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce EMNLP 2026 HAT paper analyses")
    parser.add_argument(
        "--save-plots-dir",
        type=Path,
        default=None,
        help="Optional directory to save reproduced matplotlib plots",
    )
    args = parser.parse_args()

    id_to_name = {m["id"]: m["name"] for m in MODEL_METADATA}

    print("1. ACCURACY-ON-THE-LINE COUPLING & ALGORITHM 1 TEMPORAL CLUSTERING")
    print("=" * 76)
    all_pairs = {}
    for bench in ["eclektic", "mgsmv2", "mmlu_prox_lite"]:
        pairs = get_source_target_pairs(bench, scale=0.01)
        all_pairs[bench] = pairs
        hats = get_model_hats(bench, scale=1.0)

        dual = two_stage_model_clustering(
            pairs, MODEL_CUTOFFS, cutoff_date="2025-01", model_hat_scores=hats
        )
        strict = strict_temporal_split(
            pairs, MODEL_CUTOFFS, cutoff_date="2025-01", model_hat_scores=hats
        )

        print(f"\nBenchmark: {bench}")
        print(f"  Overall Single-Line Fit R^2         : {dual.single_line_r2:.3f}")
        print(
            f"  Strict Temporal Split Median HAT    : "
            f"Pre-Jan 2025 = {strict.old_cluster.median_hat:.1f} -> "
            f"Post-Jan 2025 = {strict.new_cluster.median_hat:.1f}"
        )
        print(
            f"  Algorithm 1 Fit 1 (Red, n={len(dual.old_cluster.models):2d})       : "
            f"R^2 = {dual.old_cluster.r2:.3f}, Median HAT = {dual.old_cluster.median_hat:.1f}"
        )
        print(
            f"  Algorithm 1 Fit 2 (Blue, n={len(dual.new_cluster.models):2d})      : "
            f"R^2 = {dual.new_cluster.r2:.3f}, Median HAT = {dual.new_cluster.median_hat:.1f}"
        )
        print(
            "  Fit 2 (Blue) Models                 : "
            + ", ".join(id_to_name[m] for m in dual.new_cluster.models)
        )

        if args.save_plots_dir is not None:
            from hat_xlt.plotting import plot_dual_fit

            args.save_plots_dir.mkdir(parents=True, exist_ok=True)
            plot_dual_fit(
                pairs,
                dual,
                title=f"Dual-Fit Clustering: {bench}",
                save_path=args.save_plots_dir / f"dual_fit_{bench}.png",
            )

    if args.save_plots_dir is not None:
        from hat_xlt.plotting import plot_accuracy_on_the_line

        plot_accuracy_on_the_line(
            {
                "ECLeKTic": all_pairs["eclektic"],
                "MGSMv2": all_pairs["mgsmv2"],
                "MMLU-ProX-Lite": all_pairs["mmlu_prox_lite"],
            },
            save_path=args.save_plots_dir / "accuracy_on_the_line.png",
        )

    print("\n2. GEMMA-3 SCALING LADDER (Section 6.2, Figure 3)")
    print("=" * 76)
    gemma3_ids = ["gemma-3-1b", "gemma-3-4b", "gemma-3-12b", "gemma-3-27b"]
    for bench in ["eclektic", "mmlu_prox_lite", "mgsmv2"]:
        print(f"\n  {bench}:")
        for mid in gemma3_ids:
            row = BENCHMARK_TABLES[bench][mid]
            print(
                f"    {id_to_name[mid]:<14} | Target Acc: {row['target'][0]:5.1f}% "
                f"| XLT Gap: {row['xlt_gap'][0]:5.1f}% "
                f"| HAT Score: {row['hat'][0]:5.1f}"
            )


if __name__ == "__main__":
    main()
