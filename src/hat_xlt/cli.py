"""Command-line interface (`hat-eval`) for computing HAT Score and reproducing paper analyses."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

from hat_xlt.clustering import two_stage_model_clustering
from hat_xlt.datasets import load_instances_jsonl
from hat_xlt.metrics import (
    capped_hat_score,
    evaluate_instances,
    hat_score_krr,
    hat_score_linear_intercept,
)
from hat_xlt.paper_results import (
    BENCHMARK_TABLES,
    MODEL_CUTOFFS,
    MODEL_METADATA,
    get_model_hats,
    get_source_target_pairs,
)


def _cmd_score(args: argparse.Namespace) -> int:
    s, t = load_instances_jsonl(
        args.input,
        target_languages=args.langs.split(",") if args.langs else None,
        script_filter=args.script_filter,
    )
    report = evaluate_instances(
        s,
        t,
        n_bootstrap=args.bootstrap,
        confidence_level=args.ci,
        seed=args.seed,
    )

    out = report.to_dict(scale=100.0)
    if args.all_estimators:
        out["alternative_estimators"] = {
            "linear_with_intercept": round(hat_score_linear_intercept(s, t) * 100.0, 2),
            "krr_rbf": round(hat_score_krr(s, t) * 100.0, 2),
            "target_capped": round(capped_hat_score(s, t) * 100.0, 2),
        }

    if args.json:
        print(json.dumps(out, indent=2))
    else:
        print(f"Evaluated {report.n_items} instances ({args.bootstrap} bootstrap resamples)")
        print("-" * 60)
        print(f"Source Accuracy (%) : {report.source_accuracy.format(1)}")
        print(f"Target Accuracy (%) : {report.target_accuracy.format(1)}")
        print(f"XLT Gap (%)         : {report.xlt_gap.format(1)}")
        print(f"Transfer Score (%)  : {report.transfer_score.format(1)}")
        print(f"HAT Score (%)       : {report.hat_score.format(1)}")
        print(f"Profile Fit         : R^2 = {report.profile.r2:.4f}, MSE = {report.profile.mse:.5f}")
        if args.all_estimators:
            alt = out["alternative_estimators"]
            print("-" * 60)
            print(f"HAT (Linear + Int.) : {alt['linear_with_intercept']:.1f}")
            print(f"HAT (KRR RBF)       : {alt['krr_rbf']:.1f}")
            print(f"HAT (Capped T<=S)   : {alt['target_capped']:.1f}")

    if args.plot:
        from hat_xlt.plotting import plot_transfer_profile

        plot_transfer_profile(
            s, t, model_name=args.model_name or Path(args.input).stem, save_path=args.plot
        )
        if not args.json:
            print(f"Saved transfer profile plot to: {args.plot}")

    return 0


def _cmd_paper_summary(args: argparse.Namespace) -> int:
    benchmarks = [args.benchmark] if args.benchmark != "all" else ["eclektic", "mmlu_prox_lite", "mgsmv2"]
    names = {m["id"]: m["name"] for m in MODEL_METADATA}

    for bench in benchmarks:
        pairs = get_source_target_pairs(bench, scale=0.01)
        hats = get_model_hats(bench, scale=1.0)
        dual = two_stage_model_clustering(
            pairs, MODEL_CUTOFFS, cutoff_date="2025-01", model_hat_scores=hats
        )

        print(f"\n=== Benchmark: {bench.upper()} ===")
        print(f"Single-line Accuracy-on-the-Line R^2 : {dual.single_line_r2:.3f}")
        print(
            f"Algorithm 1 Fit 1 (Red, n={len(dual.old_cluster.models):2d})  : "
            f"R^2 = {dual.old_cluster.r2:.3f}, Median HAT = {dual.old_cluster.median_hat:.1f}"
        )
        print(
            f"Algorithm 1 Fit 2 (Blue, n={len(dual.new_cluster.models):2d}) : "
            f"R^2 = {dual.new_cluster.r2:.3f}, Median HAT = {dual.new_cluster.median_hat:.1f}"
        )
        print("-" * 74)
        print(f"{'Model':<32} {'Src (%)':>8} {'Tgt (%)':>8} {'Gap (%)':>8} {'HAT':>16}")
        print("-" * 74)
        table = BENCHMARK_TABLES[bench]
        for m in MODEL_METADATA:
            mid = m["id"]
            row = table[mid]
            src_v = row["source"][0]
            tgt_v = row["target"][0]
            gap_v = row["xlt_gap"][0]
            hat_v, hat_lo, hat_hi = row["hat"]
            print(
                f"{names[mid]:<32} {src_v:>8.1f} {tgt_v:>8.1f} {gap_v:>8.1f} "
                f"{hat_v:>5.1f} [{hat_lo:.1f}, {hat_hi:.1f}]"
            )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hat-eval",
        description="Hardness Adjusted Transfer (HAT) Score CLI (EMNLP 2026)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_score = sub.add_parser("score", help="Compute HAT Score and XLT metrics from a JSONL file")
    p_score.add_argument("input", help="Path to evaluation instances JSONL file")
    p_score.add_argument("--langs", default=None, help="Comma-separated target language codes")
    p_score.add_argument(
        "--script-filter",
        choices=["same", "different"],
        default=None,
        help="Restrict target languages to same-script or different-script relative to source_lang",
    )
    p_score.add_argument("--bootstrap", type=int, default=1000, help="Bootstrap resamples (default: 1000)")
    p_score.add_argument("--ci", type=float, default=0.95, help="Confidence level (default: 0.95)")
    p_score.add_argument("--seed", type=int, default=42, help="Bootstrap RNG seed")
    p_score.add_argument("--all-estimators", action="store_true", help="Also compute Linear+Intercept, KRR, and Capped HAT")
    p_score.add_argument("--plot", default=None, help="Optional output filepath (.png/.pdf) for Transfer Profile plot")
    p_score.add_argument("--model-name", default=None, help="Model display name for plot title")
    p_score.add_argument("--json", action="store_true", help="Emit JSON output")
    p_score.set_defaults(func=_cmd_score)

    p_paper = sub.add_parser("paper", help="Display EMNLP 2026 benchmark results and Algorithm 1 clustering")
    p_paper.add_argument(
        "--benchmark",
        choices=["all", "eclektic", "mmlu_prox_lite", "mgsmv2"],
        default="all",
        help="Benchmark to display (default: all)",
    )
    p_paper.set_defaults(func=_cmd_paper_summary)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
