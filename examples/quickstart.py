#!/usr/bin/env python3
"""Minimal end-to-end demonstration of the Hardness Adjusted Transfer (HAT) Score.

Demonstrates the two core confounding failure modes of standard XLT metrics
highlighted in Section 4 & Section 6.1 of the EMNLP 2026 paper:

  Case A (Source Competence Inflation — e.g. GPT-OSS-20B vs. GPT-OSS-120B on ECLeKTic):
      Two models share the EXACT same cross-lingual transfer profile E[T | S = s] = 0.80 * s,
      but Model B has higher source competence. Target Accuracy and Transfer Score
      falsely report large transfer gains, whereas HAT Score remains invariant (~80%).

  Case B (XLT Gap Distortion — e.g. Gemma-3-1B vs. Gemma-3-27B on MMLU-ProX-Lite):
      Model D genuinely improves transfer slope (0.58 -> 0.77) AND source competence.
      XLT Gap widens (falsely implying regression!), whereas HAT Score captures the +19pt gain.
"""

import numpy as np

from hat_xlt import evaluate_instances


def simulate_model(
    rng: np.random.Generator,
    alpha_src: float,
    beta_src: float,
    true_transfer_slope: float,
    n_items: int = 600,
    n_passes: int = 10,
    n_target_langs: int = 11,
):
    p_s = rng.beta(alpha_src, beta_src, size=n_items)
    p_t = np.clip(true_transfer_slope * p_s, 0.0, 1.0)

    s_obs = rng.binomial(n_passes, p_s) / n_passes
    t_obs = rng.binomial(n_passes, p_t[:, None], size=(n_items, n_target_langs)) / n_passes
    return s_obs, t_obs


def main() -> None:
    rng = np.random.default_rng(2026)

    print("=" * 72)
    print("CASE A: Identical True Transfer (m = 0.80), Higher Source Competence")
    print("=" * 72)
    s_a, t_a = simulate_model(rng, alpha_src=0.6, beta_src=2.4, true_transfer_slope=0.80)
    s_b, t_b = simulate_model(rng, alpha_src=1.6, beta_src=1.4, true_transfer_slope=0.80)

    rep_a = evaluate_instances(s_a, t_a, seed=1)
    rep_b = evaluate_instances(s_b, t_b, seed=1)

    for label, rep in [("Model A (Lower Source)", rep_a), ("Model B (Higher Source)", rep_b)]:
        print(
            f"{label:<24} | Src: {rep.source_accuracy.estimate*100:5.1f}% "
            f"| Tgt: {rep.target_accuracy.estimate*100:5.1f}% "
            f"| Transfer Score: {rep.transfer_score.estimate*100:5.1f}% "
            f"| HAT: {rep.hat_score.format(1)}"
        )

    print("\n" + "=" * 72)
    print("CASE B: Improved True Transfer (m: 0.58 -> 0.77) + Higher Source Acc.")
    print("=" * 72)
    s_c, t_c = simulate_model(rng, alpha_src=0.5, beta_src=2.2, true_transfer_slope=0.58)
    s_d, t_d = simulate_model(rng, alpha_src=1.8, beta_src=1.5, true_transfer_slope=0.77)

    rep_c = evaluate_instances(s_c, t_c, seed=2)
    rep_d = evaluate_instances(s_d, t_d, seed=2)

    for label, rep in [("Model C (1B-like)", rep_c), ("Model D (27B-like)", rep_d)]:
        print(
            f"{label:<24} | Src: {rep.source_accuracy.estimate*100:5.1f}% "
            f"| Tgt: {rep.target_accuracy.estimate*100:5.1f}% "
            f"| XLT Gap: {rep.xlt_gap.estimate*100:5.1f}% "
            f"| HAT: {rep.hat_score.format(1)}"
        )


if __name__ == "__main__":
    main()
