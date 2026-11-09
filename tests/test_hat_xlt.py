"""Unit tests for the `hat_xlt` reference implementation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from hat_xlt import (
    bootstrap_ci,
    capped_hat_score,
    compute_pass_rates,
    evaluate_instances,
    extract_mmlu_choice,
    extract_year_deterministic,
    format_eclektic_judge_prompt,
    format_eclektic_prompt,
    format_mgsmv2_judge_prompt,
    format_mmlu_prox_lite_prompt,
    hat_score,
    hat_score_krr,
    hat_score_linear_intercept,
    joint_cross_task_clustering,
    load_instances_jsonl,
    majority_vote_judges,
    parse_eclektic_judge_grade,
    parse_mgsmv2_judge_grade,
    partition_by_script_congruence,
    source_accuracy,
    spearman_rank_correlation,
    target_accuracy,
    transfer_profile,
    transfer_score,
    two_stage_model_clustering,
    xlt_gap,
)
from hat_xlt.cli import main as cli_main
from hat_xlt.paper_results import (
    MODEL_CUTOFFS,
    get_model_hats,
    get_source_target_pairs,
)


def test_perfect_parity_and_zero_transfer() -> None:
    s = np.linspace(0.1, 1.0, 50)
    assert hat_score(s, s) == pytest.approx(1.0, abs=1e-9)
    assert hat_score(s, np.zeros_like(s)) == pytest.approx(0.0, abs=1e-9)
    assert xlt_gap(s, s) == pytest.approx(0.0, abs=1e-9)


def test_invariance_to_source_competence_shift() -> None:
    """Core theoretical property (Section 4): shifting p_S(s) alters Target Acc, XLT Gap,
    and Transfer Score, while HAT Score remains invariant for fixed E[T | S = s] = m * s."""
    rng = np.random.default_rng(123)
    true_m = 0.72

    s_weak = rng.beta(0.7, 2.5, size=500)
    t_weak = true_m * s_weak

    s_strong = rng.beta(2.5, 0.7, size=500)
    t_strong = true_m * s_strong

    # Target accuracy and XLT gap shift substantially
    assert target_accuracy(t_strong) - target_accuracy(t_weak) > 0.25
    assert xlt_gap(s_strong, t_strong) - xlt_gap(s_weak, t_weak) > 0.10

    # HAT Score stays exact
    assert hat_score(s_weak, t_weak) == pytest.approx(true_m, abs=1e-9)
    assert hat_score(s_strong, t_strong) == pytest.approx(true_m, abs=1e-9)


def test_multilingual_2d_pass_rates_and_capping() -> None:
    rng = np.random.default_rng(7)
    binary = rng.integers(0, 2, size=(40, 5, 10))
    tgt_rates = compute_pass_rates(binary)
    src_rates = compute_pass_rates(rng.integers(0, 2, size=(40, 10)))

    assert tgt_rates.shape == (40, 5)
    h_base = hat_score(src_rates, tgt_rates)
    h_cap = capped_hat_score(src_rates, tgt_rates)
    assert 0.0 <= h_cap <= 1.0
    assert h_cap <= h_base + 1e-9


def test_alternative_estimators_preserve_ranking() -> None:
    rng = np.random.default_rng(99)
    slopes = [0.45, 0.55, 0.65, 0.75, 0.85, 0.95]
    base_scores, lin_int_scores, krr_scores = [], [], []
    for m in slopes:
        s = rng.uniform(0.05, 1.0, size=120)
        t = np.clip(m * s + rng.normal(0, 0.03, size=120), 0.0, 1.0)
        base_scores.append(hat_score(s, t))
        lin_int_scores.append(hat_score_linear_intercept(s, t))
        krr_scores.append(hat_score_krr(s, t))

    assert spearman_rank_correlation(base_scores, lin_int_scores) > 0.99
    assert spearman_rank_correlation(base_scores, krr_scores) > 0.99

    prof_krr = transfer_profile(s, t, estimator="krr")
    assert prof_krr.grid_s.shape == (201,)
    assert 0.0 <= prof_krr.hat_score <= 1.05


def test_bootstrap_ci_and_evaluate_instances() -> None:
    rng = np.random.default_rng(42)
    s = rng.uniform(0.1, 1.0, size=200)
    t = np.clip(0.8 * s + rng.normal(0, 0.04, size=200), 0.0, 1.0)

    ci = bootstrap_ci(s, t, n_bootstrap=300, seed=42)
    assert ci.ci_low < ci.estimate < ci.ci_high
    assert ci.estimate == pytest.approx(0.8, abs=0.03)

    rep = evaluate_instances(s, t, n_bootstrap=200, seed=42)
    d = rep.to_dict(scale=100.0)
    assert d["n_items"] == 200
    assert 75.0 < d["hat_score"]["estimate"] < 85.0


def test_paper_accuracy_on_the_line_and_mmlu_clustering() -> None:
    """Verify Figure 1 R^2 > 0.94 and Table 7 exact cluster recovery on MMLU-ProX-Lite."""
    for bench, expected_r2 in [
        ("eclektic", 0.984),
        ("mgsmv2", 0.943),
        ("mmlu_prox_lite", 0.974),
    ]:
        pairs = get_source_target_pairs(bench, scale=0.01)
        hats = get_model_hats(bench, scale=1.0)
        res = two_stage_model_clustering(
            pairs, MODEL_CUTOFFS, cutoff_date="2025-01", model_hat_scores=hats
        )
        assert res.single_line_r2 == pytest.approx(expected_r2, abs=0.004)

    # Verify MGSMv2 clean temporal bifurcation (all 12 post-2025 models form Fit 2, R^2 = 0.997)
    mgsm_pairs = get_source_target_pairs("mgsmv2", scale=0.01)
    mgsm_hats = get_model_hats("mgsmv2", scale=1.0)
    mgsm_res = two_stage_model_clustering(
        mgsm_pairs, MODEL_CUTOFFS, cutoff_date="2025-01", model_hat_scores=mgsm_hats
    )
    assert len(mgsm_res.old_cluster.models) == 8
    assert len(mgsm_res.new_cluster.models) == 12
    assert mgsm_res.new_cluster.r2 == pytest.approx(0.997, abs=0.002)
    assert mgsm_res.new_cluster.median_hat > mgsm_res.old_cluster.median_hat + 5.0

    # Verify MMLU-ProX-Lite dual-fit reduces total squared error over single-line fit
    mmlu_pairs = get_source_target_pairs("mmlu_prox_lite", scale=0.01)
    mmlu_hats = get_model_hats("mmlu_prox_lite", scale=1.0)
    mmlu_res = two_stage_model_clustering(
        mmlu_pairs, MODEL_CUTOFFS, cutoff_date="2025-01", model_hat_scores=mmlu_hats
    )
    assert mmlu_res.wmse_total < len(mmlu_pairs) * mmlu_res.single_line_mse
    assert mmlu_res.new_cluster.median_hat > mmlu_res.old_cluster.median_hat + 15.0

    # Joint optimization across all 3 benchmarks retains strict temporal cohorts (Appendix A.8)
    all_task_scores = {
        b: get_source_target_pairs(b, scale=0.01)
        for b in ["eclektic", "mmlu_prox_lite", "mgsmv2"]
    }
    j_old, j_new = joint_cross_task_clustering(all_task_scores, MODEL_CUTOFFS, cutoff_date="2025-01")
    assert len(j_old) == 8
    assert len(j_new) == 12


def test_prompts_extractors_and_judges() -> None:
    p_ec = format_eclektic_prompt("In welchem Land befand sich AFN Bremerhaven?")
    assert "Provide only the answer." in p_ec

    p_mc = format_mmlu_prox_lite_prompt("Pick one:", ["alpha", "beta", "gamma"])
    assert "A. alpha" in p_mc and "C. gamma" in p_mc

    j_ec = format_eclektic_judge_prompt("Who?", "Deutschland", "Deutschland")
    assert "Gold target: Deutschland" in j_ec
    assert parse_eclektic_judge_grade("A") == 1
    assert parse_eclektic_judge_grade("B") == 0
    assert parse_eclektic_judge_grade("C") == 0

    j_mg = format_mgsmv2_judge_prompt("2+2?", "4", "Therefore the answer is 4.")
    assert "<correct_answer>\n4\n</correct_answer>" in j_mg
    assert parse_mgsmv2_judge_grade("<extracted_final_answer>4</extracted_final_answer><correct>yes</correct>") == 1
    assert parse_mgsmv2_judge_grade("<correct>no</correct>") == 0

    assert extract_mmlu_choice("E") == "E"
    assert extract_mmlu_choice("Answer: (C)") == "C"
    assert extract_year_deterministic("Established in 1992 in Pune.", "1992") == 1
    assert extract_year_deterministic("Established in 1994.", "1992") == 0
    assert majority_vote_judges([1, 0, 1]) == 1
    assert majority_vote_judges([0, 1, 0]) == 0


def test_jsonl_loader_script_filtering_and_cli(tmp_path: Path) -> None:
    same_l, diff_l = partition_by_script_congruence("en", ["de", "fr", "hi", "ru", "zh"])
    assert same_l == ["de", "fr"]
    assert diff_l == ["hi", "ru", "zh"]

    jsonl_file = tmp_path / "sample_eval.jsonl"
    records = [
        {
            "id": f"q{i}",
            "source_lang": "en",
            "source_pass_rate": round(0.1 * (i % 10 + 1), 2),
            "target_pass_rates": {
                "de": round(0.85 * 0.1 * (i % 10 + 1), 3),
                "hi": round(0.60 * 0.1 * (i % 10 + 1), 3),
            },
        }
        for i in range(30)
    ]
    jsonl_file.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")

    s_same, t_same = load_instances_jsonl(jsonl_file, script_filter="same")
    s_diff, t_diff = load_instances_jsonl(jsonl_file, script_filter="different")
    assert hat_score(s_same, t_same) == pytest.approx(0.85, abs=1e-2)
    assert hat_score(s_diff, t_diff) == pytest.approx(0.60, abs=1e-2)

    rc = cli_main(["score", str(jsonl_file), "--bootstrap", "100", "--all-estimators", "--json"])
    assert rc == 0
