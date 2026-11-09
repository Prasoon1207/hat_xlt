# HAT: Hardness Adjusted Transfer Score

[![EMNLP 2026](https://img.shields.io/badge/EMNLP-2026-1d4ed8)](https://arxiv.org/abs/2606.21954)
[![arXiv](https://img.shields.io/badge/arXiv-2606.21954-b31b1b)](https://arxiv.org/abs/2606.21954)
[![Project Page](https://img.shields.io/badge/Project_Page-prasoon1207.github.io%2Fhat__xlt-0f766e)](https://prasoon1207.github.io/hat_xlt)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-slate)](LICENSE)

Official code and evaluation suite for **"Are Multilingual Models Improving? Revisiting Cross-Lingual Transfer"** (*EMNLP 2026*).

**Authors:** Prasoon Bajpai¹, Eleftheria Briakou¹, Colin Cherry¹, Preethi Jyothi¹'², Vihari Piratla¹  
*¹Google DeepMind · ²Indian Institute of Technology Bombay*

- **Project Page & Interactive Leaderboard:** [https://prasoon1207.github.io/hat_xlt](https://prasoon1207.github.io/hat_xlt)
- **Paper (arXiv):** [https://arxiv.org/abs/2606.21954](https://arxiv.org/abs/2606.21954)

---

## Why HAT Score?

Standard cross-lingual transfer (XLT) metrics—**Average Target Accuracy**, **XLT Gap** ($\mathbb{E}[S - T_\ell]$), and **Transfer Score** ($\mathbb{E}[T_\ell \mid S = 1]$)—weight conditional transfer performance by the model's source-language pass-rate density $p_S(s \mid M)$:

$$\text{Avg. Target Accuracy} = \int_0^1 \mathbb{E}_{D,\ell}[T_\ell \mid S = s, M]\, p_S(s \mid M)\, ds$$

Across 20 frontier and open-weight LLMs on **ECLeKTic**, **MGSMv2**, and **MMLU-ProX-Lite**, target accuracy scales linearly with source accuracy ($R^2 > 0.94$). Consequently:
1. **Target Accuracy & Transfer Score inflate** when general source-language capability improves—even when the underlying transfer profile $\mathbb{E}[T_\ell \mid S = s]$ is unchanged (e.g., `GPT-OSS-20B` $\to$ `GPT-OSS-120B` on ECLeKTic).
2. **XLT Gap widens** as source accuracy increases along a slope $< 1$, falsely signaling regression even when true transfer improves markedly (e.g., `Gemma-3-1B` $\to$ `Gemma-3-27B` on MMLU-ProX-Lite).

### The Hardness Adjusted Transfer (HAT) Score

**HAT Score** replaces the model-dependent source density $p_S(s \mid M)$ with a uniform prior over source difficulty $s \in [0, 1]$:

$$\text{HAT Score} = 2 \int_0^1 \mathbb{E}_{D,\ell}[T_\ell \mid S = s, M]\, ds$$

Given $N = 10$ stochastic passes per prompt in the source language ($s^{(i)} \in [0, 1]$) and target languages ($t^{(i)} = \mathbb{E}_\ell[t_\ell^{(i)}] \in [0, 1]$), we fit a low-variance zero-intercept linear transfer profile ($\mathbb{E}[T_\ell \mid S = 0] = 0$):

$$\hat{m} = \arg\min_{m} \sum_{i} \left(t^{(i)} - m\, s^{(i)}\right)^2 = \frac{\sum_i s^{(i)} t^{(i)}}{\sum_i (s^{(i)})^2}, \qquad \text{HAT Score} = 2 \int_0^1 \hat{m} s\, ds = \hat{m}$$

---

## Installation

```bash
git clone https://github.com/Prasoon1207/hat_xlt.git
cd hat_xlt

python3 -m venv .venv
.venv/bin/pip install -e ".[estimators,dev]"
```

---

## Quickstart

### 1. Python API

```python
import numpy as np
from hat_xlt import evaluate_instances, hat_score
from hat_xlt.plotting import plot_transfer_profile

# s: shape (n_items,) source pass-rates across N=10 samples
# t: shape (n_items, n_target_langs) or (n_items,) target pass-rates
rng = np.random.default_rng(0)
s = rng.uniform(0.0, 1.0, size=500)
t = np.clip(0.82 * s[:, None] + rng.normal(0, 0.05, size=(500, 11)), 0.0, 1.0)

# Direct HAT Score
print(f"HAT Score: {hat_score(s, t) * 100:.1f}")

# Complete evaluation report with 1,000 paired bootstrap resamples (95% CIs)
report = evaluate_instances(s, t, n_bootstrap=1000, seed=42)
print("Target Accuracy:", report.target_accuracy.format(1))
print("XLT Gap        :", report.xlt_gap.format(1))
print("Transfer Score :", report.transfer_score.format(1))
print("HAT Score      :", report.hat_score.format(1))

# Save Transfer Profile plot
plot_transfer_profile(s, t, model_name="My-LLM", dataset_name="ECLeKTic", save_path="profile.png")
```

### 2. Command-Line Interface (`hat-eval`)

Evaluate multi-pass model outputs from a `.jsonl` file (supports per-item `source_pass_rate` / `target_pass_rates` or raw binary verdict arrays `source_correct` / `target_correct`, plus `--script-filter {same,different}` for script congruence analysis):

```bash
# Score an evaluation log and save the HAT Transfer Profile figure
hat-eval score path/to/eval_outputs.jsonl --all-estimators --plot transfer_profile.png

# Restrict target languages to different-script targets relative to each item's source language
hat-eval score path/to/eval_outputs.jsonl --script-filter different

# Inspect all 20 models' EMNLP 2026 benchmark tables & Algorithm 1 temporal clusters
hat-eval paper --benchmark all
```

### 3. Reproducing Paper Analyses & Demonstrations

```bash
# Simulated walkthrough of Target Accuracy inflation & XLT Gap distortion vs. HAT invariance
python examples/quickstart.py

# Reproduce Accuracy-on-the-Line R^2 (>0.94), Algorithm 1 Dual-Fit clusters (Table 7), & scaling tables
python examples/reproduce_paper_analyses.py --save-plots-dir outputs/
```

---

## Repository Structure

```text
hat_xlt/
├── src/hat_xlt/
│   ├── metrics.py        # HAT Score (OLS through origin), Linear+Intercept, RBF KRR,
│   │                     # Target Capping (T'=min(T,S)), Target Acc, XLT Gap, Transfer Score,
│   │                     # and paired Efron (1979) bootstrap 95% CIs
│   ├── clustering.py     # Algorithm 1: Two-Stage Method for Model Clustering, strict
│   │                     # temporal split, joint cross-task & held-out validation splits
│   ├── prompts.py        # Zero-shot task prompts (Table 4), ECLeKTic & MGSMv2 LLM-as-a-Judge
│   │                     # prompts (Figures 6 & 7), MMLU/Year-ECLeKTic extractors, majority vote
│   ├── datasets.py       # Language lists (Table 2), script congruence mappings (Section 6.5),
│   │                     # and multi-pass JSONL evaluation loader
│   ├── paper_results.py  # Complete benchmark results & knowledge cutoffs for all 20 LLMs
│   ├── plotting.py       # Transfer Profile, Accuracy-on-the-Line, & Dual-Fit matplotlib plots
│   └── cli.py            # `hat-eval` command-line entrypoint
├── examples/
│   ├── quickstart.py
│   └── reproduce_paper_analyses.py
└── tests/
    └── test_hat_xlt.py   # Pytest suite verifying theoretical invariants & paper tables
```

---

## Citation

If you find HAT Score or our analysis useful in your research, please cite:

```bibtex
@inproceedings{bajpai2026multilingual,
  title={Are Multilingual Models Improving? Revisiting Cross-Lingual Transfer},
  author={Bajpai, Prasoon and Briakou, Eleftheria and Cherry, Colin and Jyothi, Preethi and Piratla, Vihari},
  booktitle={Proceedings of the 2026 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
  year={2026},
  url={https://arxiv.org/abs/2606.21954}
}
```
