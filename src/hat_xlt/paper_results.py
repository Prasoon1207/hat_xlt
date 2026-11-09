"""Complete empirical benchmark tables from the EMNLP 2026 HAT paper.

Includes all 20 evaluated LLMs across ECLeKTic, MMLU-ProX-Lite, and MGSMv2
(Tables 1, 6, 7, 17, 18, 19, 23, and 24, plus source-language accuracies).
"""

from __future__ import annotations

from typing import Dict, List, Tuple


MODEL_METADATA: List[Dict[str, str]] = [
    {"id": "gemini-2.5-flash", "name": "Gemini-2.5-Flash", "family": "Gemini", "cutoff": "2025-01"},
    {"id": "gemini-2.5-flash-thinking-off", "name": "Gemini-2.5-Flash (Thinking-Off)", "family": "Gemini", "cutoff": "2025-01"},
    {"id": "gemini-3-flash", "name": "Gemini-3-Flash", "family": "Gemini", "cutoff": "2025-01"},
    {"id": "gemini-3-flash-low", "name": "Gemini-3-Flash (Low)", "family": "Gemini", "cutoff": "2025-01"},
    {"id": "gemini-3-flash-minimal", "name": "Gemini-3-Flash (Minimal)", "family": "Gemini", "cutoff": "2025-01"},
    {"id": "gemma-3-1b", "name": "Gemma-3-1B", "family": "Gemma 3", "cutoff": "2024-08"},
    {"id": "gemma-3-4b", "name": "Gemma-3-4B", "family": "Gemma 3", "cutoff": "2024-08"},
    {"id": "gemma-3-12b", "name": "Gemma-3-12B", "family": "Gemma 3", "cutoff": "2024-08"},
    {"id": "gemma-3-27b", "name": "Gemma-3-27B", "family": "Gemma 3", "cutoff": "2024-08"},
    {"id": "qwen-3-4b", "name": "Qwen-3-4B", "family": "Qwen 3", "cutoff": "2024-11"},
    {"id": "qwen-3-30b-a3b", "name": "Qwen-3-30B-A3B", "family": "Qwen 3", "cutoff": "2024-11"},
    {"id": "gpt-oss-20b", "name": "GPT-OSS-20B", "family": "GPT-OSS", "cutoff": "2024-06"},
    {"id": "gpt-oss-120b", "name": "GPT-OSS-120B", "family": "GPT-OSS", "cutoff": "2024-06"},
    {"id": "gemma-4-e2b-it", "name": "Gemma-4-E2B-IT", "family": "Gemma 4", "cutoff": "2025-01"},
    {"id": "gemma-4-e4b-it", "name": "Gemma-4-E4B-IT", "family": "Gemma 4", "cutoff": "2025-01"},
    {"id": "gemma-4-26b-a4b-it", "name": "Gemma-4-26B-A4B-IT", "family": "Gemma 4", "cutoff": "2025-01"},
    {"id": "gemma-4-31b-it", "name": "Gemma-4-31B-IT", "family": "Gemma 4", "cutoff": "2025-01"},
    {"id": "claude-haiku-4.5", "name": "Claude-Haiku-4.5", "family": "Claude", "cutoff": "2025-02"},
    {"id": "claude-sonnet-4.6", "name": "Claude-Sonnet-4.6", "family": "Claude", "cutoff": "2025-05"},
    {"id": "claude-opus-4.7", "name": "Claude-Opus-4.7", "family": "Claude", "cutoff": "2026-01"},
]

MODEL_CUTOFFS: Dict[str, str] = {m["id"]: m["cutoff"] for m in MODEL_METADATA}

# Table 17 + Table 23 + Source Accuracy (all values in percentage points, 0-100)
ECLEKTIC_RESULTS: Dict[str, Dict[str, object]] = {
    "gemini-2.5-flash": {"source": (50.9, 46.8, 55.0), "target": (44.5, 41.1, 48.1), "xlt_gap": (6.4, 4.3, 8.5), "transfer_score": (66.0, 63.6, 68.4), "hat": (81.7, 78.6, 84.7)},
    "gemini-2.5-flash-thinking-off": {"source": (42.8, 38.8, 47.1), "target": (29.4, 26.5, 32.5), "xlt_gap": (13.3, 10.5, 16.3), "transfer_score": (44.9, 42.5, 47.4), "hat": (60.4, 55.9, 64.8)},
    "gemini-3-flash": {"source": (68.4, 64.1, 72.5), "target": (60.5, 56.8, 64.3), "xlt_gap": (8.0, 5.6, 10.4), "transfer_score": (78.0, 76.5, 79.6), "hat": (84.6, 82.1, 87.1)},
    "gemini-3-flash-low": {"source": (69.1, 65.0, 73.2), "target": (60.2, 56.5, 64.0), "xlt_gap": (8.9, 6.5, 11.2), "transfer_score": (76.2, 74.5, 77.9), "hat": (84.0, 81.6, 86.7)},
    "gemini-3-flash-minimal": {"source": (65.7, 61.4, 69.8), "target": (52.3, 48.9, 56.1), "xlt_gap": (13.4, 10.2, 16.3), "transfer_score": (68.0, 66.1, 69.7), "hat": (74.2, 70.9, 77.5)},
    "gemma-3-1b": {"source": (5.4, 3.7, 7.7), "target": (4.6, 3.5, 6.0), "xlt_gap": (0.8, -0.3, 2.2), "transfer_score": (32.1, 26.3, 38.0), "hat": (48.2, 34.9, 61.0)},
    "gemma-3-4b": {"source": (11.5, 8.9, 14.6), "target": (9.6, 7.7, 11.8), "xlt_gap": (1.9, 0.3, 4.0), "transfer_score": (45.8, 41.4, 50.6), "hat": (56.4, 47.0, 65.9)},
    "gemma-3-12b": {"source": (17.3, 14.0, 20.7), "target": (15.2, 12.8, 17.9), "xlt_gap": (2.1, -0.2, 4.3), "transfer_score": (56.2, 51.8, 60.5), "hat": (64.1, 56.8, 71.4)},
    "gemma-3-27b": {"source": (26.4, 22.2, 30.4), "target": (19.8, 17.2, 23.0), "xlt_gap": (6.5, 4.1, 8.9), "transfer_score": (58.8, 55.1, 62.4), "hat": (63.8, 58.4, 69.6)},
    "qwen-3-4b": {"source": (13.8, 10.8, 16.9), "target": (10.8, 8.9, 13.1), "xlt_gap": (2.9, 1.1, 5.1), "transfer_score": (38.8, 34.3, 43.1), "hat": (56.5, 47.8, 64.9)},
    "qwen-3-30b-a3b": {"source": (19.4, 16.2, 23.0), "target": (16.6, 14.1, 19.5), "xlt_gap": (2.8, 0.7, 5.1), "transfer_score": (48.9, 45.3, 52.8), "hat": (65.9, 59.7, 72.6)},
    "gpt-oss-20b": {"source": (19.7, 16.5, 23.1), "target": (18.0, 15.3, 21.2), "xlt_gap": (1.6, 0.2, 3.2), "transfer_score": (53.5, 49.4, 57.5), "hat": (80.1, 74.7, 85.3)},
    "gpt-oss-120b": {"source": (27.9, 24.3, 31.7), "target": (25.4, 22.4, 28.7), "xlt_gap": (2.4, 0.6, 4.1), "transfer_score": (61.0, 58.0, 63.9), "hat": (82.2, 77.9, 86.1)},
    "gemma-4-e2b-it": {"source": (11.9, 9.1, 14.8), "target": (9.0, 7.2, 11.0), "xlt_gap": (2.8, 0.9, 4.9), "transfer_score": (39.0, 35.0, 43.2), "hat": (51.8, 42.4, 61.3)},
    "gemma-4-e4b-it": {"source": (15.7, 12.7, 19.0), "target": (10.7, 8.7, 13.0), "xlt_gap": (5.0, 3.1, 6.8), "transfer_score": (43.9, 40.0, 48.1), "hat": (58.4, 51.2, 65.7)},
    "gemma-4-26b-a4b-it": {"source": (26.0, 21.8, 29.8), "target": (20.3, 17.4, 23.3), "xlt_gap": (5.7, 3.0, 8.3), "transfer_score": (55.2, 51.8, 58.8), "hat": (63.9, 57.9, 69.6)},
    "gemma-4-31b-it": {"source": (28.2, 23.8, 32.5), "target": (22.6, 19.5, 26.0), "xlt_gap": (5.5, 3.1, 8.0), "transfer_score": (55.3, 51.9, 58.6), "hat": (68.6, 62.9, 74.1)},
    "claude-haiku-4.5": {"source": (29.3, 25.5, 33.7), "target": (20.8, 17.8, 23.9), "xlt_gap": (8.4, 5.9, 11.3), "transfer_score": (51.1, 48.2, 54.3), "hat": (59.8, 53.8, 65.7)},
    "claude-sonnet-4.6": {"source": (52.9, 48.0, 57.8), "target": (44.1, 40.2, 48.0), "xlt_gap": (8.7, 5.6, 11.5), "transfer_score": (70.0, 67.9, 72.2), "hat": (74.7, 70.9, 78.5)},
    "claude-opus-4.7": {"source": (55.6, 50.6, 60.0), "target": (45.2, 41.6, 49.1), "xlt_gap": (10.3, 7.4, 13.5), "transfer_score": (67.0, 64.9, 69.1), "hat": (73.5, 69.7, 77.2)},
}

# Table 18 + Source Accuracy
MMLU_PROX_LITE_RESULTS: Dict[str, Dict[str, object]] = {
    "gemini-2.5-flash": {"source": (72.0, 69.0, 75.2), "target": (73.4, 73.3, 75.5), "xlt_gap": (-1.9, -3.4, -0.5), "hat": (97.5, 96.2, 98.8)},
    "gemini-2.5-flash-thinking-off": {"source": (67.5, 64.2, 70.8), "target": (61.6, 60.1, 62.2), "xlt_gap": (5.9, 4.5, 7.4), "hat": (88.9, 87.1, 90.3)},
    "gemini-3-flash": {"source": (80.8, 78.3, 83.3), "target": (82.1, 81.6, 82.6), "xlt_gap": (-1.4, -2.6, -0.3), "hat": (98.8, 97.8, 99.9)},
    "gemini-3-flash-low": {"source": (82.2, 79.6, 84.9), "target": (83.0, 82.5, 83.5), "xlt_gap": (-0.8, -2.0, 0.4), "hat": (98.3, 97.3, 99.3)},
    "gemini-3-flash-minimal": {"source": (80.4, 77.4, 83.3), "target": (77.2, 76.6, 77.8), "xlt_gap": (3.2, 1.9, 4.6), "hat": (93.5, 92.1, 94.8)},
    "gemma-3-1b": {"source": (17.0, 14.1, 19.9), "target": (14.2, 13.7, 14.7), "xlt_gap": (2.7, 0.8, 4.7), "hat": (58.4, 52.3, 64.1)},
    "gemma-3-4b": {"source": (28.2, 24.7, 31.7), "target": (24.6, 23.9, 25.2), "xlt_gap": (3.6, 1.5, 5.6), "hat": (67.5, 63.1, 71.7)},
    "gemma-3-12b": {"source": (42.0, 38.1, 45.8), "target": (34.9, 34.3, 35.6), "xlt_gap": (7.0, 4.7, 9.2), "hat": (71.6, 68.2, 74.7)},
    "gemma-3-27b": {"source": (45.7, 41.8, 49.7), "target": (39.4, 38.6, 40.1), "xlt_gap": (6.4, 4.3, 8.2), "hat": (77.2, 74.4, 80.1)},
    "qwen-3-4b": {"source": (40.8, 37.0, 44.5), "target": (32.7, 32.0, 33.4), "xlt_gap": (8.1, 6.0, 10.3), "hat": (69.3, 65.8, 72.6)},
    "qwen-3-30b-a3b": {"source": (53.9, 50.2, 57.6), "target": (43.4, 42.7, 44.2), "xlt_gap": (10.5, 8.5, 12.4), "hat": (74.8, 71.9, 77.6)},
    "gpt-oss-20b": {"source": (74.0, 71.0, 77.2), "target": (66.3, 65.7, 66.9), "xlt_gap": (7.7, 6.5, 9.0), "hat": (88.9, 87.6, 90.3)},
    "gpt-oss-120b": {"source": (79.0, 76.0, 81.8), "target": (72.3, 71.8, 73.0), "xlt_gap": (6.6, 5.4, 8.0), "hat": (90.6, 89.2, 92.0)},
    "gemma-4-e2b-it": {"source": (31.8, 28.5, 35.5), "target": (26.1, 25.5, 26.7), "xlt_gap": (5.7, 3.2, 8.6), "hat": (59.3, 54.7, 63.5)},
    "gemma-4-e4b-it": {"source": (42.9, 39.0, 46.6), "target": (35.8, 35.2, 36.5), "xlt_gap": (7.1, 5.1, 9.0), "hat": (74.8, 71.6, 77.8)},
    "gemma-4-26b-a4b-it": {"source": (53.6, 49.9, 57.5), "target": (48.7, 48.0, 49.5), "xlt_gap": (4.8, 2.6, 7.2), "hat": (78.7, 76.2, 81.1)},
    "gemma-4-31b-it": {"source": (64.1, 60.4, 67.5), "target": (59.4, 58.6, 60.1), "xlt_gap": (4.8, 2.9, 6.8), "hat": (84.7, 82.5, 86.7)},
    "claude-haiku-4.5": {"source": (45.2, 41.4, 49.1), "target": (44.6, 43.9, 45.3), "xlt_gap": (0.6, -1.9, 3.1), "hat": (76.5, 73.5, 79.2)},
    "claude-sonnet-4.6": {"source": (66.8, 63.0, 70.7), "target": (59.5, 58.7, 60.2), "xlt_gap": (7.4, 5.7, 9.0), "hat": (85.4, 83.6, 87.3)},
    "claude-opus-4.7": {"source": (82.5, 79.6, 85.2), "target": (76.7, 76.2, 77.5), "xlt_gap": (5.8, 3.8, 7.3), "hat": (89.3, 87.8, 90.9)},
}

# Table 19 + Source Accuracy
MGSMV2_RESULTS: Dict[str, Dict[str, object]] = {
    "gemini-2.5-flash": {"source": (99.5, 99.0, 99.9), "target": (98.2, 97.7, 98.6), "xlt_gap": (1.3, 0.8, 1.9), "hat": (98.7, 98.1, 99.2)},
    "gemini-2.5-flash-thinking-off": {"source": (58.4, 52.6, 64.0), "target": (55.6, 53.9, 57.5), "xlt_gap": (2.8, -0.1, 5.4), "hat": (89.8, 86.9, 92.4)},
    "gemini-3-flash": {"source": (99.8, 99.4, 100.0), "target": (99.3, 99.1, 99.6), "xlt_gap": (0.4, 0.1, 0.8), "hat": (99.6, 99.3, 99.9)},
    "gemini-3-flash-low": {"source": (99.9, 99.8, 100.0), "target": (99.4, 99.2, 99.6), "xlt_gap": (0.5, 0.2, 1.0), "hat": (99.5, 99.1, 99.8)},
    "gemini-3-flash-minimal": {"source": (86.9, 83.2, 90.6), "target": (86.5, 85.4, 87.7), "xlt_gap": (0.4, -1.8, 2.5), "hat": (96.2, 94.8, 97.5)},
    "gemma-3-1b": {"source": (50.7, 45.9, 55.6), "target": (25.9, 24.6, 27.4), "xlt_gap": (24.7, 21.3, 28.3), "hat": (47.7, 43.7, 51.8)},
    "gemma-3-4b": {"source": (86.3, 82.6, 89.6), "target": (72.9, 71.5, 74.3), "xlt_gap": (13.4, 10.8, 15.9), "hat": (82.6, 80.3, 85.0)},
    "gemma-3-12b": {"source": (96.0, 93.7, 97.8), "target": (88.1, 87.1, 89.1), "xlt_gap": (7.9, 6.3, 9.7), "hat": (91.3, 89.7, 92.8)},
    "gemma-3-27b": {"source": (97.4, 95.6, 98.9), "target": (93.3, 92.6, 94.1), "xlt_gap": (4.1, 2.7, 5.4), "hat": (95.3, 94.0, 96.5)},
    "qwen-3-4b": {"source": (97.4, 95.9, 98.7), "target": (79.4, 77.9, 80.8), "xlt_gap": (18.1, 16.2, 20.1), "hat": (81.3, 79.4, 83.2)},
    "qwen-3-30b-a3b": {"source": (98.4, 97.0, 99.5), "target": (88.6, 87.6, 89.6), "xlt_gap": (9.8, 8.2, 11.4), "hat": (89.8, 88.4, 91.3)},
    "gpt-oss-20b": {"source": (99.0, 98.3, 99.6), "target": (95.2, 94.6, 95.8), "xlt_gap": (3.8, 3.0, 4.6), "hat": (96.1, 95.4, 96.9)},
    "gpt-oss-120b": {"source": (99.6, 98.7, 100.0), "target": (97.0, 96.4, 97.5), "xlt_gap": (2.6, 2.0, 3.2), "hat": (97.4, 96.8, 98.0)},
    "gemma-4-e2b-it": {"source": (18.0, 13.8, 22.3), "target": (13.0, 11.9, 14.1), "xlt_gap": (5.0, 2.2, 8.0), "hat": (53.7, 45.1, 62.6)},
    "gemma-4-e4b-it": {"source": (21.9, 17.5, 26.6), "target": (17.5, 16.3, 18.8), "xlt_gap": (4.4, 2.3, 6.8), "hat": (68.0, 60.3, 74.1)},
    "gemma-4-26b-a4b-it": {"source": (99.5, 98.6, 100.0), "target": (96.5, 95.9, 97.1), "xlt_gap": (3.0, 1.9, 4.2), "hat": (96.8, 95.7, 97.7)},
    "gemma-4-31b-it": {"source": (99.6, 99.0, 100.0), "target": (98.3, 97.8, 98.6), "xlt_gap": (1.4, 0.6, 2.1), "hat": (98.4, 97.8, 99.0)},
    "claude-haiku-4.5": {"source": (52.8, 46.4, 58.5), "target": (54.0, 52.3, 55.9), "xlt_gap": (-1.3, -4.4, 2.2), "hat": (86.0, 82.1, 89.6)},
    "claude-sonnet-4.6": {"source": (96.1, 93.7, 98.2), "target": (98.7, 98.3, 99.1), "xlt_gap": (-2.6, -4.9, -0.5), "hat": (99.4, 98.7, 100.0)},
    "claude-opus-4.7": {"source": (89.4, 85.6, 92.9), "target": (90.4, 89.4, 91.5), "xlt_gap": (-1.0, -2.8, 1.1), "hat": (97.3, 96.0, 98.4)},
}

BENCHMARK_TABLES = {
    "eclektic": ECLEKTIC_RESULTS,
    "mmlu_prox_lite": MMLU_PROX_LITE_RESULTS,
    "mgsmv2": MGSMV2_RESULTS,
}


def get_source_target_pairs(benchmark: str, scale: float = 0.01) -> Dict[str, Tuple[float, float]]:
    """Return `model_id -> (source_accuracy, target_accuracy)` for a paper benchmark.

    Note: Uses exact `source = target + xlt_gap` where `target` and `xlt_gap` are reported
    in Tables 17-19, falling back to Table `source_performance` point estimates.
    """
    table = BENCHMARK_TABLES[benchmark]
    pairs: Dict[str, Tuple[float, float]] = {}
    for m_id, row in table.items():
        s_est = float(row["source"][0]) * scale  # type: ignore[index]
        t_est = float(row["target"][0]) * scale  # type: ignore[index]
        pairs[m_id] = (s_est, t_est)
    return pairs


def get_model_hats(benchmark: str, scale: float = 1.0) -> Dict[str, float]:
    """Return `model_id -> hat_score` for a paper benchmark."""
    table = BENCHMARK_TABLES[benchmark]
    return {m_id: float(row["hat"][0]) * scale for m_id, row in table.items()}  # type: ignore[index]
