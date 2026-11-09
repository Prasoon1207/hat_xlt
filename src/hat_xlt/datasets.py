"""Dataset language metadata, script congruence mappings, and JSONL loaders.

Implements dataset specifications from Section 5.2, Section 6.5 (script similarity),
and Appendix A.1 (Table 2) of:
"Are Multilingual Models Improving? Revisiting Cross-Lingual Transfer" (EMNLP 2026).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np


# Language codes and script families across all three benchmarks (Table 2 & Section 6.5)
LANGUAGE_SCRIPTS: Dict[str, str] = {
    "en": "Latin",       # English
    "af": "Latin",       # Afrikaans
    "ar": "Arabic",      # Arabic
    "bn": "Bengali",     # Bengali
    "cs": "Latin",       # Czech
    "de": "Latin",       # German
    "es": "Latin",       # Spanish
    "fr": "Latin",       # French
    "he": "Hebrew",      # Hebrew
    "hi": "Devanagari",  # Hindi
    "hu": "Latin",       # Hungarian
    "id": "Latin",       # Indonesian
    "it": "Latin",       # Italian
    "ja": "CJK_Kana",    # Japanese
    "ko": "Hangul",      # Korean
    "mr": "Devanagari",  # Marathi
    "ne": "Devanagari",  # Nepali
    "pt": "Latin",       # Portuguese
    "ru": "Cyrillic",    # Russian
    "sr": "Cyrillic",    # Serbian
    "sw": "Latin",       # Swahili
    "te": "Telugu",      # Telugu
    "th": "Thai",        # Thai
    "uk": "Cyrillic",    # Ukrainian
    "ur": "Arabic",      # Urdu
    "vi": "Latin",       # Vietnamese
    "wo": "Latin",       # Wolof
    "yo": "Latin",       # Yoruba
    "zh": "CJK",         # Chinese
    "zu": "Latin",       # Zulu
}

DATASET_LANGUAGES: Dict[str, Tuple[str, ...]] = {
    "mgsmv2": ("en", "bn", "zh", "fr", "de", "ja", "ru", "es", "sw", "te", "th"),
    "eclektic": ("en", "zh", "fr", "de", "he", "hi", "id", "it", "ja", "ko", "pt", "es"),
    "mmlu_prox_lite": (
        "en", "af", "ar", "bn", "zh", "cs", "fr", "de", "hi", "hu",
        "id", "it", "ja", "ko", "mr", "ne", "pt", "ru", "sr", "es",
        "sw", "te", "th", "uk", "ur", "vi", "wo", "yo", "zu",
    ),
}


def partition_by_script_congruence(
    source_lang: str,
    target_langs: Sequence[str],
    script_map: Mapping[str, str] = LANGUAGE_SCRIPTS,
) -> Tuple[List[str], List[str]]:
    """Split target languages into `(same_script, different_script)` relative to `source_lang` (Section 6.5)."""
    if source_lang not in script_map:
        raise KeyError(f"Unknown source language code `{source_lang}`.")
    src_script = script_map[source_lang]
    same_script = [lang for lang in target_langs if script_map.get(lang) == src_script]
    diff_script = [lang for lang in target_langs if script_map.get(lang) != src_script]
    return same_script, diff_script


def load_instances_jsonl(
    path: Union[str, Path],
    target_languages: Optional[Sequence[str]] = None,
    script_filter: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load instance-level `(source_pass_rates, target_pass_rates)` from a JSONL file.

    Supported per-line JSON schemas:
      1. Precomputed pass-rates:
         `{"id": "q1", "source_lang": "en", "source_pass_rate": 0.8, "target_pass_rates": {"de": 0.7, "hi": 0.5}}`
         or `"target_pass_rate": 0.6`
      2. Raw binary sample verdicts (`N` passes, e.g. `N=10`):
         `{"id": "q1", "source_lang": "en", "source_correct": [1,1,0,...], "target_correct": {"de": [1,0,...], ...}}`

    Args:
        path: Path to `.jsonl` file.
        target_languages: Optional subset of target language codes to average over.
        script_filter: Optional `"same"` or `"different"` to restrict each item's target
            languages by script congruence with `source_lang` (supports per-item source
            languages as in ECLeKTic!).

    Returns:
        `(source_pass_rates, target_pass_rates)` as 1D float64 NumPy arrays of shape `(n_items,)`.
    """
    src_rates: List[float] = []
    tgt_rates: List[float] = []

    filepath = Path(path)
    with filepath.open("r", encoding="utf-8") as f:
        for line_num, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue
            record = json.loads(line)

            # Extract source pass-rate
            if "source_pass_rate" in record:
                s_val = float(record["source_pass_rate"])
            elif "source_correct" in record:
                s_arr = np.asarray(record["source_correct"], dtype=np.float64)
                s_val = float(np.mean(s_arr))
            else:
                raise ValueError(
                    f"Line {line_num}: expected `source_pass_rate` or `source_correct`."
                )

            # Extract target pass-rate(s)
            if "target_pass_rate" in record and target_languages is None and script_filter is None:
                t_val = float(record["target_pass_rate"])
            else:
                per_lang: Dict[str, float] = {}
                if "target_pass_rates" in record:
                    tp = record["target_pass_rates"]
                    if isinstance(tp, dict):
                        per_lang = {k: float(v) for k, v in tp.items()}
                    else:
                        t_val = float(np.mean(np.asarray(tp, dtype=np.float64)))
                        src_rates.append(s_val)
                        tgt_rates.append(t_val)
                        continue
                elif "target_correct" in record:
                    tc = record["target_correct"]
                    if isinstance(tc, dict):
                        per_lang = {
                            k: float(np.mean(np.asarray(v, dtype=np.float64)))
                            for k, v in tc.items()
                        }
                    else:
                        t_val = float(np.mean(np.asarray(tc, dtype=np.float64)))
                        src_rates.append(s_val)
                        tgt_rates.append(t_val)
                        continue
                else:
                    raise ValueError(
                        f"Line {line_num}: expected `target_pass_rate(s)` or `target_correct`."
                    )

                langs = list(per_lang.keys())
                if target_languages is not None:
                    allowed = set(target_languages)
                    langs = [l for l in langs if l in allowed]

                if script_filter is not None:
                    src_lang = record.get("source_lang", "en")
                    same_l, diff_l = partition_by_script_congruence(src_lang, langs)
                    if script_filter == "same":
                        langs = same_l
                    elif script_filter == "different":
                        langs = diff_l
                    else:
                        raise ValueError(
                            f"Invalid `script_filter`={script_filter!r}; expected 'same' or 'different'."
                        )

                if not langs:
                    continue
                t_val = float(np.mean([per_lang[l] for l in langs]))

            src_rates.append(s_val)
            tgt_rates.append(t_val)

    if not src_rates:
        raise ValueError(f"No valid evaluation instances loaded from {filepath}.")

    return np.asarray(src_rates, dtype=np.float64), np.asarray(tgt_rates, dtype=np.float64)
