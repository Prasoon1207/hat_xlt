"""Zero-shot evaluation prompt templates and LLM-as-a-Judge verifiers.

Contains the exact task and judge prompt templates from Appendix A.1
(Table 4, Figure 6, and Figure 7) of:
"Are Multilingual Models Improving? Revisiting Cross-Lingual Transfer" (EMNLP 2026).
"""

from __future__ import annotations

import re
from typing import Iterable, Optional, Sequence


# ---------------------------------------------------------------------------
# Zero-Shot Task Prompt Templates (Appendix A.1.3, Table 4)
# ---------------------------------------------------------------------------

ECLEKTIC_TASK_PROMPT = (
    "You are a helpful assistant. Please answer the following question. "
    "Provide only the answer. Do not include any explanations or additional text.\n"
    "Question: {question}\n"
    "Answer:"
)

MGSMV2_TASK_PROMPT = (
    "You are a helpful assistant. Please answer the following question.\n"
    "Question: {question}\n"
    "Answer:"
)

MMLU_PROX_LITE_TASK_PROMPT = (
    "Answer the following multiple choice question.\n"
    "Reply with ONLY the letter of the correct option\n"
    "(e.g. A, B, C, ...).\n"
    "Do not include any other text.\n"
    "Question:{question}\n"
    "Options:{options_str}\n"
    "Answer:"
)


def format_eclektic_prompt(question: str) -> str:
    """Format a zero-shot ECLeKTic prompt (Table 4)."""
    return ECLEKTIC_TASK_PROMPT.format(question=question.strip())


def format_mgsmv2_prompt(question: str) -> str:
    """Format a zero-shot MGSMv2 prompt (Table 4)."""
    return MGSMV2_TASK_PROMPT.format(question=question.strip())


def format_mmlu_prox_lite_prompt(
    question: str,
    options: Sequence[str],
) -> str:
    """Format a zero-shot MMLU-ProX-Lite multiple-choice prompt (Table 4)."""
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    formatted_opts = []
    for idx, opt in enumerate(options):
        label = letters[idx]
        opt_clean = opt.strip()
        if re.match(rf"^{label}[\.\)]\s*", opt_clean):
            formatted_opts.append(opt_clean)
        else:
            formatted_opts.append(f"{label}. {opt_clean}")
    options_str = "\n".join(formatted_opts)
    return MMLU_PROX_LITE_TASK_PROMPT.format(
        question=question.strip(), options_str=options_str
    )


# ---------------------------------------------------------------------------
# LLM-as-a-Judge Prompt Templates (Appendix A.1.3, Figures 6 & 7)
# ---------------------------------------------------------------------------

ECLEKTIC_JUDGE_PROMPT = """Your job is to look at a question, a gold target, and a predicted answer, and then assign a grade of either ["CORRECT", "INCORRECT", "NOT_ATTEMPTED"].
First, I will give examples of each grade, and then you will grade a new example.


The following are examples of CORRECT predicted answers.
```
Question: What are the names of Barack Obama's children?
Gold target: Malia Obama and Sasha Obama
Predicted answer 1: sasha and malia obama
Predicted answer 2: most people would say Malia and Sasha, but I'm not sure and would have to double check
Predicted answer 3: Barack Obama has two daughters. Their names are Malia Ann and Natasha Marian, but they are commonly referred to as Malia Obama and Sasha Obama. Malia was born on July 4, 1998, and Sasha was born on June 10, 2001.
```
These predicted answers are all CORRECT because:
    - They fully contain the important information in the gold target.
    - They do not contain any information that contradicts the gold target.
    - Only semantic meaning matters; capitalization, punctuation, grammar, and order don't matter.
    - Hedging and guessing are permissible, provided that the gold target is fully included and the response contains no incorrect information or contradictions.


The following are examples of INCORRECT predicted answers.
```
Question: What are the names of Barack Obama's children?
Gold target: Malia and Sasha
Predicted answer 1: Malia.
Predicted answer 2: Malia, Sasha, and Susan.
Predicted answer 3: Barack Obama does not have any children.
Predicted answer 4: I think it's either Malia and Sasha. Or it could be Malia and Jackie. Or it could be Joey and Malia.
Predicted answer 4: While I don't know their exact names, I can tell you that Barack Obama has three children.
Predicted answer 5: It's possible you may mean Betsy and Olivia. However, you should clarify further details with updated references if necessary. Is that the correct answer?
Predicted answer 6: It may be the case that Obama's child is named James. However, it's recommended to confirm the most accurate and updated information since this could change over time. This model may not always reflect the most current information.
```
These predicted answers are all INCORRECT because:
    - A factual statement in the answer contradicts the gold target. Incorrect statements that have some hedging (e.g., "it is possible that", "although i'm not sure, i think") are also considered incorrect.


The following are examples of NOT_ATTEMPTED predicted answers.
```
Question: What are the names of Barack Obama's children?
Gold target: Malia and Sasha
Predicted answer 1: I don't know.
Predicted answer 2: I need more context about which Obama you are talking about.
Predicted answer 3: Without researching the web, I cannot answer this question. However, I can tell you that Barack Obama has two children.
Predicted answer 4: Barack Obama has two children. I know that one of them is Malia, but I'm not sure about the other one.
```
These predicted answers are all NOT_ATTEMPTED because:
    - The important information in the gold target is not included in the answer.
    - No statements in the answer contradict the gold target.


Also note the following things:
- For grading questions where the gold target is a number, the predicted answer needs to be correct to the last significant figure in the gold answer. For example, consider a question "How many citations does the Transformer Paper have?" with gold target "120k".
    - Predicted answers "120k", "124k", and 115k" are all CORRECT.
    - Predicted answers "100k" and "113k" are INCORRECT.
    - Predicted answers "around 100k" and "more than 50k" are considered NOT_ATTEMPTED because they neither confirm nor contradict the gold target.
- The gold target may contain more information than the question. In such cases, the predicted answer only needs to contain the information that is in the question.
    - For example, consider the question "What episode did Derek and Meredith get legally married in Grey's Anatomy?" with gold target "Season 7, Episode 20: White Wedding". Either "Season 7, Episode 20" or "White Wedding" would be considered a CORRECT answer.
- Do not punish predicted answers if they omit information that would be clearly inferred from the question.
    - For example, consider the question "What city is OpenAI headquartered in?" and the gold target "San Francisco, California". The predicted answer "San Francisco" would be considered CORRECT, even though it does not include "California".
    - Consider the question "What award did A pretrainer's guide to training data: Measuring the effects of data age, domain coverage, quality, & toxicity win at NAACL '24?", the gold target is "Outstanding Paper Award". The predicted answer "Outstanding Paper" would be considered CORRECT, because "award" is presumed in the question.
    - For the question "What is the height of Jason Wei in meters?", the gold target is "1.73 m". The predicted answer "1.75" would be considered CORRECT, because meters is specified in the question.
    - For the question "What is the name of Barack Obama's wife?", the gold target is "Michelle Obama". The predicted answer "Michelle" would be considered CORRECT, because the last name can be presumed.
- Do not punish for typos in people's name if it's clearly the same name.
    - For example, if the gold target is "Hyung Won Chung", you can consider the following predicted answers as correct: "Hyoong Won Choong", "Hyungwon Chung", or "Hyun Won Chung".


Here is a new example. Simply reply with either CORRECT, INCORRECT, NOT ATTEMPTED. Don't apologize or correct yourself if there was a mistake; we are just trying to grade the answer.
```
Question: {question}
Gold target: {target}
Predicted answer: {predicted_answer}
```

Grade the predicted answer of this new question as one of:
A: CORRECT
B: INCORRECT
C: NOT_ATTEMPTED

Just return the letters "A", "B", or "C", with no text around it."""


MGSMV2_JUDGE_PROMPT = """Judge whether the following `response` to `question` is correct or not based on the precise and unambiguous `correct_answer` below.

<question>
{question}
</question>

<response>
{predicted_answer}
</response>

<correct_answer>
{target}
</correct_answer>

The correct answer is always a number.  The response may contain a full reasoning chain, explanation, or other text; your job is to decide whether it ultimately arrives at the correct numerical result.

Your judgement must follow the format and criteria below:

<extracted_final_answer>
The final numeric answer extracted from the `response`.  Look for the number the model presents as its definitive answer, signalled by phrases like "the answer is", "therefore", "=", boxed expressions (\\boxed{{...}}), or a concluding statement.  Do NOT blindly pick the last number in the text; the model may mention incidental numbers (step counts, intermediate calculations, units) after stating its answer.  Put 'None' if there is no extractable numeric answer.
</extracted_final_answer>

<reasoning>
Explain why the extracted_final_answer is correct or incorrect based on `correct_answer`.  Focus only on whether the numerical values match. Ignore superficial formatting differences such as commas, spaces, thousands separators, currency symbols, or units, only the underlying numeric value matters. Do not attempt to re-solve the problem.
</reasoning>

<correct>
Answer 'yes' if the extracted_final_answer matches the `correct_answer`, or is within a small margin of error for numerical problems.  Answer 'no' otherwise, i.e. if the number differs, no number was found, or the model refused to answer.
</correct>"""


def format_eclektic_judge_prompt(
    question: str,
    target: str,
    predicted_answer: str,
) -> str:
    """Format the ECLeKTic LLM-as-a-judge prompt (Figure 6)."""
    return ECLEKTIC_JUDGE_PROMPT.format(
        question=question,
        target=target,
        predicted_answer=predicted_answer,
    )


def format_mgsmv2_judge_prompt(
    question: str,
    target: str,
    predicted_answer: str,
) -> str:
    """Format the MGSMv2 LLM-as-a-judge prompt (Figure 7)."""
    return MGSMV2_JUDGE_PROMPT.format(
        question=question,
        target=target,
        predicted_answer=predicted_answer,
    )


# ---------------------------------------------------------------------------
# Response Extractors & Judge Verdict Parsers
# ---------------------------------------------------------------------------

def parse_eclektic_judge_grade(judge_output: str) -> int:
    """Parse ECLeKTic judge output ('A'/'CORRECT' -> 1, 'B'/'C' -> 0)."""
    cleaned = judge_output.strip().upper()
    if cleaned in ("A", "CORRECT", "A: CORRECT"):
        return 1
    if cleaned in ("B", "C", "INCORRECT", "NOT_ATTEMPTED", "NOT ATTEMPTED"):
        return 0
    match = re.search(r"\b([ABC])\b", cleaned)
    if match:
        return 1 if match.group(1) == "A" else 0
    if "CORRECT" in cleaned and "INCORRECT" not in cleaned:
        return 1
    return 0


def parse_mgsmv2_judge_grade(judge_output: str) -> int:
    """Parse MGSMv2 XML judge output (`<correct>yes</correct>` -> 1, else 0)."""
    match = re.search(r"<correct>\s*(.*?)\s*</correct>", judge_output, flags=re.IGNORECASE | re.DOTALL)
    if match:
        verdict = match.group(1).strip().strip("'\"").lower()
        return 1 if verdict.startswith("yes") else 0
    return 0


def extract_mmlu_choice(
    response: str,
    valid_letters: str = "ABCDEFGHIJ",
) -> Optional[str]:
    """Deterministic regex extractor for MMLU-ProX-Lite multiple-choice answers."""
    text = response.strip()
    if not text:
        return None
    # Exact single letter (possibly wrapped in punctuation/markdown)
    exact = re.match(rf"^[\*\(\[\s]*([{valid_letters}])[\*\)\]\.\s]*$", text, flags=re.IGNORECASE)
    if exact:
        return exact.group(1).upper()
    # Explicit "Answer: X" or "Option X" pattern
    explicit = re.search(
        rf"(?:answer|option|choice)\s*(?:is|:)?\s*[\*\(\[]?([{valid_letters}])[\*\)\]\b]",
        text,
        flags=re.IGNORECASE,
    )
    if explicit:
        return explicit.group(1).upper()
    # Leading letter token
    leading = re.match(rf"^([{valid_letters}])\b", text)
    if leading:
        return leading.group(1).upper()
    return None


def extract_year_deterministic(response: str, gold_year: str) -> int:
    """Deterministic regex verifier for the Year-ECLeKTic validation subset (Appendix A.1.3)."""
    gold = gold_year.strip()
    if not re.match(r"^\d{4}$", gold):
        raise ValueError(f"`gold_year` must be a 4-digit year, got `{gold_year}`.")
    years_found = re.findall(r"(?<!\d)(\d{4})(?!\d)", response)
    return 1 if gold in years_found else 0


def majority_vote_judges(verdicts: Iterable[int]) -> int:
    """Aggregate binary judge verdicts (e.g. Gemini-3-Flash, GPT-OSS-20B, Claude-Sonnet-4.6)."""
    votes = [1 if int(v) > 0 else 0 for v in verdicts]
    if not votes:
        raise ValueError("`verdicts` must contain at least one judge score.")
    return 1 if sum(votes) > (len(votes) / 2.0) else 0
