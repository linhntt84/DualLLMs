from __future__ import annotations

from src.data.gsm8k_loader import normalize_numeric_answer


def numeric_answers_equal(predicted: str | None, gold: str | None) -> bool:
    if predicted is None or gold is None:
        return False
    return normalize_numeric_answer(predicted) == normalize_numeric_answer(gold)


def score_answer(parsed_final_answer: str | None, gold_final_answer: str) -> bool:
    return numeric_answers_equal(parsed_final_answer, gold_final_answer)
