from __future__ import annotations

import re
from dataclasses import dataclass

from src.data.gsm8k_loader import normalize_numeric_answer


@dataclass(frozen=True)
class ParsedAnswer:
    value: str | None
    parse_success: bool


FINAL_ANSWER_RE = re.compile(
    r"Final answer:\s*([-+]?\d[\d,]*(?:\.\d+)?)",
    flags=re.IGNORECASE,
)
BOXED_ANSWER_RE = re.compile(
    r"\\boxed\{\s*([-+]?\d[\d,]*(?:\.\d+)?)\s*\}",
)
NUMBER_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")


def parse_final_answer(text: str | None) -> ParsedAnswer:
    if not text:
        return ParsedAnswer(value=None, parse_success=False)

    for pattern in (FINAL_ANSWER_RE, BOXED_ANSWER_RE, NUMBER_RE):
        matches = pattern.findall(text)
        if matches:
            return ParsedAnswer(value=normalize_numeric_answer(matches[-1]), parse_success=True)

    return ParsedAnswer(value=None, parse_success=False)
