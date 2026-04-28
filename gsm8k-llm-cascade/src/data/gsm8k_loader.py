from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Iterable, Iterator, Optional


@dataclass(frozen=True)
class GSM8KSample:
    sample_id: str
    question: str
    answer: str
    gold_final_answer: str


def extract_gsm8k_final_answer(answer_text: str) -> str:
    """Extract the numeric final answer from GSM8K's answer field."""
    marker_match = re.search(r"####\s*([^\n\r]+)", answer_text)
    candidate = marker_match.group(1) if marker_match else answer_text
    numbers = re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?", candidate)
    if not numbers:
        raise ValueError("Could not extract final numeric answer from GSM8K answer field.")
    return normalize_numeric_answer(numbers[-1])


def normalize_numeric_answer(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip().replace(",", "")
    text = text.rstrip(".")
    if not text:
        return ""
    try:
        decimal_value = Decimal(text)
    except InvalidOperation:
        return text
    normalized = format(decimal_value.normalize(), "f")
    if "." in normalized:
        normalized = normalized.rstrip("0").rstrip(".")
    return "0" if normalized == "-0" else normalized


def load_gsm8k_split(split: str = "test", num_samples: Optional[int] = None) -> list[GSM8KSample]:
    """Load GSM8K via Hugging Face datasets and return normalized samples."""
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "The 'datasets' package is required to load GSM8K. Install project requirements first."
        ) from exc

    dataset = load_dataset("gsm8k", "main", split=split)
    if num_samples is not None:
        dataset = dataset.select(range(min(num_samples, len(dataset))))

    samples: list[GSM8KSample] = []
    for idx, row in enumerate(dataset):
        answer = row["answer"]
        samples.append(
            GSM8KSample(
                sample_id=str(idx),
                question=row["question"],
                answer=answer,
                gold_final_answer=extract_gsm8k_final_answer(answer),
            )
        )
    return samples


def iter_gsm8k_split(split: str = "test", num_samples: Optional[int] = None) -> Iterator[GSM8KSample]:
    yield from load_gsm8k_split(split=split, num_samples=num_samples)
