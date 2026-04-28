from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import yaml

from src.data.gsm8k_loader import iter_gsm8k_split
from src.eval.answer_parser import parse_final_answer
from src.eval.scorer import score_answer
from src.logging_utils.jsonl_logger import JSONLLogger, write_json
from src.models.model_runner import create_model_runner
from src.prompts.prompt_registry import build_prompt


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def apply_overrides(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    updated = dict(config)
    if args.num_samples is not None:
        updated["num_samples"] = args.num_samples
    if args.output_dir is not None:
        updated["output_dir"] = args.output_dir
    return updated


def ensure_output_dirs(output_dir: str | Path) -> dict[str, Path]:
    root = Path(output_dir)
    paths = {
        "root": root,
        "logs": root / "logs",
        "metrics": root / "metrics",
        "error_analysis": root / "error_analysis",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def empty_d0_record(config: dict[str, Any], sample: Any) -> dict[str, Any]:
    return {
        "experiment_id": config["experiment_id"],
        "design_id": config["design_id"],
        "sample_id": sample.sample_id,
        "question": sample.question,
        "gold_final_answer": sample.gold_final_answer,
        "llm1_model": None,
        "llm1_prompt_id": None,
        "llm1_decoding_config": None,
        "a1_text": None,
        "llm2_model": config["llm2_model_name"],
        "llm2_prompt_id": config["llm2_prompt_id"],
        "llm2_decoding_config": config["llm2_decoding"],
        "a2_text": None,
        "parsed_final_answer": None,
        "parse_success": False,
        "is_correct": False,
        "a1_parsed_answer_optional": None,
        "a1_is_correct_optional": None,
        "a1_format_compliance_optional": None,
        "a1_num_tokens_optional": None,
        "a2_num_tokens_optional": None,
        "runtime_seconds_optional": None,
        "error_message_optional": None,
    }


def run_d0(config: dict[str, Any]) -> dict[str, Any]:
    output_paths = ensure_output_dirs(config["output_dir"])
    log_path = output_paths["logs"] / "D0_direct_llm2.jsonl"
    metrics_path = output_paths["metrics"] / "D0_metrics.json"
    error_path = output_paths["error_analysis"] / "D0_errors.jsonl"

    for path in (log_path, error_path):
        path.write_text("", encoding="utf-8")

    logger = JSONLLogger(log_path)
    error_logger = JSONLLogger(error_path)
    runner = create_model_runner(config["llm2_model_name"])

    totals = {
        "num_samples": 0,
        "num_correct": 0,
        "num_parse_success": 0,
        "a2_token_sum": 0,
        "a2_token_count": 0,
    }

    samples = iter_gsm8k_split(
        split=config.get("dataset_split", "test"),
        num_samples=config.get("num_samples"),
    )
    for sample in samples:
        started = time.perf_counter()
        record = empty_d0_record(config, sample)
        try:
            prompt = build_prompt(config["llm2_prompt_id"], question=sample.question)
            generation = runner.generate(prompt, config["llm2_decoding"])
            parsed = parse_final_answer(generation.text)
            is_correct = score_answer(parsed.value, sample.gold_final_answer)

            record.update(
                {
                    "a2_text": generation.text,
                    "parsed_final_answer": parsed.value,
                    "parse_success": parsed.parse_success,
                    "is_correct": is_correct,
                    "a2_num_tokens_optional": generation.num_tokens,
                }
            )
        except Exception as exc:
            record["error_message_optional"] = f"{type(exc).__name__}: {exc}"
        finally:
            record["runtime_seconds_optional"] = round(time.perf_counter() - started, 6)
            logger.write(record)
            if not record["parse_success"] or not record["is_correct"]:
                error_logger.write(record)

            totals["num_samples"] += 1
            totals["num_correct"] += int(bool(record["is_correct"]))
            totals["num_parse_success"] += int(bool(record["parse_success"]))
            if record["a2_num_tokens_optional"] is not None:
                totals["a2_token_sum"] += int(record["a2_num_tokens_optional"])
                totals["a2_token_count"] += 1

    num_samples = totals["num_samples"]
    metrics = {
        "num_samples": num_samples,
        "a2_accuracy": totals["num_correct"] / num_samples if num_samples else 0.0,
        "parse_success_rate": totals["num_parse_success"] / num_samples if num_samples else 0.0,
        "avg_a2_tokens": (
            totals["a2_token_sum"] / totals["a2_token_count"]
            if totals["a2_token_count"]
            else 0.0
        ),
    }
    write_json(metrics_path, metrics)
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run D0 direct LLM2 GSM8K baseline.")
    parser.add_argument("--config", required=True, help="Path to D0 YAML config.")
    parser.add_argument("--num_samples", type=int, default=None, help="Override number of GSM8K samples.")
    parser.add_argument("--output_dir", default=None, help="Override output directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = apply_overrides(load_config(args.config), args)
    metrics = run_d0(config)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
