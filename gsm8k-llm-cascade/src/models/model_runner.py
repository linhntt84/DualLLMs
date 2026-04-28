from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class GenerationResult:
    text: str
    num_tokens: int | None = None


class ModelRunner(Protocol):
    model_name: str

    def generate(self, prompt: str, decoding_config: dict[str, Any]) -> GenerationResult:
        ...


class TransformersModelRunner:
    def __init__(self, model_name: str):
        self.model_name = model_name
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "The 'torch' and 'transformers' packages are required for LLM2 generation."
            ) from exc

        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._device, self._torch_dtype, self._use_device_map = self._select_device_options()
        model_kwargs: dict[str, Any] = {"torch_dtype": self._torch_dtype}
        if self._use_device_map:
            model_kwargs["device_map"] = "auto"

        self._model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        if not self._use_device_map:
            self._model.to(self._device)
        self._model.eval()

    def _select_device_options(self) -> tuple[str, Any, bool]:
        if self._torch.cuda.is_available():
            return "cuda", self._torch.float16, True
        if self._torch.backends.mps.is_available():
            return "mps", self._torch.float16, False
        return "cpu", self._torch.float32, False

    def generate(self, prompt: str, decoding_config: dict[str, Any]) -> GenerationResult:
        messages = [{"role": "user", "content": prompt}]
        if hasattr(self._tokenizer, "apply_chat_template"):
            model_input = self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            model_input = prompt

        inputs = self._tokenizer(model_input, return_tensors="pt").to(self._model.device)
        max_new_tokens = int(decoding_config.get("max_new_tokens", 512))
        temperature = float(decoding_config.get("temperature", 0.0))
        do_sample = temperature > 0.0
        generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "top_p": float(decoding_config.get("top_p", 1.0)),
        }
        if do_sample:
            generation_kwargs["temperature"] = temperature

        with self._torch.no_grad():
            generated = self._model.generate(**inputs, **generation_kwargs)

        input_length = inputs["input_ids"].shape[-1]
        output_ids = generated[0][input_length:]
        text = self._tokenizer.decode(output_ids, skip_special_tokens=True).strip()
        return GenerationResult(text=text, num_tokens=int(output_ids.shape[-1]))


class MockModelRunner:
    """Deterministic test runner for smoke tests without model dependencies."""

    model_name = "mock-llm2"

    def generate(self, prompt: str, decoding_config: dict[str, Any]) -> GenerationResult:
        return GenerationResult(text="Solution:\nMock solution.\nFinal answer: 0", num_tokens=6)


def create_model_runner(model_name: str) -> ModelRunner:
    if model_name == "mock":
        return MockModelRunner()
    return TransformersModelRunner(model_name)
