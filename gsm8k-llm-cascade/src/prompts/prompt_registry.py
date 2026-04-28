from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PromptTemplate:
    prompt_id: str
    template: str

    def render(self, **kwargs: object) -> str:
        return self.template.format(**kwargs)


D0_DIRECT_LLM2 = PromptTemplate(
    prompt_id="d0_direct_llm2",
    template=(
        "You are solving a grade-school math word problem.\n\n"
        "Question:\n"
        "{question}\n\n"
        "Solve the problem step by step. At the end, provide the final numeric answer "
        "in exactly this format:\n\n"
        "Final answer: <number>"
    ),
)


_PROMPTS = {
    D0_DIRECT_LLM2.prompt_id: D0_DIRECT_LLM2,
}


def get_prompt_template(prompt_id: str) -> PromptTemplate:
    try:
        return _PROMPTS[prompt_id]
    except KeyError as exc:
        available = ", ".join(sorted(_PROMPTS))
        raise ValueError(f"Unknown prompt_id '{prompt_id}'. Available prompts: {available}") from exc


def build_prompt(prompt_id: str, **kwargs: object) -> str:
    return get_prompt_template(prompt_id).render(**kwargs)
