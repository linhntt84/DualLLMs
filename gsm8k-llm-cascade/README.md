# GSM8K LLM Cascade

This repo implements a prompting-only experimental framework for GSM8K.

Current stage: D0 direct LLM2 baseline.

No training in the current stage.

## D0 Pipeline

```text
Question -> LLM2 -> A2 -> parse final answer -> score -> log