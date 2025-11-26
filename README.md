# MoE Usage-Aware Quantization Pipeline

This repo glues together:

- Router tracing for MoE models (DeepSeek-MoE, etc.)
- Usage-aware bit assignment (per expert)
- Mixed-precision GPTQ quantization via a vendored MoE-Quantization repo
- LM evaluation via lm-eval-harness

## Layout

- `external/MoE-Quantization/` — fork of the MoE-Quantization repo (patched AutoGPTQ/Optimum/etc.)
- `routing/` — tracing router outputs and expert usage
- `quantization/` — usage-aware bit assignment and quantization wrappers
- `eval/` — LM evaluation tools (e.g., GSM8K, WMT)
- `utils/` — shared dataset + model utilities