# -*- coding: utf-8 -*-
"""
Dataset Loader for MoE Router Tracing
-------------------------------------
Supports:
- gsm8k (math reasoning)
- wmt14 (translation)
- humaneval (python codegen)
- ds1000 (data science codegen)
- swebench (bug fixing)
- agentbench (LLM-as-agent tasks)
- wikitext2 (continuous corpora via _make_batches)
"""

import random
import torch
from datasets import load_dataset, Dataset


# ------------------------------------------------------------------------------
# Helper: safe selection for structured datasets
# ------------------------------------------------------------------------------
def safe_select(data, nsamples):
    """Return a dataset subset of up to nsamples examples."""
    n = min(len(data), nsamples)
    return data.select(range(n))


# ------------------------------------------------------------------------------
# Helper: continuous corpus window sampling (for WikiText-like data)
# ------------------------------------------------------------------------------
def _make_batches(tokenizer, text_list, seqlen: int, nsamples: int):
    """
    Tokenize the entire text corpus and randomly slice nsamples
    windows of length seqlen. Used for language-model-style data.
    """
    text = "\n".join(text_list)
    enc = tokenizer(text, return_tensors="pt", truncation=False)
    dataset = []
    for _ in range(nsamples):
        i = random.randint(0, enc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        # inp = enc.input_ids[:, i:j]
        inp  = enc.input_ids[0, i:j].contiguous() 
        attn = torch.ones_like(inp)
        dataset.append({"input_ids": inp, "attention_mask": attn})
    return dataset


# ------------------------------------------------------------------------------
# Structured Dataset Loader
# ------------------------------------------------------------------------------
def get_dataset_samples(name, tokenizer, seqlen, nsamples):
    name = name.lower().strip()
    text_list = []

    # --------------------------------------------------------------------------
    # 1. Continuous text corpora (WikiText)
    # --------------------------------------------------------------------------
    if name in ["wikitext2", "wikitext"]:
        data = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        text = "".join([" \n" if s == "" else s for s in data["text"][:1000]])
        return _make_batches(tokenizer, [text], seqlen, nsamples)

    # --------------------------------------------------------------------------
    # 2. GSM8K (math reasoning)
    # --------------------------------------------------------------------------
    elif name == "gsm8k":
        data = load_dataset("gsm8k", "main", split="train")
        subset = safe_select(data, nsamples)
        text_list = [f"Q: {ex['question']}\nA: {ex['answer']}" for ex in subset]

    # --------------------------------------------------------------------------
    # 3. WMT14 (translation)
    # --------------------------------------------------------------------------
    elif name in ["wmt", "wmt14"]:
        data = load_dataset("wmt14", "de-en", split="train")
        subset = safe_select(data, nsamples)
        text_list = [
            f"Translate from German to English:\nGerman: {ex['translation']['de']}\nEnglish: {ex['translation']['en']}"
            for ex in subset
        ]

    # --------------------------------------------------------------------------
    # 4. HumanEval (Python code generation)
    # --------------------------------------------------------------------------
    elif name == "humaneval":
        data = load_dataset("openai_humaneval", split="test")
        subset = safe_select(data, nsamples)
        text_list = [
            f"Problem: {ex['prompt']}\nReference solution:\n{ex['canonical_solution']}"
            for ex in subset
        ]

    # --------------------------------------------------------------------------
    # 5. DS-1000 (Data Science / pandas)
    # --------------------------------------------------------------------------
    elif name == "ds1000":
        data = load_dataset("xlangai/DS-1000", split="test")
        subset = safe_select(data, nsamples)
        text_list = [
            f"Instruction: {ex['prompt']}\nCode:\n{ex['reference_code']}"
            for ex in subset
        ]

    # --------------------------------------------------------------------------
    # 6. SWE-Bench (Bug fixing)
    # --------------------------------------------------------------------------
    elif name == "swebench":
        data = load_dataset("princeton-nlp/SWE-bench", split="dev")
        subset = safe_select(data, nsamples)
        text_list = [
            f"Repository: {ex['repo']}\nIssue: {ex['problem_statement']}\nPatch:\n{ex['test_patch'][:400]}"
            for ex in subset
        ]

    # Need to find proper dataset for AgentBench
    # # --------------------------------------------------------------------------
    # # 7. AgentBench (Agentic tasks)
    # # --------------------------------------------------------------------------
    # elif name == "agentbench":
    #     data = load_dataset("THUDM/AgentBench", "all", split="train[:200]")
    #     subset = safe_select(data, nsamples)
    #     text_list = []
    #     for ex in subset:
    #         prompt = ex.get("instruction") or ex.get("input") or ""
    #         output = ex.get("output") or ""
    #         text_list.append(f"Task: {prompt}\nResponse: {output}")

    else:
        raise ValueError(f"❌ Unknown dataset name: {name}")

    # --------------------------------------------------------------------------
    # Tokenize and pad/truncate to uniform sequence length
    # --------------------------------------------------------------------------
    # enc = tokenizer(
    #     text_list,
    #     return_tensors="pt",
    #     truncation=True,
    #     padding=False,
    #     max_length=seqlen,
    # )

    # dataset = [
    #     {"input_ids": inp, "attention_mask": mask}
    #     for inp, mask in zip(enc["input_ids"], enc["attention_mask"])
    # ]

    encodings = [tokenizer(t, return_tensors="pt", truncation=True, padding=False, max_length=seqlen)
             for t in text_list]

    # print([e["input_ids"].shape for e in encodings])

    dataset = [
        {"input_ids": e["input_ids"].squeeze(0), "attention_mask": e["attention_mask"].squeeze(0)}
        for e in encodings
    ]

    return dataset
