import torch
import numpy as np
from typing import List, Dict


def inspect_calibration_dataset(dataset, tokenizer, max_print: int = 3):
    """
    Prints useful diagnostics to verify calibration quality.
    """

    print("\n🔍 Calibration Dataset Inspector")
    print("=" * 50)

    lengths = []
    token_freq = {}

    for i, sample in enumerate(dataset):
        ids = sample["input_ids"]
        L = len(ids)
        lengths.append(L)

        # Collect token frequency stats
        for t in ids.tolist():
            token_freq[t] = token_freq.get(t, 0) + 1

        # Print first few decoded samples
        if i < max_print:
            decoded = tokenizer.decode(ids[:200])
            print(f"\n--- Sample {i} (len={L}) ---")
            print(decoded[:500])

    lengths = np.array(lengths)

    print("\n📊 Length Stats")
    print(f"  Num samples: {len(dataset)}")
    print(f"  Min length:  {lengths.min()}")
    print(f"  Max length:  {lengths.max()}")
    print(f"  Mean length: {lengths.mean():.1f}")

    unique_tokens = len(token_freq)
    total_tokens = sum(token_freq.values())

    print("\n📊 Token Stats")
    print(f"  Unique tokens: {unique_tokens}")
    print(f"  Total tokens:  {total_tokens}")

    # Show most common tokens
    most_common = sorted(token_freq.items(), key=lambda x: x[1], reverse=True)[:10]
    print("\n🔥 Top 10 most frequent tokens:")
    for tok, cnt in most_common:
        print(f"  {tokenizer.decode([tok])!r}: {cnt}")

    print("\n✅ Dataset inspection complete.\n")
