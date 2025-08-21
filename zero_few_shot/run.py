"""Command line interface for zero‑ and few‑shot inference.

This script provides a unified way to run any of the supported models
on a CSV of Yemeni proverbs. The user can specify zero‑shot (k=0)
or few‑shot (k>0) configurations, temperature, maximum token count
and output location. Results are written to a CSV with a standard
schema so that downstream evaluation or aggregation can be performed
easily.

Example usage:

```bash
python -m zero_few_shot.run --model gpt4o \
  --input data/Test_data.csv --output results/zero_few.csv \
  --k 3 --temp 0.2 --max_new_tokens 256
```
"""

from __future__ import annotations

import argparse
import importlib
import os
from typing import Any, Dict

import pandas as pd

from .common import load_data, select_few_shots


MODEL_MAP: Dict[str, str] = {
    "gpt4o": "zero_few_shot.gpt4o.GPT4oClient",
    "gemini": "zero_few_shot.gemini.GeminiClient",
    "allam7b": "zero_few_shot.allam7b.Allam7BClient",
    "llama3_8b": "zero_few_shot.llama3_8b.LLaMA3Client",
    "mistral7b": "zero_few_shot.mistral7b.Mistral7BClient",
    "deepseek7b": "zero_few_shot.deepseek7b.DeepSeek7BClient",
    "jais13b": "zero_few_shot.jais13b.JaisClient",
    "jais7b": "zero_few_shot.jais13b.JaisClient",  # alias for 7B model
}


def load_class(path: str):
    """Dynamically load a class from a dotted path."""
    module_name, class_name = path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Zero/Few‑shot proverb explanation generator")
    parser.add_argument("--model", required=True, choices=MODEL_MAP.keys(), help="Model driver to use")
    parser.add_argument("--input", required=True, help="Path to input CSV with a 'proverb' column")
    parser.add_argument("--output", required=True, help="Path to output CSV to write results")
    parser.add_argument("--k", type=int, default=0, help="Number of few‑shot examples to include")
    parser.add_argument("--temp", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--max_new_tokens", type=int, default=256, help="Maximum number of generated tokens")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for few‑shot sampling")
    args = parser.parse_args()

    # Load the requested client class
    class_path = MODEL_MAP[args.model]
    ClientClass = load_class(class_path)

    # Load data and sample few‑shot examples
    df = load_data(args.input)
    examples = select_few_shots(df, k=args.k, seed=args.seed)

    # Instantiate client (additional kwargs could be added via CLI in future)
    client: Any = ClientClass()
    results = client.generate(df, examples, temperature=args.temp, max_tokens=args.max_new_tokens)

    # Ensure output directory exists
    out_dir = os.path.dirname(os.path.abspath(args.output))
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    pd.DataFrame(results).to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"Wrote {len(results)} results to {args.output}")


if __name__ == "__main__":
    main()
