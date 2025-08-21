#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Inference script for ALLaM (decoder‑only) with unified English prompt and
Arabic‑only answers.

This version improves robustness when loading test CSVs by normalising
column names (accepting ``Proverbs`` and ``Explanation``) and fixes
minor inconsistencies in the output format.

- Loads a base ALLaM model and optionally a PEFT adapter.
- Reads `Test_data.csv` (expects at least a `proverb` column, case
  insensitive, and optionally a `reference_explanation` or
  `explanation` column).
- Generates explanations and writes a CSV file with results.
"""

import argparse
import os
import datetime
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


PROMPT = (
    "[INST] You are a linguistic expert in Yemeni proverbs. "
    "Explain the figurative meaning of the following proverb clearly and concisely "
    "in Modern Standard Arabic only. Do not translate into English.\n"
    "Proverb: {proverb}\nExplanation:\n[/INST]\n"
)


def build_prompt(p: str) -> str:
    return PROMPT.format(proverb=str(p))


def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise column names for the test CSV.

    Accepts ``Proverbs`` or ``proverbs`` and maps to ``proverb``.  Accepts
    ``Explanation`` or ``reference_explanation`` (any case) and maps to
    ``reference_explanation``.
    """
    rename_map: dict[str, str] = {}
    lower_to_orig = {c.lower(): c for c in df.columns}
    # Proverb column
    if "proverb" in lower_to_orig:
        rename_map[lower_to_orig["proverb"]] = "proverb"
    elif "proverbs" in lower_to_orig:
        rename_map[lower_to_orig["proverbs"]] = "proverb"
    # Explanation column
    if "reference_explanation" in lower_to_orig:
        rename_map[lower_to_orig["reference_explanation"]] = "reference_explanation"
    elif "explanation" in lower_to_orig:
        rename_map[lower_to_orig["explanation"]] = "reference_explanation"
    if rename_map:
        df = df.rename(columns=rename_map)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_id", type=str, default="ALLaM-AI/ALLaM-7B-Instruct-preview")
    ap.add_argument("--adapter_dir", type=str, default=None, help="Optional PEFT adapter directory")
    ap.add_argument("--test_path", type=str, default="data/Test_data.csv")
    ap.add_argument("--out_path", type=str, default=None)
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    args = ap.parse_args()

    # Load tokenizer (prefer adapter dir if provided to include any added tokens)
    tok_src = args.adapter_dir if args.adapter_dir else args.base_id
    tok = AutoTokenizer.from_pretrained(tok_src, use_fast=True, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # Load model
    model = AutoModelForCausalLM.from_pretrained(args.base_id, trust_remote_code=True)
    if args.adapter_dir:
        model = PeftModel.from_pretrained(model, args.adapter_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    # Read test data
    df = pd.read_csv(args.test_path)
    df = _standardize_columns(df)
    if "proverb" not in df.columns:
        raise ValueError("Test CSV must contain a 'proverb' (or 'Proverbs') column.")

    provs = df["proverb"].astype(str).tolist()
    # Reference explanations if available
    refs = df["reference_explanation"].astype(str).tolist() if "reference_explanation" in df.columns else [None] * len(provs)

    gens: list[str] = []
    eos_id = tok.eos_token_id
    pad_id = tok.pad_token_id

    do_sample = args.temperature > 0.0

    for p in provs:
        prompt = build_prompt(p)
        enc = tok(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=args.max_new_tokens,
                do_sample=do_sample,
                temperature=args.temperature if do_sample else None,
                top_p=args.top_p if do_sample else None,
                eos_token_id=eos_id,
                pad_token_id=pad_id,
            )
        # Keep only generated continuation
        new_tokens = out[0][enc["input_ids"].shape[-1] :]
        gen_text = tok.decode(new_tokens, skip_special_tokens=True).strip()
        gens.append(gen_text)

    # Build output DataFrame
    out_df = pd.DataFrame(
        {
            "Proverb": provs,
            "Reference_Explanation": refs,
            "Generated_Explanation": gens,
        }
    )

    # Determine output path
    if args.out_path:
        out_path = args.out_path
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
    else:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join("results", f"finetune_{ts}.csv")
        os.makedirs("results", exist_ok=True)

    out_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print("Saved:", out_path)
    print(out_df.head(5))


if __name__ == "__main__":
    main()