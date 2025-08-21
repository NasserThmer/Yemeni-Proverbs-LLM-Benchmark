#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Single-model fine-tuning script (causal LM) for ALLaM-style models.

- Target: ALLaM Instruct models (decoder-only), e.g. `ALLaM-AI/ALLaM-7B-Instruct-preview`.
- Uses a unified **English** instruction prompt; the model must answer **in Modern Standard Arabic**.
- Expects CSVs with columns: `proverb` and `reference_explanation`.
- Trains on Train_data.csv and evaluates on Validation_data.csv (eval_loss + optional preview generations).

Example:
  python fine_tuning/train.py \    --model_id ALLaM-AI/ALLaM-7B-Instruct-preview \    --data_dir data \    --output_dir results/finetune_allam \    --lr 2e-5 --epochs 3 --bsz 2 --grad_accum 8 \    --cutoff_len 1024 --fp16 --gradient_checkpointing
"""

import argparse
import os
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from datasets import Dataset, DatasetDict
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    set_seed,
)

# Unified English instruction (answer must be Arabic only)
INSTRUCTION = (
    "You are a linguistic expert in Yemeni proverbs. "
    "Explain the figurative meaning of the following proverb clearly and concisely "
    "in Modern Standard Arabic only. Do not translate into English.\n"
)

def build_prompt(proverb: str) -> str:
    return INSTRUCTION + f"Proverb: {proverb}\nExplanation:"

def load_splits(data_dir: str) -> DatasetDict:
    train_path = os.path.join(data_dir, "Train_data.csv")
    val_path = os.path.join(data_dir, "Validation_data.csv")
    if not os.path.exists(train_path) or not os.path.exists(val_path):
        raise FileNotFoundError(f"Expected Train_data.csv and Validation_data.csv under {data_dir}")
    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)
    for df in (df_train, df_val):
        if "proverb" not in df.columns or "reference_explanation" not in df.columns:
            raise ValueError("CSV must include 'proverb' and 'reference_explanation' columns")
    return DatasetDict(
        train=Dataset.from_pandas(df_train, preserve_index=False),
        validation=Dataset.from_pandas(df_val, preserve_index=False),
    )

@dataclass
class PreprocessConfig:
    cutoff_len: int = 1024

def build_preprocess(tokenizer: AutoTokenizer, cfg: PreprocessConfig):
    # For causal LM SFT: mask the prompt tokens (loss on answer only)
    def _proc(batch: Dict[str, List[str]]) -> Dict[str, List[List[int]]]:
        inputs = []
        attn = []
        labels = []
        for prov, ref in zip(batch["proverb"], batch["reference_explanation"]):
            prompt = build_prompt(str(prov))
            full = prompt + " " + str(ref).strip()
            tok_full = tokenizer(full, max_length=cfg.cutoff_len, truncation=True)
            tok_prompt = tokenizer(prompt, max_length=cfg.cutoff_len, truncation=True)

            input_ids = tok_full["input_ids"]
            attention_mask = tok_full["attention_mask"]
            lab = input_ids.copy()

            # Mask prompt tokens
            prompt_len = len(tok_prompt["input_ids"])
            lab[:prompt_len] = [-100] * min(prompt_len, len(lab))

            inputs.append(input_ids)
            attn.append(attention_mask)
            labels.append(lab)
        return {"input_ids": inputs, "attention_mask": attn, "labels": labels}
    return _proc

def preview_generations(model, tokenizer, dataset, max_new_tokens=128, num_examples=3):
    print("\n[Preview] Generations on first validation examples:")
    model.eval()
    for i in range(min(num_examples, len(dataset))):
        row = dataset[i]
        prompt = build_prompt(str(row["proverb"]))
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id,
            )
        gen = tokenizer.decode(out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True).strip()
        print(f"{i+1}. {gen}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="ALLaM-AI/ALLaM-7B-Instruct-preview")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--output_dir", type=str, default="results/finetune_allam")
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--bsz", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--cutoff_len", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)

    # Load model & tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=True, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        trust_remote_code=True,
    )

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    ds = load_splits(args.data_dir)
    proc = build_preprocess(tokenizer, PreprocessConfig(args.cutoff_len))
    ds_tok = ds.map(proc, batched=True, remove_columns=ds["train"].column_names)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.bsz,
        per_device_eval_batch_size=args.bsz,
        gradient_accumulation_steps=args.grad_accum,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        logging_steps=50,
        fp16=args.fp16,
        bf16=args.bf16,
        report_to=["none"],
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset(ds_tok["train"]),
        eval_dataset(ds_tok["validation"]),
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # Optional preview
    preview_generations(model, tokenizer, ds["validation"])

if __name__ == "__main__":
    main()
