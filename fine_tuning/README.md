# Fine-Tuning (ALLaM) — Training & Evaluation

This guide explains how to use the two scripts in `fine_tuning/` to produce **Modern Standard Arabic** explanations for **Yemeni proverbs** with an **English, unified instruction prompt**. The setup targets **decoder-only ALLaM**-style models (causal LM).

> Project layout assumed: `data/` (CSV files), `results/` (outputs), and `fine_tuning/` (code).

---

## Files in this folder

- `train.py` — Supervised fine-tuning (SFT) for a **causal LM**. The prompt tokens are masked in the labels so the loss is computed **only on the answer**.
- `evaluate.py` — Runs generation on `data/Test_data.csv` and writes a CSV with: `Proverb`, `Reference_Explanation` (optional), `Generated_Explanation`.
  - If you trained with PEFT (LoRA), load **base + adapter**.
  - If you trained a full fine-tune, load the **checkpoint directory**.

---

## Requirements

Use the project-level `requirements.txt`. You typically need:
- `transformers`, `datasets`, `torch`, `pandas`, `numpy`
- `peft` (only if you evaluate with a LoRA/PEFT adapter)
- If the model is gated on Hugging Face, export `HF_TOKEN` before running.

---

## Data format

### Train / Validation
- Files: `data/Train_data.csv`, `data/Validation_data.csv`
- Required columns:
  - `proverb`: the proverb text
  - `reference_explanation`: the gold/reference explanation (training target)

### Test
- File: `data/Test_data.csv`
- Required columns:
  - `proverb`
- Optional:
  - `reference_explanation` (for later comparison/metrics)

---

## Unified instruction (English → Arabic-only output)

Both scripts use the same instruction (wrapped in the model’s chat format as needed):
```
You are a linguistic expert in Yemeni proverbs.
Explain the figurative meaning of the following proverb clearly and concisely
in Modern Standard Arabic only. Do not translate into English.
Proverb: <TEXT>
Explanation:
```
This normalizes behavior across models while **forcing Arabic-only answers**.

---

## 1) Training — `train.py`

### Concept
We do **SFT for causal LMs**: concatenate *prompt + reference answer*, then mask the prompt tokens in `labels` with `-100` so the loss is computed only on the answer portion.

### Key arguments
- `--model_id` : base model ID (e.g., `ALLaM-AI/ALLaM-7B-Instruct-preview`).
- `--data_dir` : dataset directory (default: `data`).
- `--output_dir` : where to save the trained model/checkpoints.
- `--lr`, `--epochs`, `--bsz`, `--grad_accum` : training hyperparameters.
- `--cutoff_len` : maximum tokenized sequence length.
- `--fp16` / `--bf16` : mixed precision if hardware supports it.
- `--gradient_checkpointing` : memory saving at a small speed cost.

### Example
```bash
python fine_tuning/train.py   --model_id ALLaM-AI/ALLaM-7B-Instruct-preview   --data_dir data   --output_dir results/finetune_allam   --lr 2e-5 --epochs 3 --bsz 2 --grad_accum 8   --cutoff_len 1024 --fp16 --gradient_checkpointing
```

### Training outputs
- The best checkpoint (by `eval_loss`) and the final model are saved in `--output_dir`.
- Keep the entire folder to reload later.

---

## 2) Evaluation / Generation — `evaluate.py`

You can evaluate in two common ways:

### (A) Full fine-tune checkpoint
Load the trained model from your `--output_dir` and run generation on `Test_data.csv`.

```bash
python fine_tuning/evaluate.py   --checkpoint_dir results/finetune_allam   --test_path data/Test_data.csv   --out_path results/finetune.csv   --max_new_tokens 256 --temperature 0.0
```

### (B) PEFT (LoRA) adapter
Load **base + adapter** (if you used a PEFT method).

```bash
python fine_tuning/evaluate.py   --base_id ALLaM-AI/ALLaM-7B-Instruct-preview   --adapter_dir results/finetune_allam_adapter   --test_path data/Test_data.csv   --out_path results/finetune.csv   --max_new_tokens 256 --temperature 0.0
```

### Output
A CSV with columns:
- `Proverb`
- `Reference_Explanation` (if present in your test CSV)
- `Generated_Explanation`

You can later compute BERTScore, SAS, cosine similarity, etc., from this file.

---

## Practical tips

- **Memory**: For 7B models, reduce `--bsz`, increase `--grad_accum`, and enable `--gradient_checkpointing` if needed.
- **Lengths**: Increase `--cutoff_len` only when necessary. Keep `--max_new_tokens` moderate during evaluation.
- **Determinism vs. diversity**: Use `--temperature 0.0` for stable outputs; increase `temperature` (and optionally adjust `top_p`) for more diverse generations.
- **Access**: If the Hugging Face model is gated, export `HF_TOKEN` before running.

---

## Trusted references (official docs)

- Transformers — Trainer & `TrainingArguments` (training):  
  https://huggingface.co/docs/transformers/main_classes/trainer  
  https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments

- Transformers — Causal LM (language modeling):  
  https://huggingface.co/docs/transformers/tasks/language_modeling

- Transformers — Generation (sampling/decoding):  
  https://huggingface.co/docs/transformers/generation  
  https://huggingface.co/docs/transformers/generation_strategies

- Datasets (loading & preprocessing):  
  https://huggingface.co/docs/datasets

- PEFT (parameter-efficient finetuning):  
  https://huggingface.co/docs/peft/index
