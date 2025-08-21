# Results

This folder stores experiment outputs and optional figures.

- `zero_few.csv`: Aggregated outputs from **Zero/Few-shot** runs across the 7 drivers (under `zero_few_shot/`).
- `finetune.csv`: Outputs from the **fine‑tuned** model evaluated on `data/Test_data.csv`.
- `figures/`: Optional charts derived from the CSVs (e.g., counts per model, length histograms, metrics plots).

## Create `zero_few.csv`

1) Run each driver once to produce a temporary CSV (examples; adjust keys/tokens as needed):
```bash
python -m zero_few_shot.run --model gpt4o     --input data/Test_data.csv --output tmp_gpt4o.csv     --k 3 --temp 0.2 --max_new_tokens 64
python -m zero_few_shot.run --model gemini    --input data/Test_data.csv --output tmp_gemini.csv    --k 3
python -m zero_few_shot.run --model allam7b   --input data/Test_data.csv --output tmp_allam.csv     --k 2
python -m zero_few_shot.run --model llama3_8b --input data/Test_data.csv --output tmp_llama3.csv    --k 2
python -m zero_few_shot.run --model mistral7b --input data/Test_data.csv --output tmp_mistral.csv   --k 2
python -m zero_few_shot.run --model deepseek7b --input data/Test_data.csv --output tmp_deepseek.csv --k 2
python -m zero_few_shot.run --model jais13b   --input data/Test_data.csv --output tmp_jais.csv      --k 2
```

2) Merge them into a single file:
```bash
head -n 1 tmp_gpt4o.csv > results/zero_few.csv
for f in tmp_*.csv; do tail -n +2 "$f" >> results/zero_few.csv; done
```

**Columns (schema)**:  
`id, proverb, model, prediction, prompt, temperature, max_tokens, k_shots, task, split`

## Create `finetune.csv`

After training, generate on `Test_data.csv` and save to `results/finetune.csv`.

**Full fine‑tune checkpoint**
```bash
python fine_tuning/evaluate.py   --checkpoint_dir results/finetune_allam   --test_path data/Test_data.csv   --out_path results/finetune.csv   --max_new_tokens 256 --temperature 0.0
```

**PEFT/LoRA (base + adapter)**
```bash
python fine_tuning/evaluate.py   --base_id ALLaM-AI/ALLaM-7B-Instruct-preview   --adapter_dir results/finetune_allam_adapter   --test_path data/Test_data.csv   --out_path results/finetune.csv   --max_new_tokens 256 --temperature 0.0
```

**Alternative (standalone inference script)**
```bash
python inference_allam_english_prompt.py   --base_id ALLaM-AI/ALLaM-7B-Instruct-preview   --test_path data/Test_data.csv   --out_path results/finetune.csv   --max_new_tokens 256 --temperature 0.0
```

**Columns (schema)**:  
`Proverb, Reference_Explanation (optional), Generated_Explanation`

## Figures

Save any plots derived from the CSVs under:
```
results/figures/
├── samples_per_model.png
└── pred_len_hist.png
```
