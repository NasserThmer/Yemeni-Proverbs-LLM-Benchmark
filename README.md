# Yemeni Proverbs Benchmark (Improved)

This repository contains a benchmark and tooling for explaining Yemeni Arabic
proverbs with large language models.  It includes the original dataset,
zero‑/few‑shot evaluation scripts for several frontier and open‑source models,
and a fine‑tuning pipeline.  The present version fixes a number of issues in
the upstream repository and clarifies installation and usage requirements.

## Repository Structure

```
Yemeni‑Proverbs‑Benchmark/
├── data/                  # Train/validation/test splits (CSV)
├── zero_few_shot/         # drivers for zero‑/few‑shot evaluation
│   ├── common.py          # shared helpers for loading data and building prompts
│   ├── gpt4o.py           # OpenAI GPT‑4o client
│   ├── gemini.py          # Google Gemini client
│   ├── allam7b.py         # ALLaM‑7B client
│   ├── llama3_8b.py       # LLaMA‑3‑8B client
│   ├── mistral7b.py       # Mistral‑7B client
│   ├── deepseek7b.py      # DeepSeek‑7B Chat client
│   └── jais13b.py         # Jais chat client
├── fine_tuning/
│   ├── train.py           # single‑model fine‑tuning script
│   └── evaluate.py        # inference for fine‑tuned models
├── results/               # will contain generated results
├── requirements.txt       # dependencies
└── README.md              # this file
```

### Data Format

Each CSV in `data/` must contain at least a column for the proverb text.  The
default name for this column is `proverb`, but the loader will accept
`proverbs` (capitalized or lower‑case) and map it to the expected name.  If
reference explanations are available, place them in a column named
`reference_explanation` or simply `explanation` – these will be aliased
automatically when loading.

### Installation

```bash
git clone <this repository>
cd Yemeni‑Proverbs‑Benchmark
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Several of the drivers require API keys.  Set the following environment
variables as needed before running any scripts:

* **`OPENAI_API_KEY`** – required for the GPT‑4o driver.
* **`GEMINI_API_KEY`** or **`GOOGLE_API_KEY`** – required for the Gemini driver.
* **`HF_TOKEN`** – optional Hugging Face access token for gated models like
  ALLaM, LLaMA‑3‑8B, Mistral‑7B, DeepSeek 7B and Jais.  If a model is gated,
  you must set this token or the driver will raise an error during model load.

### Zero‑/Few‑Shot Usage

Each model has a corresponding driver in `zero_few_shot/`.  These drivers
accept the following arguments:

* `--mode` – either `zero` for zero‑shot evaluation or `few` for few‑shot.
* `--shots` – number of few‑shot examples to include (default 5; ignored in
  zero‑shot mode).
* `--train` – path to `Train_data.csv` when using few‑shot mode.
* `--test` – path to the test CSV.

Example (LLaMA‑3‑8B):

```bash
python zero_few_shot/llama3_8b.py --mode zero --test data/Test_data.csv
python zero_few_shot/llama3_8b.py --mode few --shots 5 --train data/Train_data.csv --test data/Test_data.csv
```

### Fine‑Tuning Usage

The fine‑tuning pipeline trains a single decoder‑only model using a unified
English instruction prompt; the model must answer in Modern Standard Arabic.

```bash
# Train
python fine_tuning/train.py --model_id ALLaM-AI/ALLaM-7B-Instruct-preview \
    --data_dir data --output_dir results/finetune_allam --lr 2e-5 --epochs 3 --bsz 2 \
    --grad_accum 8 --cutoff_len 1024 --fp16

# Evaluate on test set
python fine_tuning/evaluate.py \
    --base_id ALLaM-AI/ALLaM-7B-Instruct-preview \
    --adapter_dir results/finetune_allam \
    --test_path data/Test_data.csv \
    --out_path results/finetune.csv
```

### Notes

This repository is provided under the MIT license for the code and CC‑BY 4.0
for the data.  Please ensure you have permission to use any third‑party
models accessed through the Hugging Face Hub or other providers.
