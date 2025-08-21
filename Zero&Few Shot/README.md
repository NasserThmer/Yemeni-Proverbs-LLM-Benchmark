# Zero & Few-Shot Drivers

This folder contains **drivers** for running experiments in **Zero-shot** and **Few-shot** settings using seven different LLMs.

---

## 📂 Files

- `common.py` : Shared utilities (data loading, evaluation, saving results).
- `gpt4o.py` : Driver for GPT-4o.
- `gemini.py` : Driver for Gemini 1.5 Pro.
- `allam7b.py` : Driver for ALLAM-7B.
- `llama3_8b.py` : Driver for LLaMA-3-8B-Instruct.
- `mistral7b.py` : Driver for Mistral-7B-Instruct.
- `deepseek7b.py` : Driver for DeepSeek-7B-Chat.
- `jais13b.py` : Driver for Jais-13B.

---

## ⚙️ Requirements

Install the project dependencies:

```bash
pip install -r ../requirements.txt
```

---

## 🚀 Running Experiments

Each driver can be executed directly:

```bash
python gpt4o.py --mode zero   # Run Zero-shot
python gpt4o.py --mode few    # Run Few-shot
```

Replace `gpt4o.py` with any other driver file.

---

## 📊 Outputs

- Results are saved in:  
  `../results/zero_few.csv`
- Columns include:
  - `model`: Model name
  - `mode`: Zero or Few
  - `proverb`: Input proverb
  - `explanation`: Model-generated explanation
  - `reference`: Reference explanation
  - `metrics`: Evaluation metrics (Cosine, BERTScore, SAS)

---

## 📝 Notes

- All **prompts** are unified in English across models.  
- The number of examples in Few-shot can be adjusted via a parameter inside each driver.  
- `common.py` is used to reuse functions across different models.
