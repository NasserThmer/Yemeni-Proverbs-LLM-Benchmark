# Yemeni Proverbs Dataset

This folder contains the curated dataset of Yemeni proverbs paired with expert explanations.

---

## Files

- **Train_data.csv** → 2125 samples (used for training and few-shot exemplars)
- **Validation_data.csv** → 455 samples (used for validation during fine-tuning)
- **Test_data.csv** → 456 samples (held-out evaluation set)

---

## Format

Each CSV contains two columns:

| Column        | Description                                      |
|---------------|--------------------------------------------------|
| `text`        | Yemeni proverb in dialect                        |
| `explanation` | Expert explanation in Modern Standard Arabic     |

**Example row:**

| text                        | explanation                                                                 |
|-----------------------------|-----------------------------------------------------------------------------|
| ما تحيي الميه الا بالتسعين  | يشير هذا المثل إلى أن تحقيق الأهداف يتطلب جهدًا كبيرًا ومثابرة حتى الاقتراب من النهاية. |

---

## Usage

- **Few-shot** → `Train_data.csv` is used to sample *k* examples for prompting.  
- **Fine-tuning** → `Train_data.csv` + `Validation_data.csv` used for model training/validation.  
- **Evaluation** → `Test_data.csv` is used for final evaluation across all settings.  

---

## License

- **Dataset**: Released under **CC BY 4.0** license.  
You are free to share and adapt with attribution. See [LICENSE-DATA](../LICENSE-DATA) for details.

---

## Notes

- All texts are anonymized proverbs and their explanations; no personal information is included.
- Please cite the benchmark if you use this dataset.
