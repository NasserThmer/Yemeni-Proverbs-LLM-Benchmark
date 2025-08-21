"""Zero- and few‑shot inference for Yemeni proverb explanations.

This package contains a common utility module and a number of model‑specific
drivers. Each driver implements a simple interface for generating
explanations from a list of proverbs using either the OpenAI API, the
Google Generative AI API (Gemini), or open‑source large language models
hosted on Hugging Face. All prompts are specified in English to ensure
consistent behaviour across different models, but the models are asked
explicitly to produce explanations in Modern Standard Arabic.

Example usage:

```python
from zero_few_shot import gpt4o
from zero_few_shot.common import load_data, select_few_shots

df = load_data("data/Test_data.csv")
examples = select_few_shots(df, k=2)
client = gpt4o.GPT4oClient()
results = client.generate(df, examples, temperature=0.2, max_tokens=256)
```

The `run.py` script in this directory provides a unified command line
interface around these drivers. See that script for further details.
"""

# Re‑export commonly used classes and functions to simplify imports for
# callers that only need to generate explanations.
from .common import load_data, select_few_shots  # noqa: F401