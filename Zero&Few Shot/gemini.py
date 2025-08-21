"""Driver for generating proverb explanations using Google Gemini models.

This driver uses the ``google-generativeai`` library to call Gemini
models. Prompts are constructed in English and instruct the model to
produce Modern Standard Arabic explanations. An API key must be
provided via the ``GEMINI_API_KEY`` or ``GOOGLE_API_KEY`` environment
variable, or explicitly when instantiating the client.

Gemini models currently enforce token limits on the combined prompt and
response. The `max_tokens` parameter applies only to the output.
"""

from __future__ import annotations

import os
import time
from typing import Iterable, List, Optional

import pandas as pd

try:
    import google.generativeai as genai
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "The google-generativeai package is required for Gemini support. "
        "Install it via `pip install google-generativeai`."
    ) from exc

from .common import FewShotExample, build_prompt


class GeminiClient:
    """Client for interacting with Google Gemini via the generative AI API.

    Parameters
    ----------
    model: str, optional
        Name of the Gemini model. Defaults to ``gemini-1.5-pro``.
    api_key: str, optional
        API key for Gemini. If not provided, the ``GEMINI_API_KEY``
        or ``GOOGLE_API_KEY`` environment variable is used. Raises
        ``ValueError`` if no key is available.
    """

    def __init__(self, model: str = "gemini-1.5-pro", api_key: Optional[str] = None) -> None:
        if api_key is None:
            api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError(
                "Gemini API key must be provided via the api_key argument or GEMINI_API_KEY/GOOGLE_API_KEY env vars"
            )
        self.model_name = model
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model)

    def generate(
        self,
        df: pd.DataFrame,
        examples: Iterable[FewShotExample] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 256,
    ) -> List[dict]:
        """Generate explanations for a DataFrame of proverbs using Gemini.

        Parameters
        ----------
        df: pandas.DataFrame
            Data frame containing at least a 'proverb' column.
        examples: Iterable[FewShotExample], optional
            Few‑shot examples to include in each prompt.
        temperature: float, optional
            Sampling temperature for the model. Defaults to 0.0.
        max_tokens: int, optional
            Maximum number of output tokens. Defaults to 256.

        Returns
        -------
        List[dict]
            Results for each proverb, keyed similarly to GPT4oClient.
        """
        results: List[dict] = []
        k_shots = len(examples) if examples else 0
        ex_list = list(examples) if examples else []
        for idx, row in df.iterrows():
            proverb = str(row["proverb"])
            prompt = build_prompt(proverb, ex_list)
            try:
                response = self.model.generate_content(
                    prompt,
                    generation_config={
                        "max_output_tokens": max_tokens,
                        "temperature": temperature,
                        "top_p": 1.0,
                    },
                )
                content = response.text.strip()
            except Exception as exc:
                content = f"Error: {exc}"
            results.append(
                {
                    "id": row.get("id", idx),
                    "proverb": proverb,
                    "model": self.model_name,
                    "prediction": content,
                    "prompt": prompt,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "k_shots": k_shots,
                    "task": "zero-few-shot",
                    "split": row.get("split", "test"),
                }
            )
            # Short delay to respect quota limits
            time.sleep(0.2)
        return results