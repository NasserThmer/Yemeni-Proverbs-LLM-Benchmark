"""Driver for generating proverb explanations using OpenAI's GPT‑4o model.

This module encapsulates calls to the OpenAI chat completions API.  It
constructs prompts in English and requests explanations in Modern
Standard Arabic.  The model name and API key are configurable via
environment variables.

Note: This driver requires the ``openai`` Python package to be
installed and a valid API key set in the ``OPENAI_API_KEY``
environment variable.  See the OpenAI documentation for details on
usage quotas and pricing.
"""

from __future__ import annotations

import os
import time
from typing import Iterable, List, Optional

import openai
import pandas as pd

from .common import FewShotExample, build_prompt


class GPT4oClient:
    """Client for interacting with OpenAI's GPT‑4o via chat completions.

    Parameters
    ----------
    model: str, optional
        The name of the GPT‑4o model to use.  Defaults to ``gpt‑4o``.
    api_key: str, optional
        API key for OpenAI.  If not provided, the ``OPENAI_API_KEY``
        environment variable is used.  Raises ``ValueError`` if no key
        is available.
    """

    def __init__(self, model: str = "gpt-4o", api_key: Optional[str] = None) -> None:
        if api_key is None:
            api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OpenAI API key must be provided via the api_key argument or OPENAI_API_KEY env var"
            )
        self.model = model
        openai.api_key = api_key

    def generate(
        self,
        df: pd.DataFrame,
        examples: Iterable[FewShotExample] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 64,
    ) -> List[dict]:
        """Generate explanations for a DataFrame of proverbs.

        Parameters
        ----------
        df: pandas.DataFrame
            Data frame containing at least a 'proverb' column.
        examples: Iterable[FewShotExample], optional
            Few‑shot examples to include in each prompt.  If ``None``,
            zero‑shot prompts are used.
        temperature: float, optional
            Sampling temperature for the model.  Lower values reduce
            randomness.  Defaults to 0.0.
        max_tokens: int, optional
            Maximum number of new tokens to generate.  Defaults to 256.

        Returns
        -------
        List[dict]
            A list of dictionaries containing the results for each
            proverb.  Each dict has keys: ``id``, ``proverb``, ``model``,
            ``prediction``, ``prompt``, ``temperature``, ``max_tokens``,
            ``k_shots``, ``task``, and ``split``.  See the README for
            details.
        """
        results: List[dict] = []
        k_shots = len(examples) if examples else 0
        # Build once to avoid re‑evaluating the iterable repeatedly
        ex_list = list(examples) if examples else []
        for idx, row in df.iterrows():
            proverb = str(row["proverb"])
            prompt = build_prompt(proverb, ex_list)
            # Create chat messages: system and user
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ]
            try:
                response = openai.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                content = response.choices[0].message.content.strip()
            except Exception as exc:
                # In case of API error, record the exception and continue
                content = f"Error: {exc}"
            results.append(
                {
                    "id": row.get("id", idx),
                    "proverb": proverb,
                    "model": self.model,
                    "prediction": content,
                    "prompt": prompt,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "k_shots": k_shots,
                    "task": "zero-few-shot",
                    "split": row.get("split", "test"),
                }
            )
            # polite delay to avoid rate limits
            time.sleep(0.2)
        return results