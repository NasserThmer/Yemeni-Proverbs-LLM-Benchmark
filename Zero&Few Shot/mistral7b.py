"""Driver for generating proverb explanations using Mistral‑7B‑Instruct.

This module wraps the Mistral 7B Instruct v0.3 model from Hugging
Face. Prompts are created in English and require the model to
respond in Modern Standard Arabic.
"""

from __future__ import annotations

import os
from typing import Iterable, List, Optional

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login

from .common import FewShotExample, build_prompt


class Mistral7BClient:
    """Client for generating explanations with Mistral‑7B‑Instruct v0.3.

    Parameters
    ----------
    model: str, optional
        Identifier of the Mistral model. Defaults to
        ``mistralai/Mistral-7B-Instruct-v0.3``.
    hf_token: str, optional
        Hugging Face access token.
    device_map: str, optional
        Device mapping. Defaults to ``"auto"``.
    dtype: torch.dtype, optional
        Precision. Defaults to ``torch.float16``.
    """

    def __init__(
        self,
        model: str = "mistralai/Mistral-7B-Instruct-v0.3",
        hf_token: Optional[str] = None,
        device_map: str = "auto",
        dtype: torch.dtype = torch.float16,
    ) -> None:
        token = hf_token or os.getenv("HF_TOKEN")
        if token:
            try:
                login(token)
            except Exception:
                pass
        self.model_name = model
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForCausalLM.from_pretrained(
            model,
            device_map=device_map,
            torch_dtype=dtype,
        )

    def generate(
        self,
        df: pd.DataFrame,
        examples: Iterable[FewShotExample] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 256,
    ) -> List[dict]:
        """Generate explanations for each proverb using Mistral‑7B.

        See ``Allam7BClient.generate`` for parameter descriptions.
        """
        results: List[dict] = []
        k_shots = len(examples) if examples else 0
        ex_list = list(examples) if examples else []
        for idx, row in df.iterrows():
            proverb = str(row["proverb"])
            prompt = build_prompt(proverb, ex_list)
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=temperature > 0,
                    temperature=temperature or 1.0,
                    top_p=1.0,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            input_len = inputs["input_ids"].shape[-1]
            generated_ids = outputs[0][input_len:]
            prediction = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            results.append(
                {
                    "id": row.get("id", idx),
                    "proverb": proverb,
                    "model": self.model_name,
                    "prediction": prediction,
                    "prompt": prompt,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "k_shots": k_shots,
                    "task": "zero-few-shot",
                    "split": row.get("split", "test"),
                }
            )
        return results