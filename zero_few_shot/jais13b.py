"""Driver for generating proverb explanations using Jais models.

This driver supports the 7B Jais Chat model hosted on Hugging Face.
Prompts are constructed in English to instruct the model to produce
Modern Standard Arabic explanations.

If you wish to use a larger Jais model (e.g. 13B), set the model
identifier accordingly when instantiating the client.
"""

from __future__ import annotations

import os
from typing import Iterable, List, Optional

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login

from .common import FewShotExample, build_prompt


class JaisClient:
    """Client for generating explanations with Jais chat models.

    Parameters
    ----------
    model: str, optional
        Hugging Face identifier for the Jais model.  Defaults to
        ``inceptionai/jais-adapted-7b-chat``.
    hf_token: str, optional
        Hugging Face token for gated models.
    device_map: str, optional
        Device mapping.  Defaults to ``"auto"``.
    dtype: torch.dtype, optional
        Desired precision.  Defaults to ``torch.float16``.
    """

    def __init__(
        self,
        model: str = "inceptionai/jais-adapted-7b-chat",
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
        max_tokens: int = 64,
    ) -> List[dict]:
        """Generate explanations for each proverb using a Jais chat model.

        See ``Allam7BClient.generate`` for parameter explanations.
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