"""Driver for generating proverb explanations using the ALLaM‑7B model.

This driver loads the ALLaM‑7B Instruct model from Hugging Face and
performs zero‑ or few‑shot generation via the ``generate`` method.  It
relies on English prompts but instructs the model to produce its
responses in Modern Standard Arabic.

Note: You may need to set the ``HF_TOKEN`` environment variable to
authenticate with Hugging Face if the model requires gated access.
"""

from __future__ import annotations

import os
from typing import Iterable, List, Optional

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login

from .common import FewShotExample, build_prompt


class Allam7BClient:
    """Client for generating explanations with ALLaM‑7B Instruct preview.

    Parameters
    ----------
    model: str, optional
        Model identifier on Hugging Face.  Defaults to
        ``ALLaM-AI/ALLaM-7B-Instruct-preview``.
    hf_token: str, optional
        Hugging Face access token.  If provided (or if the ``HF_TOKEN``
        environment variable is set), the client will call
        ``huggingface_hub.login`` before loading the model.
    device_map: str, optional
        Device mapping passed to ``from_pretrained``.  Defaults to
        ``"auto"``.
    dtype: torch.dtype, optional
        Desired model precision.  Defaults to ``torch.float16``.
    """

    def __init__(
        self,
        model: str = "ALLaM-AI/ALLaM-7B-Instruct-preview",
        hf_token: Optional[str] = None,
        device_map: str = "auto",
        dtype: torch.dtype = torch.float16,
    ) -> None:
        token = hf_token or os.getenv("HF_TOKEN")
        if token:
            try:
                login(token)
            except Exception:
                # Ignore login failures; they will surface during model load if critical
                pass
        self.model_name = model
        self.tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model,
            device_map=device_map,
            torch_dtype=dtype,
            trust_remote_code=True,
        )

    def generate(
        self,
        df: pd.DataFrame,
        examples: Iterable[FewShotExample] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 64,
    ) -> List[dict]:
        """Generate explanations for each proverb using ALLaM‑7B.

        Parameters
        ----------
        df: pandas.DataFrame
            Data frame with a 'proverb' column.
        examples: Iterable[FewShotExample], optional
            Few‑shot examples.  If provided, they are inserted into the
            prompt ahead of the target proverb.
        temperature: float, optional
            Sampling temperature.  Defaults to 0.0 (deterministic).
        max_tokens: int, optional
            Maximum number of new tokens to generate.  Defaults to 256.

        Returns
        -------
        List[dict]
            Generation results in a consistent format.
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
            # Slice off the prompt tokens to isolate the generated portion
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