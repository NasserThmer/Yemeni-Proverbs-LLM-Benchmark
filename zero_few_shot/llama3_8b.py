"""Driver for generating proverb explanations using Meta‑Llama‑3‑8B.

This driver loads the Meta‑Llama‑3‑8B‑Instruct model from Hugging Face.
It constructs prompts in English, instructs the model to answer in
Modern Standard Arabic and returns the generated explanations.  Access
to this model may require a Hugging Face token; set the ``HF_TOKEN``
environment variable accordingly.
"""

from __future__ import annotations

import os
from typing import Iterable, List, Optional

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login

from .common import FewShotExample, build_prompt


class LLaMA3Client:
    """Client for generating explanations with Meta‑Llama‑3‑8B‑Instruct.

    Parameters
    ----------
    model: str, optional
        Model identifier.  Defaults to ``meta-llama/Meta-Llama-3-8B-Instruct``.
    hf_token: str, optional
        Hugging Face access token.  Uses ``HF_TOKEN`` environment variable if
        omitted.
    device_map: str, optional
        Device mapping for loading the model.  Defaults to ``"auto"``.
    dtype: torch.dtype, optional
        Desired model precision.  Defaults to ``torch.bfloat16``.
    """

    def __init__(
        self,
        model: str = "meta-llama/Meta-Llama-3-8B-Instruct",
        hf_token: Optional[str] = None,
        device_map: str = "auto",
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        token = hf_token or os.getenv("HF_TOKEN")
        if token:
            try:
                login(token)
            except Exception:
                pass
        self.model_name = model
        self.tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model,
            torch_dtype=dtype,
            device_map=device_map,
            trust_remote_code=True,
        )

    def generate(
        self,
        df: pd.DataFrame,
        examples: Iterable[FewShotExample] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 64,
    ) -> List[dict]:
        """Generate explanations for each proverb using LLaMA‑3‑8B.

        See ``Allam7BClient.generate`` for parameter definitions.
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