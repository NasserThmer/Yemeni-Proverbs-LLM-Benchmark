"""Driver for generating proverb explanations using DeepSeek LLM 7B Chat.

The DeepSeek chat models follow a ChatML‑like conversational format
requiring special tokens like ``<|user|>`` and ``<|assistant|>``.
This driver wraps a standard prompt from ``build_prompt`` with those
tokens and uses the Hugging Face model to generate responses.
"""

from __future__ import annotations

import os
from typing import Iterable, List, Optional

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login

from .common import FewShotExample, build_prompt


class DeepSeek7BClient:
    """Client for generating explanations with DeepSeek LLM 7B Chat.

    Parameters
    ----------
    model: str, optional
        Model identifier. Defaults to ``deepseek-ai/deepseek-llm-7b-chat``.
    hf_token: str, optional
        Hugging Face access token.
    device_map: str, optional
        Device mapping. Defaults to ``"auto"``.
    dtype: torch.dtype, optional
        Desired precision. Defaults to ``torch.bfloat16``.
    """

    def __init__(
        self,
        model: str = "deepseek-ai/deepseek-llm-7b-chat",
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
        max_tokens: int = 256,
    ) -> List[dict]:
        """Generate explanations for each proverb using DeepSeek 7B Chat.

        See ``Allam7BClient.generate`` for parameter descriptions.
        """
        results: List[dict] = []
        k_shots = len(examples) if examples else 0
        ex_list = list(examples) if examples else []
        for idx, row in df.iterrows():
            proverb = str(row["proverb"])
            # Build base prompt and wrap in ChatML tokens
            base_prompt = build_prompt(proverb, ex_list)
            chatml_prompt = f"<|user|>\n{base_prompt}\n<|assistant|>\n"
            inputs = self.tokenizer(chatml_prompt, return_tensors="pt").to(self.model.device)
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
                    "prompt": chatml_prompt,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "k_shots": k_shots,
                    "task": "zero-few-shot",
                    "split": row.get("split", "test"),
                }
            )
        return results