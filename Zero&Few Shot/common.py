"""Common utilities for zero‑ and few‑shot generation of proverb explanations.

This module centralises data loading, few‑shot example selection and prompt
construction. It is intentionally model‑agnostic so that each driver can
compose its own messages while relying on these helpers for the heavy
lifting.

All prompts throughout this repository are written in English. We instruct
the models to answer in Modern Standard Arabic by clearly stating that
requirement in the prompt. English instructions tend to reduce spurious
translations and improve consistency across different models.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import pandas as pd


@dataclass
class FewShotExample:
    """Container for a proverb and its reference explanation.

    When performing few‑shot inference, a small set of reference examples
    helps steer the model towards concise and accurate outputs. This
    dataclass is used to represent those examples as a pair of strings.
    """

    proverb: str
    explanation: str

    def as_prompt_block(self, index: int) -> str:
        """Render this example as a numbered prompt block.

        The block includes the example index, the proverb and its
        explanation. Delimiters (e.g. `Example 1`) are avoided at the
        end of the explanation to ensure clean concatenation in larger
        prompts. Newlines are preserved for clarity.
        """
        return (
            f"Example {index}:\n"
            f"Proverb: {self.proverb}\n"
            f"Explanation: {self.explanation}\n\n"
        )


def load_data(csv_path: str) -> pd.DataFrame:
    """Load a CSV file containing proverbs and optional explanations.

    The CSV is expected to have at least a column named ``proverb``.
    Optionally a column named ``reference_explanation`` may be present;
    this column is used when selecting few‑shot examples. Any other
    columns are carried through but unused by the generation functions.

    Parameters
    ----------
    csv_path: str
        Path to the CSV file to load.

    Returns
    -------
    pandas.DataFrame
        Loaded data frame.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Could not find CSV file at {csv_path}")
    df = pd.read_csv(csv_path)
    if "proverb" not in df.columns:
        raise ValueError("Input CSV must contain a 'proverb' column")
    return df


def select_few_shots(df: pd.DataFrame, k: int, seed: int = 42) -> List[FewShotExample]:
    """Select a random subset of examples for few‑shot prompting.

    Only rows where ``reference_explanation`` is non‑empty are eligible
    for selection. The random seed ensures reproducibility.

    Parameters
    ----------
    df: pandas.DataFrame
        Data frame containing at least 'proverb' and 'reference_explanation'.
    k: int
        Number of examples to sample. If zero or fewer, an empty list
        is returned.
    seed: int
        Random seed for reproducible sampling.

    Returns
    -------
    List[FewShotExample]
        A list of sampled few‑shot examples.
    """
    if k <= 0:
        return []
    if "reference_explanation" not in df.columns:
        return []
    # Drop rows with missing or blank explanations
    eligible = df.dropna(subset=["reference_explanation"])
    eligible = eligible[eligible["reference_explanation"].astype(str).str.strip() != ""]
    if len(eligible) == 0:
        return []
    # Sample without replacement
    random.seed(seed)
    sample_indices = random.sample(list(eligible.index), min(k, len(eligible)))
    examples: List[FewShotExample] = []
    for i, idx in enumerate(sample_indices, start=1):
        row = eligible.loc[idx]
        proverb = str(row["proverb"]).strip()
        explanation = str(row["reference_explanation"]).strip()
        examples.append(FewShotExample(proverb=proverb, explanation=explanation))
    return examples


def build_prompt(proverb: str, examples: Iterable[FewShotExample] | None = None) -> str:
    """Construct a unified prompt for zero‑ or few‑shot generation.

    The prompt instructs the model (in English) to act as a linguistic
    expert on Yemeni proverbs and to provide a figurative explanation in
    Modern Standard Arabic. If examples are provided, they are inserted
    before the target proverb to serve as guidance.

    Parameters
    ----------
    proverb: str
        The proverb for which an explanation is requested.
    examples: Iterable[FewShotExample], optional
        Iterable of few‑shot examples. If ``None`` or empty, a zero‑shot
        prompt is created.

    Returns
    -------
    str
        A fully formed prompt ready for generation.
    """
    intro = (
        "You are a linguistic expert in Yemeni proverbs. "
        "Your task is to explain the figurative meaning of the proverb below "
        "clearly and concisely in Modern Standard Arabic only. "
        "Do not translate the proverb or use English.\n\n"
    )
    example_text = ""
    if examples:
        for idx, ex in enumerate(examples, start=1):
            example_text += ex.as_prompt_block(idx)
        # Append a separator for the target proverb
        example_text += "Proverb: {}\nExplanation:".format(proverb)
    else:
        example_text = f"Proverb: {proverb}\nExplanation:"
    return intro + example_text