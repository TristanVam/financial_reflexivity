from __future__ import annotations

import logging
from functools import cached_property

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

logger = logging.getLogger(__name__)


class FinBERTSentimentEngine:
    name = "finbert"

    def __init__(self, model_name: str = "ProsusAI/finbert") -> None:
        self.model_name = model_name

    @cached_property
    def tokenizer(self):
        return AutoTokenizer.from_pretrained(self.model_name)

    @cached_property
    def model(self):
        return AutoModelForSequenceClassification.from_pretrained(self.model_name)

    @torch.no_grad()
    def score(self, texts: pd.Series) -> pd.Series:
        if texts.empty:
            return pd.Series(dtype=float)

        inputs = self.tokenizer(
            texts.tolist(),
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt",
        )
        outputs = self.model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1).cpu().numpy()
        # FinBERT order: negative, neutral, positive
        labels = np.array([-1.0, 0.0, 1.0])
        scores = probs @ labels
        return pd.Series(scores, index=texts.index, name="sentiment")


__all__ = ["FinBERTSentimentEngine"]
