# tradingbot/data/sentiment.py
from __future__ import annotations

from pathlib import Path
from typing import Iterable, List
from transformers import AutoTokenizer, pipeline, TFAutoModelForSequenceClassification

from tradingbot import cfg, get_logger

log = get_logger(__name__)

class SentimentAnalyzer:
    def __init__(
        self,
        model_name: str = "cardiffnlp/twitter-roberta-base-sentiment",
    ) -> None:
        self.pipe = pipeline(
            task="sentiment-analysis",
            model=model_name,
            tokenizer=model_name,
            framework="pt",       # PyTorch (lighter than TF here)
            device=-1,            # -1 → force CPU
            truncation=True,
            batch_size=32,
        )
        log.info("Sentiment pipeline initialised on CPU | model=%s", model_name)

    # ------------------------------------------------------------------ #
    # public
    # ------------------------------------------------------------------ #

    def score(self, texts: Iterable[str]) -> List[float]:
        """Return (positive – negative) sentiment score for each text."""
        # Tokenise → TF tensors
        outputs = self.pipe(list(texts))
        to_val = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}
        return [to_val[o["label"].lower()] * o["score"] for o in outputs]