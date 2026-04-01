"""Minimal NLP pipeline for application portfolio use.

This example uses TF-IDF + Logistic Regression for a simple text classification task.
It is intentionally lightweight and readable so it can be reviewed easily by an admissions committee.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


@dataclass
class NLPClassifier:
    model: Pipeline | None = None

    def build(self) -> Pipeline:
        self.model = Pipeline(
            steps=[
                (
                    "tfidf",
                    TfidfVectorizer(
                        lowercase=True,
                        ngram_range=(1, 2),
                        max_features=5000,
                    ),
                ),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=1000,
                        solver="liblinear",
                    ),
                ),
            ]
        )
        return self.model

    def train(self, texts: Iterable[str], labels: Iterable[int]) -> Pipeline:
        if self.model is None:
            self.build()
        assert self.model is not None
        self.model.fit(list(texts), list(labels))
        return self.model

    def predict(self, texts: List[str]) -> List[int]:
        if self.model is None:
            raise ValueError("Model is not trained.")
        return self.model.predict(texts).tolist()


if __name__ == "__main__":
    texts = [
        "Folklore stories preserve cultural memory",
        "This tale contains mythic archetypes and oral tradition",
        "Network routing and protocols are system concepts",
        "Compiler optimization improves program performance",
    ]
    labels = [1, 1, 0, 0]

    clf = NLPClassifier()
    clf.train(texts, labels)
    preds = clf.predict([
        "Traditional Bengali tales often carry moral lessons",
        "Distributed systems require consistency and fault tolerance",
    ])
    print(preds)
