# FastAPI endpoint for AI inference (MSc application demo)

from __future__ import annotations

import time
from typing import List

from fastapi import FastAPI
from pydantic import BaseModel

from model import NLPClassifier

app = FastAPI(title="NLP Research Demo API")
classifier = NLPClassifier()

# Tiny in-memory demo training.
classifier.train(
    [
        "folklore narrative oral tradition",
        "cultural story village legend",
        "computer architecture operating system",
        "data structures algorithms compiler",
    ],
    [1, 1, 0, 0],
)


class PredictRequest(BaseModel):
    texts: List[str]


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/predict")
def predict(req: PredictRequest) -> dict:
    start = time.perf_counter()
    preds = classifier.predict(req.texts)
    latency_ms = (time.perf_counter() - start) * 1000
    return {
        "predictions": preds,
        "latency_ms": round(latency_ms, 3),
        "count": len(req.texts),
    }
