"""Utility helpers for simple text preprocessing."""

from __future__ import annotations

import re


def clean_text(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text
