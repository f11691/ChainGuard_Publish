"""Output-side leakage guard for protected text and canaries."""
from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
import re
from typing import Any

from .semantic_intent import normalise_text


@dataclass(frozen=True)
class OutputAssessment:
    blocked: bool
    risk: float
    reasons: tuple[str, ...]
    redacted_output: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "blocked": self.blocked,
            "risk": self.risk,
            "reasons": list(self.reasons),
            "redacted_output": self.redacted_output,
        }


class OutputGuard:
    def __init__(
        self,
        *,
        canaries: list[str] | None = None,
        protected_texts: list[str] | None = None,
        ngram_size: int = 5,
        overlap_threshold: float = 0.42,
        fuzzy_threshold: float = 0.82,
    ):
        self.canaries = [normalise_text(v) for v in (canaries or []) if v]
        self.protected_texts = [normalise_text(v) for v in (protected_texts or []) if v]
        self.ngram_size = ngram_size
        self.overlap_threshold = overlap_threshold
        self.fuzzy_threshold = fuzzy_threshold

    @staticmethod
    def _ngrams(text: str, n: int) -> set[tuple[str, ...]]:
        tokens = re.findall(r"[a-z0-9_-]+", text)
        return {tuple(tokens[i:i+n]) for i in range(max(0, len(tokens)-n+1))}

    def inspect(self, output: str) -> OutputAssessment:
        norm = normalise_text(output)
        reasons: list[str] = []
        risk = 0.0
        for canary in self.canaries:
            if canary and canary in norm:
                reasons.append("exact_canary_match")
                risk = 1.0
        out_ngrams = self._ngrams(norm, self.ngram_size)
        for protected in self.protected_texts:
            if len(protected) >= 12 and protected in norm:
                reasons.append("protected_substring_match")
                risk = max(risk, 1.0)
                continue
            p_ngrams = self._ngrams(protected, self.ngram_size)
            if p_ngrams:
                overlap = len(out_ngrams & p_ngrams) / len(p_ngrams)
                if overlap >= self.overlap_threshold:
                    reasons.append(f"protected_ngram_overlap:{overlap:.3f}")
                    risk = max(risk, min(0.99, 0.60 + overlap * 0.40))
            ratio = SequenceMatcher(None, norm, protected).ratio()
            if ratio >= self.fuzzy_threshold:
                reasons.append(f"protected_fuzzy_match:{ratio:.3f}")
                risk = max(risk, ratio)
        blocked = risk >= 0.78
        redacted = "[BLOCKED: protected information detected in model output]" if blocked else None
        return OutputAssessment(blocked, risk, tuple(reasons), redacted)
