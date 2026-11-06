"""Small local intent classifier used as one ensemble signal.

The classifier is deliberately transparent about its limitations: it is trained
on a bundled development corpus and must not be treated as an external model or
independent evidence of generalization.
"""
from __future__ import annotations

from dataclasses import dataclass
from importlib import resources
import json
from typing import Any


@dataclass(frozen=True)
class MLIntentPrediction:
    label: str
    confidence: float
    malicious_probability: float
    probabilities: dict[str, float]

    def as_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "confidence": self.confidence,
            "malicious_probability": self.malicious_probability,
            "probabilities": self.probabilities,
        }


class SklearnIntentClassifier:
    _GLOBAL_PIPELINE = None
    """Word+character n-gram logistic-regression intent classifier."""

    MALICIOUS_LABELS = {
        "protected_disclosure", "instruction_override", "indirect_instruction"
    }

    def __init__(self, training_rows: list[dict[str, str]] | None = None):
        self._training_rows = training_rows
        self._pipeline = None

    @staticmethod
    def _load_rows() -> list[dict[str, str]]:
        root = resources.files("pinj.chainguard.resources")
        rows = json.loads(root.joinpath("intent_training.json").read_text(encoding="utf-8"))
        external = root.joinpath("zachz_training.json")
        if external.is_file():
            rows.extend(json.loads(external.read_text(encoding="utf-8")))
        return rows

    def _ensure(self) -> None:
        if self._pipeline is not None:
            return
        if self.__class__._GLOBAL_PIPELINE is not None:
            self._pipeline = self.__class__._GLOBAL_PIPELINE
            return
        try:
            from sklearn.pipeline import FeatureUnion, Pipeline
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.linear_model import LogisticRegression
        except ImportError as exc:  # pragma: no cover - optional dependency path
            raise RuntimeError(
                "scikit-learn is required for the local intent classifier; "
                "install pinj[ml] or disable use_ml"
            ) from exc

        rows = self._training_rows or self._load_rows()
        texts = [str(row["text"]) for row in rows]
        labels = [str(row["label"]) for row in rows]
        features = FeatureUnion([
            ("word", TfidfVectorizer(
                lowercase=True, strip_accents="unicode", ngram_range=(1, 2),
                min_df=1, max_df=0.995, sublinear_tf=True,
            )),
            ("char", TfidfVectorizer(
                analyzer="char_wb", lowercase=True, strip_accents="unicode",
                ngram_range=(3, 5), min_df=1, max_features=40000,
                sublinear_tf=True,
            )),
        ])
        model = LogisticRegression(
            max_iter=2500, class_weight="balanced", C=3.0,
            solver="lbfgs", random_state=1701,
        )
        self._pipeline = Pipeline([("features", features), ("model", model)])
        self._pipeline.fit(texts, labels)
        self.__class__._GLOBAL_PIPELINE = self._pipeline

    def predict(self, text: str, *, context: str = "") -> MLIntentPrediction:
        self._ensure()
        combined = f"{context}\nCURRENT: {text}" if context else text
        probabilities = self._pipeline.predict_proba([combined])[0]
        classes = [str(v) for v in self._pipeline.classes_]
        by_label = {label: float(prob) for label, prob in zip(classes, probabilities)}
        label = max(by_label, key=by_label.get)
        malicious = sum(by_label.get(name, 0.0) for name in self.MALICIOUS_LABELS)
        return MLIntentPrediction(
            label=label,
            confidence=by_label[label],
            malicious_probability=min(1.0, malicious),
            probabilities=by_label,
        )
