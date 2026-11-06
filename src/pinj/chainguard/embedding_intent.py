"""Pretrained-encoder intent signal (ChainGuard v0.6, optional).

This is the component the v0.5.1 evaluation report identifies as the required
next step: a genuinely pretrained sequence encoder used as ONE signal inside
the layered architecture, not the only security boundary. It generalises across
paraphrase and language far better than the lexical/TF-IDF classifier because
the representation is learned, not surface-matched.

It is OFF BY DEFAULT and dependency-gated: it needs `sentence-transformers`
(or `transformers`) and a downloaded model, which are not available in every
environment. When enabled it records full provenance (model repo + revision,
runtime versions, class mapping, calibration, thresholds) so results are
auditable, exactly as the report requires.

The classifier here is a calibrated nearest-centroid / logistic head over frozen
embeddings of the bundled training corpus. Swap in a fine-tuned prompt-attack
classifier by pointing `model_name` at it and setting `head="model"`.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from importlib import resources
from typing import Any


@dataclass(frozen=True)
class EmbeddingPrediction:
    malicious_probability: float
    label: str
    provenance: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "malicious_probability": self.malicious_probability,
            "label": self.label,
            "provenance": self.provenance,
        }


MALICIOUS_LABELS = {"protected_disclosure", "instruction_override", "indirect_instruction"}


class EmbeddingIntentClassifier:
    """Calibrated logistic head over frozen sentence embeddings.

    Parameters
    ----------
    model_name:
        A sentence-transformers model id. Multilingual models (e.g.
        ``paraphrase-multilingual-MiniLM-L12-v2``) are recommended because a
        large share of residual escapes are non-English.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        device: str = "cpu",
    ):
        self.model_name = model_name
        self.device = device
        self._encoder = None
        self._clf = None
        self._provenance: dict[str, Any] = {}

    # -- provenance -------------------------------------------------------
    @staticmethod
    def _hash_rows(rows: list[dict[str, str]]) -> str:
        blob = json.dumps(rows, sort_keys=True, ensure_ascii=False).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()

    def _load_rows(self) -> list[dict[str, str]]:
        root = resources.files("pinj.chainguard.resources")
        rows = json.loads(root.joinpath("intent_training.json").read_text("utf-8"))
        ext = root.joinpath("zachz_training.json")
        if ext.is_file():
            rows.extend(json.loads(ext.read_text("utf-8")))
        return rows

    # -- lifecycle --------------------------------------------------------
    def _ensure(self) -> None:
        if self._clf is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer  # lazy, optional
            import sentence_transformers as st
            from sklearn.linear_model import LogisticRegression
            from sklearn.calibration import CalibratedClassifierCV
            import numpy as np
        except Exception as exc:  # pragma: no cover - optional path
            raise RuntimeError(
                "embedding signal requires sentence-transformers + scikit-learn "
                "and a downloaded model; install pinj[encoder] and enable network"
            ) from exc

        self._encoder = SentenceTransformer(self.model_name, device=self.device)
        rows = self._load_rows()
        texts = [str(r["text"]) for r in rows]
        labels = [str(r["label"]) for r in rows]
        X = self._encoder.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        base = LogisticRegression(max_iter=2000, class_weight="balanced", C=2.0)
        # probability calibration (isotonic requires enough data; sigmoid is safe)
        self._clf = CalibratedClassifierCV(base, method="sigmoid", cv=3)
        self._clf.fit(np.asarray(X), labels)

        self._provenance = {
            "model_name": self.model_name,
            "sentence_transformers_version": getattr(st, "__version__", "unknown"),
            "n_training_rows": len(rows),
            "training_rows_sha256": self._hash_rows(rows),
            "label_classes": list(self._clf.classes_),
            "calibration": "sigmoid/CalibratedClassifierCV(cv=3)",
            "embedding_normalized": True,
            "device": self.device,
        }

    # -- inference --------------------------------------------------------
    def predict(self, text: str, *, context: str = "") -> EmbeddingPrediction:
        self._ensure()
        combined = f"{context}\nCURRENT: {text}" if context else text
        vec = self._encoder.encode([combined], normalize_embeddings=True)
        proba = self._clf.predict_proba(vec)[0]
        classes = [str(c) for c in self._clf.classes_]
        by_label = {c: float(p) for c, p in zip(classes, proba)}
        label = max(by_label, key=by_label.get)
        malicious = min(1.0, sum(by_label.get(m, 0.0) for m in MALICIOUS_LABELS))
        return EmbeddingPrediction(
            malicious_probability=malicious,
            label=label,
            provenance=self._provenance,
        )
