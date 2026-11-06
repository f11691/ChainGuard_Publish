"""Model-backed prompt-attack classifier proxies.

The original prototype inferred the positive class from ``LABEL_1`` and
inverted every other label.  That is unsafe for models such as Llama Prompt
Guard 2, whose documented labels are ``BENIGN`` and ``MALICIOUS``.  This
implementation inspects all returned class scores and maps labels explicitly.
"""
from __future__ import annotations

from collections.abc import Iterable

from .base import Proxy, ProxyResult, Decision


DEFAULT_POSITIVE_LABELS = {
    "malicious", "injection", "jailbreak", "unsafe", "attack", "label_1", "1"
}
DEFAULT_NEGATIVE_LABELS = {"benign", "safe", "label_0", "0"}


def normalise_label(label: str) -> str:
    return str(label).strip().casefold().replace("-", "_").replace(" ", "_")


def positive_probability(
    scores: Iterable[dict],
    positive_labels: set[str] | None = None,
    negative_labels: set[str] | None = None,
) -> float:
    """Map a full classifier score vector to P(attack).

    Raises a clear error for unknown label schemes rather than silently
    inverting a score and producing a plausible but wrong probability.
    """
    positives = {normalise_label(v) for v in (positive_labels or DEFAULT_POSITIVE_LABELS)}
    negatives = {normalise_label(v) for v in (negative_labels or DEFAULT_NEGATIVE_LABELS)}
    rows = list(scores)
    if not rows:
        raise ValueError("classifier returned no scores")

    positive_total = 0.0
    known = False
    unknown: list[str] = []
    for row in rows:
        label = normalise_label(row["label"])
        score = float(row["score"])
        if label in positives:
            positive_total += score
            known = True
        elif label in negatives:
            known = True
        else:
            unknown.append(label)
    if not known or unknown:
        raise ValueError(
            "unknown classifier labels: " + ", ".join(sorted(set(unknown or [normalise_label(r['label']) for r in rows])))
            + "; configure positive_labels/negative_labels explicitly"
        )
    return max(0.0, min(1.0, positive_total))


class ClassifierGuardProxy(Proxy):
    name = "classifier_guard"

    def __init__(
        self,
        model_name: str = "protectai/deberta-v3-base-prompt-injection-v2",
        threshold: float = 0.5,
        device: str = "cpu",
        positive_labels: set[str] | None = None,
        negative_labels: set[str] | None = None,
    ):
        self.model_name = model_name
        self.threshold = threshold
        self.device = device
        self.positive_labels = positive_labels
        self.negative_labels = negative_labels
        self._pipe = None

    def _ensure(self):
        if self._pipe is None:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            self._pipe = pipeline(
                "text-classification",
                model=model,
                tokenizer=tokenizer,
                truncation=True,
                max_length=512,
                device=-1 if self.device == "cpu" else 0,
                top_k=None,
            )
        return self._pipe

    def injection_probability(self, prompt: str) -> float:
        output = self._ensure()(prompt)
        # Depending on transformers version, top_k=None returns either
        # [{label, score}, ...] or [[{label, score}, ...]].
        if output and isinstance(output[0], list):
            output = output[0]
        return positive_probability(output, self.positive_labels, self.negative_labels)

    def _run(self, prompt: str) -> ProxyResult:
        probability = self.injection_probability(prompt)
        decision = Decision.BLOCK if probability >= self.threshold else Decision.ALLOW
        return ProxyResult(decision, prompt, 0.0, {"p_attack": probability})


class LlamaPromptGuard2Proxy(ClassifierGuardProxy):
    name = "llama_prompt_guard_2"

    def __init__(self, threshold: float = 0.5, device: str = "cpu"):
        super().__init__(
            model_name="meta-llama/Llama-Prompt-Guard-2-86M",
            threshold=threshold,
            device=device,
            positive_labels={"malicious"},
            negative_labels={"benign"},
        )


# Historical alias retained for callers of the original package.
LlamaGuardProxy = LlamaPromptGuard2Proxy
