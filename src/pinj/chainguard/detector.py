"""Layered ChainGuard session detector."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .accumulator import AccumulatorConfig, IntentAccumulator, Action
from .conversation_state import SecurityConversationState
from .ml_intent import SklearnIntentClassifier
from .provenance import PromptSegment, PromptSource, ProvenancePolicy
from .risk_policy import RiskPolicy
from .semantic_intent import SemanticIntentAnalyzer
from .signals import PerTurnSignal, DriftSignal, FrameTracker


_DEFAULT_SENSITIVE = [
    "disclose protected system or developer instructions",
    "override higher-priority application policy",
    "extract hidden runtime configuration or secrets",
]


@dataclass(frozen=True)
class TurnVerdict:
    turn_index: int
    c_t: float
    d_t: float
    r_t: float
    evidence: float
    escalation_bonus: float
    score: float
    action: Action
    details: dict[str, Any] = field(default_factory=dict)


class ChainGuard:
    """Maintain bounded lexical, semantic, provenance, and conversation state."""

    def __init__(
        self,
        per_turn: PerTurnSignal | None = None,
        drift: DriftSignal | None = None,
        frames: FrameTracker | None = None,
        cfg: AccumulatorConfig | None = None,
        *,
        semantic: SemanticIntentAnalyzer | None = None,
        conversation_state: SecurityConversationState | None = None,
        provenance_policy: ProvenancePolicy | None = None,
        risk_policy: RiskPolicy | None = None,
        intent_classifier: SklearnIntentClassifier | None = None,
        use_ml: bool = True,
    ):
        self.per_turn = per_turn or PerTurnSignal()
        self.drift = drift or DriftSignal(_DEFAULT_SENSITIVE, backend="lexical")
        self.frames = frames or FrameTracker()
        self.acc = IntentAccumulator(cfg=cfg or AccumulatorConfig())
        self.semantic = semantic or SemanticIntentAnalyzer()
        self.security_state = conversation_state or SecurityConversationState()
        self.provenance_policy = provenance_policy or ProvenancePolicy()
        self.risk_policy = risk_policy or RiskPolicy()
        self.intent_classifier = intent_classifier or (SklearnIntentClassifier() if use_ml else None)
        self._t = 0

    @staticmethod
    def _coerce_segment(turn: str | PromptSegment, source: PromptSource | str | None) -> PromptSegment:
        if isinstance(turn, PromptSegment):
            return turn
        resolved = source or PromptSource.USER
        if isinstance(resolved, str):
            resolved = PromptSource(resolved)
        return PromptSegment(text=str(turn), source=resolved)

    def step(
        self,
        turn: str | PromptSegment,
        *,
        source: PromptSource | str | None = None,
    ) -> TurnVerdict:
        self._t += 1
        segment = self._coerce_segment(turn, source)
        text = segment.text

        # Legacy transparent signals are retained as low-weight ensemble inputs.
        c_t = self.per_turn(text)
        d_t = self.drift(text)
        r_t = self.frames.update_and_score(text)
        accumulated_score = self.acc.update(c_t, d_t, r_t)

        context_summary = self.security_state.context_summary()
        semantic = self.semantic.analyse(text, source=segment.source.value)
        ml_prediction = None
        ml_error = None
        if self.intent_classifier is not None:
            try:
                ml_prediction = self.intent_classifier.predict(text, context=context_summary)
            except Exception as exc:  # fail open to remaining independent layers
                ml_error = f"{type(exc).__name__}: {exc}"
        state = self.security_state.update(text, semantic)
        provenance = self.provenance_policy.assess(segment, semantic)
        legacy_details = dict(getattr(self.per_turn, "details", {}))
        decoded_semantic = []
        for item in legacy_details.get("decoded_candidates", []):
            decoded_analysis = self.semantic.analyse(str(item.get("decoded", "")), source=segment.source.value)
            decoded_semantic.append(decoded_analysis.as_dict())
        decoded_high = any(
            float(item.get("score", 0.0)) >= 0.75
            for item in legacy_details.get("decoded_candidates", [])
        ) or any(item["high_confidence"] or item["semantic_risk"] >= 0.76 for item in decoded_semantic)
        legacy_high = bool(legacy_details.get("high_confidence")) or bool(getattr(self.frames, "details", {}).get("high_confidence"))
        decision = self.risk_policy.decide(
            semantic=semantic,
            ml=ml_prediction,
            state=state,
            provenance=provenance,
            legacy_score=accumulated_score,
            legacy_high_confidence=legacy_high,
            decoded_high_confidence=decoded_high,
        )

        return TurnVerdict(
            turn_index=self._t,
            c_t=c_t,
            d_t=d_t,
            r_t=r_t,
            evidence=self.acc.last_evidence,
            escalation_bonus=self.acc.last_bonus,
            score=decision.risk,
            action=decision.action,
            details={
                "semantic": semantic.as_dict(),
                "ml_intent": ml_prediction.as_dict() if ml_prediction else None,
                "ml_error": ml_error,
                "decoded_semantic": decoded_semantic,
                "conversation_state": state.as_dict(),
                "provenance": provenance.as_dict(),
                "per_turn": dict(getattr(self.per_turn, "details", {})),
                "drift": dict(getattr(self.drift, "details", {})),
                "frame": dict(getattr(self.frames, "details", {})),
                "legacy": {
                    "per_turn": dict(getattr(self.per_turn, "details", {})),
                    "drift": dict(getattr(self.drift, "details", {})),
                    "frame": dict(getattr(self.frames, "details", {})),
                    "accumulated_score": accumulated_score,
                },
                "decision": {**decision.as_dict(), "forced_block": decision.forced},
            },
        )

    def reset(self) -> None:
        self.acc.reset()
        self.security_state.reset()
        for component in (self.per_turn, self.drift, self.frames):
            reset = getattr(component, "reset", None)
            if callable(reset):
                reset()
        self._t = 0
