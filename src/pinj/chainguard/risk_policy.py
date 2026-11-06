"""Calibrated decision policy for ChainGuard's layered signals."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .accumulator import Action
from .conversation_state import StateAssessment
from .ml_intent import MLIntentPrediction
from .provenance import ProvenanceAssessment
from .semantic_intent import (
    ExecutionMode,
    IntentAnalysis,
    IntentClass,
    TargetProvenance,
)


@dataclass(frozen=True)
class RiskPolicyConfig:
    block_threshold: float = 0.78
    restricted_threshold: float = 0.48
    ml_block: float = 0.92
    ml_high: float = 0.72
    ml_medium: float = 0.56


@dataclass(frozen=True)
class PolicyDecision:
    action: Action
    risk: float
    reasons: tuple[str, ...]
    forced: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "action": self.action.value,
            "risk": self.risk,
            "reasons": list(self.reasons),
            "forced": self.forced,
        }


class RiskPolicy:
    def __init__(self, cfg: RiskPolicyConfig | None = None):
        self.cfg = cfg or RiskPolicyConfig()

    def decide(
        self,
        *,
        semantic: IntentAnalysis,
        ml: MLIntentPrediction | None,
        state: StateAssessment,
        provenance: ProvenanceAssessment,
        legacy_score: float,
        legacy_high_confidence: bool = False,
        decoded_high_confidence: bool = False,
    ) -> PolicyDecision:
        reasons: list[str] = []

        # Explicit benign scoping is allowed unless provenance itself is unsafe.
        benign_scoped = (
            semantic.intent_class in {
                IntentClass.BENIGN_SECURITY_DISCUSSION,
                IntentClass.BENIGN_TEXT_PROCESSING,
            }
            or semantic.execution_mode in {ExecutionMode.DISCUSS, ExecutionMode.PROTECT}
            or semantic.target_provenance in {
                TargetProvenance.PUBLIC,
                TargetProvenance.USER_PROVIDED,
            }
        )

        if decoded_high_confidence and not benign_scoped:
            return PolicyDecision(Action.BLOCK, 0.99, ("decoded_attack_payload",), True)
        if legacy_high_confidence and not benign_scoped and semantic.execution_mode == ExecutionMode.EXECUTE:
            return PolicyDecision(Action.BLOCK, max(0.90, semantic.semantic_risk), ("transparent_high_confidence_match",), True)
        if provenance.high_confidence:
            return PolicyDecision(Action.BLOCK, provenance.risk, (provenance.violation or "provenance_violation",), True)
        if state.high_confidence:
            return PolicyDecision(Action.BLOCK, state.risk, state.reasons, True)
        if semantic.high_confidence and not benign_scoped:
            return PolicyDecision(Action.BLOCK, semantic.semantic_risk, (semantic.intent_class.value,), True)

        ml_risk = ml.malicious_probability if ml else 0.0
        # A very high-confidence local classifier may block independently, but
        # never overrides explicit public/user-provided, quoted, discussion, or
        # protective scope. This threshold is intentionally conservative.
        if ml and ml_risk >= self.cfg.ml_block and not benign_scoped:
            return PolicyDecision(
                Action.BLOCK,
                ml_risk,
                ("high_confidence_ml_attack", ml.label),
                False,
            )
        # A classifier alone never produces a hard block on clearly public,
        # user-provided, discussion, or protective requests.
        if benign_scoped:
            risk = max(semantic.semantic_risk, 0.15 * ml_risk, 0.10 * legacy_score)
            if provenance.risk >= self.cfg.restricted_threshold:
                reasons.append("ambiguous_untrusted_content")
                return PolicyDecision(Action.SOFT_CHALLENGE, provenance.risk, tuple(reasons), False)
            return PolicyDecision(Action.ALLOW, min(risk, 0.20), ("explicit_benign_scope",), False)

        risk = min(
            1.0,
            0.48 * semantic.semantic_risk
            + 0.27 * ml_risk
            + 0.18 * state.risk
            + 0.05 * provenance.risk
            + 0.02 * legacy_score,
        )
        if ml and ml_risk >= self.cfg.ml_high and semantic.semantic_risk >= 0.55:
            risk = max(risk, 0.84)
            reasons.append("semantic_ml_agreement")
        elif ml and ml_risk >= self.cfg.ml_medium:
            reasons.append("ml_attack_signal")
        if semantic.semantic_risk >= 0.55:
            reasons.append("semantic_risk")
        if state.risk >= 0.45:
            reasons.append("cross_turn_risk")

        if risk >= self.cfg.block_threshold:
            return PolicyDecision(Action.BLOCK, risk, tuple(reasons or ["ensemble_threshold"]), False)
        if risk >= self.cfg.restricted_threshold:
            return PolicyDecision(Action.SOFT_CHALLENGE, risk, tuple(reasons or ["ambiguous_risk"]), False)
        return PolicyDecision(Action.ALLOW, risk, tuple(reasons or ["low_risk"]), False)
