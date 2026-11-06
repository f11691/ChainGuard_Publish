"""ChainGuard v0.6: additive semantic-generalisation layer over v0.5.

Composition (conservative, precision-preserving):

1. Run the v0.5 layered detector. If it blocks, block.
2. Otherwise, decode the turn (multi-encoding) and score both the raw turn and
   any decoded plaintext with the compositional ConceptScorer.
3. Maintain a leaky cross-turn "protected-focus" accumulator so a goal that is
   built up across several benign-looking turns and cashed in with a late
   execution cue is caught (the chained-attack case).
4. Escalate to BLOCK only when concept/cross-turn evidence is high AND the base
   semantic layer did not mark the turn benign-scoped (discussion, example,
   protective, public, user-provided). This gate is what keeps false positives
   from rising.

Every escalation records its reason for audit. The layer never *downgrades* a
base block, only adds new detections.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .accumulator import Action
from .concept_disclosure import ConceptScorer, ConceptScore
from .decoders import decode_candidates
from .detector import ChainGuard, TurnVerdict


@dataclass
class V6Config:
    concept_block: float = 0.72        # single-turn concept risk -> block
    concept_challenge: float = 0.50    # -> soft challenge
    cross_turn_lambda: float = 0.70    # leaky memory for protected focus
    cross_turn_block: float = 0.95     # accumulated focus -> block
    decoded_block: float = 0.70        # decoded-plaintext concept risk -> block


@dataclass
class ChainGuardV6:
    base: ChainGuard = field(default_factory=ChainGuard)
    scorer: ConceptScorer = field(default_factory=ConceptScorer)
    cfg: V6Config = field(default_factory=V6Config)
    embedding: Any = None            # optional EmbeddingIntentClassifier
    embedding_block: float = 0.90    # calibrated P(malicious) -> block
    _focus: float = 0.0

    # --- helpers ---------------------------------------------------------
    def _base_benign_scoped(self, verdict: TurnVerdict) -> bool:
        sem = (verdict.details or {}).get("semantic", {})
        return (
            sem.get("intent_class") in {"benign_security_discussion", "benign_text_processing"}
            or sem.get("execution_mode") in {"security_discussion", "protective_request"}
            or sem.get("target_provenance") in {"public_or_attached", "user_provided"}
        )

    def _best_decoded_score(self, text: str) -> tuple[float, dict[str, Any] | None]:
        best = 0.0
        best_ev = None
        for cand in decode_candidates(text):
            cs = self.scorer.score(cand.text)
            if cs.risk > best:
                best, best_ev = cs.risk, {"method": cand.method, **cs.as_dict()}
        return best, best_ev

    # --- main ------------------------------------------------------------
    def step(self, turn: str, **kw) -> TurnVerdict:
        verdict = self.base.step(turn, **kw)
        # 1. base block wins
        if verdict.action == Action.BLOCK:
            return verdict

        benign_scoped = self._base_benign_scoped(verdict)
        concept: ConceptScore = self.scorer.score(turn)
        decoded_risk, decoded_ev = self._best_decoded_score(turn)

        # 2. leaky cross-turn protected-focus accumulator
        focus_input = concept.risk if concept.intent in {
            "protected_disclosure", "protected_reference",
            "instruction_override", "exfiltration",
        } else 0.0
        self._focus = self.cfg.cross_turn_lambda * self._focus + focus_input
        cross_turn_hit = (
            self._focus >= self.cfg.cross_turn_block
            and concept.executable
            and not benign_scoped
        )

        # 3. decide escalation (never downgrade)
        # optional pretrained-encoder signal (the report's ceiling-breaker)
        emb_risk = 0.0
        emb_ev = None
        if self.embedding is not None:
            try:
                pred = self.embedding.predict(turn)
                emb_risk = pred.malicious_probability
                emb_ev = pred.as_dict()
            except Exception as exc:  # fail open to lexical layers
                emb_ev = {"error": f"{type(exc).__name__}: {exc}"}

        new_action = verdict.action
        reason = None
        risk = verdict.score
        if not benign_scoped:
            if emb_risk >= self.embedding_block:
                new_action, reason, risk = Action.BLOCK, "embedding_attack", max(risk, emb_risk)
            elif decoded_risk >= self.cfg.decoded_block:
                new_action, reason, risk = Action.BLOCK, "decoded_concept_attack", max(risk, decoded_risk)
            elif concept.risk >= self.cfg.concept_block and concept.executable:
                new_action, reason, risk = Action.BLOCK, f"concept_{concept.intent}", max(risk, concept.risk)
            elif cross_turn_hit:
                new_action, reason, risk = Action.BLOCK, "cross_turn_protected_focus", max(risk, self._focus)
            elif concept.risk >= self.cfg.concept_challenge and verdict.action == Action.ALLOW:
                new_action, reason, risk = Action.SOFT_CHALLENGE, f"concept_ambiguous_{concept.intent}", max(risk, concept.risk)

        if new_action == verdict.action and reason is None:
            return verdict

        details = dict(verdict.details or {})
        details["v6"] = {
            "concept": concept.as_dict(),
            "decoded": decoded_ev,
            "embedding": emb_ev,
            "cross_turn_focus": self._focus,
            "benign_scoped": benign_scoped,
            "escalation_reason": reason,
        }
        return TurnVerdict(
            turn_index=verdict.turn_index,
            c_t=verdict.c_t, d_t=verdict.d_t, r_t=verdict.r_t,
            evidence=verdict.evidence, escalation_bonus=verdict.escalation_bonus,
            score=risk, action=new_action, details=details,
        )

    def reset(self) -> None:
        self.base.reset()
        self._focus = 0.0
