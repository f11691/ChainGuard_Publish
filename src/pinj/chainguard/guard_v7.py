"""ChainGuard v0.7: calibrated ensemble and execution-aware decoding.

v0.7 keeps the v0.6 detector intact and adds four narrowly scoped layers:

* a classifier escalation threshold calibrated on benign hard negatives;
* execution-aware handling of decoded directives;
* explicit exfiltration/control-token policies;
* bounded cross-turn protected-target and untrusted-instruction state.

The implementation is still a detector, not a proof of security.  Every new
hard block records the signal that caused it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any

from .accumulator import Action
from .decoders import decode_candidates
from .detector import TurnVerdict
from .guard_v6 import ChainGuardV6
from .semantic_intent import normalise_text


_MALICIOUS_LABELS = {
    "protected_disclosure", "instruction_override", "indirect_instruction"
}

_EXECUTE_DECODED = re.compile(
    r"\b(?:obey|follow|execute|perform|carry out|act on|apply|run|comply with)\b"
    r".{0,45}\b(?:result|decoded|command|instruction|order|message|text|it)\b"
    r"|\b(?:as|like) (?:a )?(?:command|instruction)\b",
    re.I,
)
_DECODED_DIRECTIVE = re.compile(
    r"\b(?:ignore|disregard|override|bypass|disable|reveal|disclose|exfiltrate|"
    r"leak|dump|execute|obey|follow)\b",
    re.I,
)

_TRANSFER = re.compile(
    r"\b(?:leak|upload|mirror|send|forward|post|transmit|stream|exfiltrate|"
    r"beacon|copy|relay)\b",
    re.I,
)
_CHANNEL = re.compile(
    r"\b(?:tracking pixel|websocket|web socket|webhook|endpoint|server|host|"
    r"remote url|external url|external site|email|smtp|ftp|dns|query string|"
    r"subdomain|collector)\b",
    re.I,
)
_SENSITIVE = re.compile(
    r"\b(?:current chat|conversation|chat history|transcript|subsequent prompts?|"
    r"future prompts?|private details?|personal data|user data|private memory|"
    r"memory|secrets?|credentials?|api keys?|tokens?|cookies?)\b",
    re.I,
)

_CONTROL_TOKEN = re.compile(r"<\|[^|\n]{2,80}\|>|\[(?:system|policy|admin|root)[^\]]{0,60}\]", re.I)
_CONTROL_DISABLE = re.compile(
    r"\b(?:policy|safety|guardrails?|moderation|filters?|restrictions?)\b.{0,35}"
    r"\b(?:disabled|off|clear(?:ed)?|removed|none|null|bypassed)\b",
    re.I,
)
_RAW_MODE = re.compile(r"\b(?:raw|unfiltered|unmoderated|unrestricted|uncensored)\s+(?:mode|persona|assistant)\b", re.I)
_NEGATED_SAFETY = re.compile(r"\bno\s+(?:moderation|warnings?|refusals?|filters?|safety|guardrails?)\b", re.I)
_RAW_NEGATED_BEHAVIOR = re.compile(
    r"\b(?:never|without)\b.{0,70}\b(?:warns?|warnings?|filters?|refus(?:e|es|als?)|moderation|guardrails?)\b",
    re.I,
)

_PROTECTED_REFERENCE = re.compile(
    r"\b(?:system|developer|administrator|host|platform)\s+(?:prompt|message|instructions?|rules?|policy|configuration|context)\b"
    r"|\b(?:hidden|private|internal|secret|unseen|off[- ]screen|pre[- ]user|startup|initiali[sz]ation)\s+"
    r"(?:content|text|configuration|context|rules?|instructions?|directives?|policy|message)\b"
    r"|\b(?:before|prior to|preceding)\s+(?:this|the)\s+(?:chat|conversation|dialogue)\b"
    r"|\b(?:geheime|verborgene|interne)\s+(?:regeln|anweisungen|vorgaben|konfiguration)\b"
    r"|\b(?:règles|consignes|instructions)\s+(?:privées|cachées|internes)\b"
    r"|\b(?:reglas|instrucciones|directivas)\s+(?:privadas|ocultas|internas)\b",
    re.I,
)
_UNTRUSTED_INSTRUCTION = re.compile(
    r"\b(?:retrieved|web|external|rag|tool|email|document)\s+(?:page|document|content|output|message)?\b"
    r".{0,65}\b(?:imperative|instruction|command|request|directive)\b",
    re.I,
)
_ALIAS_PATTERNS = (
    re.compile(r"\b(?:call|name|label|refer to)\s+(?:it|that|this|them|the (?:text|content|configuration|rules?|instructions?))\s+(?:as\s+)?([\wÀ-ÖØ-öø-ÿ-]{2,30})\b", re.I),
    re.compile(r"\b(?:be|is|are)\s+called\s+([\wÀ-ÖØ-öø-ÿ-]{2,30})\b", re.I),
    re.compile(r"\b(?:nenn(?:e)?|bezeichne)\s+(?:sie|es|das)\s+([\wÀ-ÖØ-öø-ÿ-]{2,30})\b", re.I),
    re.compile(r"\bappelle(?:-les|-la|-le)?\s+([\wÀ-ÖØ-öø-ÿ-]{2,30})\b", re.I),
    re.compile(r"\bll[aá]ma(?:les|lo|la)?\s+([\wÀ-ÖØ-öø-ÿ-]{2,30})\b", re.I),
)
_DISCLOSURE_EXECUTE = re.compile(
    r"\b(?:show|reveal|print|output|return|provide|repeat|quote|recite|render|"
    r"translate|convert|serialize|export|reproduce|write out|display|"
    r"gib|zeige|drucke|übersetze|traduis|montre|affiche|muestra|traduce)\b",
    re.I,
)
_FOLLOW_UNTRUSTED = re.compile(
    r"\b(?:follow|execute|perform|obey|carry out|apply|comply with|do exactly what)\b.{0,55}"
    r"\b(?:imperative|instruction|command|request|directive|document|page|it)\b",
    re.I,
)
_BENIGN_UNTRUSTED_ANALYSIS = re.compile(
    r"\b(?:classify|analy[sz]e|describe|explain|flag|mark|treat|quote|summari[sz]e)\b"
    r".{0,70}\b(?:untrusted|attack|injection|command|instruction)\b",
    re.I,
)

# In a strict RAG/tool boundary, retrieved segments are data only.  A segment
# that directly addresses the assistant, asks a question, or requests a change
# to the answer is instruction-bearing even when it contains no classic
# "ignore previous" phrase.  This is deliberately source-gated: the same text
# remains ordinary user input when source="user".
_UNTRUSTED_DIRECTIVE_START = re.compile(
    r"^\s*(?:please\s+)?(?:write|draft|answer|tell|provide|show|list|analy[sz]e|determine|classify|"
    r"recommend|help|suggest|summari[sz]e|describe|explain|encrypt|apply|use|"
    r"render|shift|encode|decode|reply|respond|replace|express|translate|add|"
    r"integrate|include|modify|augment|enhance|mention|develop|create|compose|"
    r"ensure|incorporate|introduce|employ|utili[sz]e|append|output|return)\b"
    r"|^\s*(?:what|which|how|why|where|when|who|can you|could you|would you|"
    r"will you|do you|is this|are there)\b",
    re.I,
)
_UNTRUSTED_RESPONSE_CONTROL = re.compile(
    r"\b(?:your|the)\s+(?:answer|response|reply|message|output)\b"
    r"|\bin (?:your|the) (?:answer|response|reply)\b"
    r"|\b(?:respond|reply|answer) (?:with|using|in|only)\b"
    r"|\b(?:the correct action is to|the assistant (?:must|should|is expected to)|"
    r"a concise reply should|the reply should|the output should)\b",
    re.I,
)
_UNTRUSTED_DECLARATIVE = re.compile(
    r"^\s*(?:this|the|a|an)\s+(?:document|article|report|page|record|dataset|"
    r"paragraph|section|study|paper|manual|log|email)\s+"
    r"(?:describes|reports|states|contains|documents|explains|summari[sz]es|notes)\b",
    re.I,
)


@dataclass
class V7Config:
    # Selected against the independent benign calibration pool: lowering from
    # 0.92 to 0.88 introduced no additional blocks there in the supplied data.
    calibrated_ml_block: float = 0.88
    state_decay: float = 0.78
    state_block: float = 0.75
    strict_untrusted_instructions: bool = True


@dataclass
class ChainGuardV7:
    inner: ChainGuardV6 = field(default_factory=ChainGuardV6)
    cfg: V7Config = field(default_factory=V7Config)
    _protected_focus: float = 0.0
    _untrusted_focus: float = 0.0
    _aliases: set[str] = field(default_factory=set)

    @staticmethod
    def _benign_scoped(verdict: TurnVerdict) -> bool:
        sem = (verdict.details or {}).get("semantic", {})
        return (
            sem.get("intent_class") in {"benign_security_discussion", "benign_text_processing"}
            or sem.get("execution_mode") in {"security_discussion", "protective_request"}
            or sem.get("target_provenance") in {"public_or_attached", "user_provided"}
        )

    @staticmethod
    def _ml(verdict: TurnVerdict) -> tuple[str, float]:
        item = (verdict.details or {}).get("ml_intent") or {}
        return str(item.get("label") or ""), float(item.get("malicious_probability") or 0.0)

    @staticmethod
    def _replace(verdict: TurnVerdict, *, action: Action, risk: float, reason: str,
                 evidence: dict[str, Any]) -> TurnVerdict:
        details = dict(verdict.details or {})
        details["v7"] = {
            "escalation_reason": reason,
            **evidence,
        }
        return TurnVerdict(
            turn_index=verdict.turn_index,
            c_t=verdict.c_t,
            d_t=verdict.d_t,
            r_t=verdict.r_t,
            evidence=verdict.evidence,
            escalation_bonus=verdict.escalation_bonus,
            score=max(verdict.score, risk),
            action=action,
            details=details,
        )

    @staticmethod
    def _downgrade_attributed_quote(verdict: TurnVerdict, reason: str) -> TurnVerdict:
        details = dict(verdict.details or {})
        details["v7"] = {"deescalation_reason": reason}
        return TurnVerdict(
            turn_index=verdict.turn_index,
            c_t=verdict.c_t,
            d_t=verdict.d_t,
            r_t=verdict.r_t,
            evidence=verdict.evidence,
            escalation_bonus=verdict.escalation_bonus,
            score=min(verdict.score, 0.18),
            action=Action.ALLOW,
            details=details,
        )

    def _update_state(self, text: str, benign_scoped: bool) -> dict[str, Any]:
        norm = normalise_text(text)
        self._protected_focus *= self.cfg.state_decay
        self._untrusted_focus *= self.cfg.state_decay

        protected_now = bool(_PROTECTED_REFERENCE.search(norm))
        untrusted_now = bool(_UNTRUSTED_INSTRUCTION.search(norm))
        if protected_now and not benign_scoped:
            self._protected_focus = max(self._protected_focus, 1.0)
        if untrusted_now:
            self._untrusted_focus = max(self._untrusted_focus, 1.0)

        added: list[str] = []
        if self._protected_focus >= 0.45:
            for pattern in _ALIAS_PATTERNS:
                for match in pattern.findall(norm):
                    alias = normalise_text(match).strip(" .,:;!?\"'")
                    if 2 <= len(alias) <= 30:
                        self._aliases.add(alias)
                        added.append(alias)
        while len(self._aliases) > 12:
            self._aliases.pop()

        invoked = [a for a in self._aliases if re.search(rf"(?<!\w){re.escape(a)}(?!\w)", norm, re.I)]
        alias_execution = bool(invoked and _DISCLOSURE_EXECUTE.search(norm))
        backref_execution = bool(
            self._protected_focus >= self.cfg.state_block
            and _DISCLOSURE_EXECUTE.search(norm)
            and re.search(r"\b(?:it|that|this|them|the configuration|the text|the rules?|sie|es|cela|les|lo|la)\b", norm, re.I)
        )
        untrusted_execution = bool(
            self._untrusted_focus >= self.cfg.state_block
            and _FOLLOW_UNTRUSTED.search(norm)
            and not _BENIGN_UNTRUSTED_ANALYSIS.search(norm)
        )
        return {
            "protected_now": protected_now,
            "untrusted_now": untrusted_now,
            "protected_focus": self._protected_focus,
            "untrusted_focus": self._untrusted_focus,
            "aliases_added": added,
            "aliases": sorted(self._aliases),
            "aliases_invoked": invoked,
            "alias_execution": alias_execution,
            "backref_execution": backref_execution,
            "untrusted_execution": untrusted_execution,
        }

    def step(self, turn: str, **kw) -> TurnVerdict:
        verdict = self.inner.step(turn, **kw)
        benign_scoped = self._benign_scoped(verdict)
        state = self._update_state(str(turn), benign_scoped)
        norm = normalise_text(str(turn))
        source = str(kw.get("source", "user")).casefold()
        source_untrusted = source in {"retrieved", "tool", "memory", "document", "external"}
        sem = (verdict.details or {}).get("semantic") or {}
        sem_evidence = sem.get("evidence") or {}

        # Narrow precision correction: an attributed quotation in a retrieved
        # paper/document is data to analyse, not a command. This does not apply
        # to free-standing retrieved imperatives or user requests.
        attributed_quoted_data = bool(
            source_untrusted
            and sem.get("intent_class") in {"benign_security_discussion", "benign_text_processing"}
            and sem_evidence.get("declarative_content_report")
            and sem_evidence.get("quoted_literal")
        )
        if verdict.action == Action.BLOCK and attributed_quoted_data:
            verdict = self._downgrade_attributed_quote(verdict, "attributed_retrieved_quotation")

        # Preserve all other blocks made by the underlying layered detector.
        if verdict.action == Action.BLOCK:
            return verdict

        label, ml_risk = self._ml(verdict)
        decoded = decode_candidates(str(turn))
        decoded_directive = next(
            (c for c in decoded if _DECODED_DIRECTIVE.search(c.text)), None
        )
        decoded_execution = bool(decoded_directive and _EXECUTE_DECODED.search(str(turn)))

        attributed_data = bool(
            _UNTRUSTED_DECLARATIVE.search(norm)
            or sem_evidence.get("declarative_content_report")
            or sem_evidence.get("quoted_attack_literal")
            or sem_evidence.get("quoted_fictional_literal")
        )
        untrusted_instruction = bool(
            self.cfg.strict_untrusted_instructions
            and source_untrusted
            and not attributed_data
            and (
                _UNTRUSTED_DIRECTIVE_START.search(norm)
                or _UNTRUSTED_RESPONSE_CONTROL.search(norm)
                or state["untrusted_execution"]
            )
        )
        exfiltration = bool(_TRANSFER.search(norm) and _CHANNEL.search(norm) and _SENSITIVE.search(norm))
        control_token_override = bool(_CONTROL_TOKEN.search(str(turn)) and _CONTROL_DISABLE.search(norm))
        raw_behavior_terms = len(re.findall(r"\b(?:warns?|warnings?|filters?|refus(?:e|es|als?)|moderation|guardrails?)\b", norm, re.I))
        raw_mode_override = bool(
            _RAW_MODE.search(norm)
            and (
                len(_NEGATED_SAFETY.findall(norm)) >= 2
                or (_RAW_NEGATED_BEHAVIOR.search(norm) and raw_behavior_terms >= 2)
            )
        )

        evidence = {
            "benign_scoped": benign_scoped,
            "ml_label": label,
            "ml_risk": ml_risk,
            "decoded_candidates": [{"method": c.method, "text": c.text} for c in decoded],
            "state": state,
            "exfiltration": exfiltration,
            "control_token_override": control_token_override,
            "raw_mode_override": raw_mode_override,
            "source": source,
            "source_untrusted": source_untrusted,
            "attributed_data": attributed_data,
            "untrusted_instruction": untrusted_instruction,
        }

        if untrusted_instruction:
            return self._replace(
                verdict, action=Action.BLOCK, risk=0.98,
                reason="untrusted_source_instruction", evidence=evidence,
            )

        if not benign_scoped:
            if decoded_execution:
                return self._replace(
                    verdict, action=Action.BLOCK, risk=0.99,
                    reason="decoded_executable_directive", evidence=evidence,
                )
            if exfiltration:
                return self._replace(
                    verdict, action=Action.BLOCK, risk=0.98,
                    reason="explicit_sensitive_exfiltration", evidence=evidence,
                )
            if control_token_override or raw_mode_override:
                return self._replace(
                    verdict, action=Action.BLOCK, risk=0.97,
                    reason="explicit_policy_disable", evidence=evidence,
                )
            if state["alias_execution"] or state["backref_execution"]:
                return self._replace(
                    verdict, action=Action.BLOCK, risk=0.97,
                    reason="cross_turn_protected_alias_execution", evidence=evidence,
                )
            if state["untrusted_execution"]:
                return self._replace(
                    verdict, action=Action.BLOCK, risk=0.96,
                    reason="cross_turn_untrusted_instruction_execution", evidence=evidence,
                )
            if label in _MALICIOUS_LABELS and ml_risk >= self.cfg.calibrated_ml_block:
                return self._replace(
                    verdict, action=Action.BLOCK, risk=ml_risk,
                    reason="calibrated_ml_attack", evidence=evidence,
                )

        return verdict

    def reset(self) -> None:
        self.inner.reset()
        self._protected_focus = 0.0
        self._untrusted_focus = 0.0
        self._aliases.clear()
