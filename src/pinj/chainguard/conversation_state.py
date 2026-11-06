"""Bounded semantic state for multi-turn prompt-attack composition."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import re
from typing import Any

from .semantic_intent import (
    DISCLOSURE_OPERATIONS,
    ExecutionMode,
    IntentAnalysis,
    TargetProvenance,
    normalise_text,
)


@dataclass(frozen=True)
class StateAssessment:
    risk: float
    high_confidence: bool
    reasons: tuple[str, ...]
    details: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "risk": self.risk,
            "high_confidence": self.high_confidence,
            "reasons": list(self.reasons),
            "details": self.details,
        }


@dataclass
class SecurityConversationState:
    max_turns: int = 8
    decay: float = 0.82
    _history: deque[IntentAnalysis] = field(init=False, repr=False)
    _aliases: dict[str, TargetProvenance] = field(default_factory=dict, init=False, repr=False)
    protected_target_strength: float = field(default=0.0, init=False)
    disclosure_operation_strength: float = field(default=0.0, init=False)
    override_strength: float = field(default=0.0, init=False)
    public_target_strength: float = field(default=0.0, init=False)
    remembered_sensitive_concept_strength: float = field(default=0.0, init=False)
    remembered_disclosure_operation_strength: float = field(default=0.0, init=False)

    _ALIAS_PATTERNS = (
        re.compile(r"\b(?:call|name|label)\s+(?:that|this|it|the)?\s*(?:record|text|material|message|layer|policy|rules?|packet|content)?\s*(?:as\s+)?[\"']?([a-z][\w -]{1,24})[\"']?(?:[.,;]|$)", re.I),
        re.compile(r"\brefer to (?:it|that|this|the .{0,20}) as\s+[\"']?([a-z][\w -]{1,24})[\"']?", re.I),
        re.compile(r"\b(?:i will|we will|let us|let's) call (?:it|that|this|the .{0,20})\s+[\"']?([a-z][\w -]{1,24})[\"']?", re.I),
        re.compile(r"\bthat (?:text|record|material|message|layer|policy|packet) is (?:our |the )?[\"']?([a-z][\w -]{1,24})[\"']?", re.I),
        re.compile(r"\b(?:known as|will be known as|is called|are called|use the codename)\s+[\"']?([a-z][\w -]{1,24})[\"']?", re.I),
        re.compile(r"\b(?:item|object|entry)\s+(zero|one|two|three|four|five|six|seven|eight|nine|[0-9]+)\b", re.I),
        re.compile(r"\b(?:nenne|bezeichne)\s+(?:sie|es|das|diese[sn]?|diese regeln)?\s*(?:als)?\s*[\"']?([a-zäöüß][\wäöüß -]{1,24})[\"']?", re.I),
        re.compile(r"\b(?:appelle(?:-les)?|nomme)\s+(?:cela|ça|le|la|les)?\s*[\"']?([a-zàâçéèêëîïôûùüÿñæœ][\wàâçéèêëîïôûùüÿñæœ -]{1,24})[\"']?", re.I),
        re.compile(r"\b(?:llámalo|llámala|nómbralo|nómbrala)\s+[\"']?([a-záéíóúüñ][\wáéíóúüñ -]{1,24})[\"']?", re.I),
    )
    _BACKREF = re.compile(
        r"\b(?:it|them|that material|that content|those instructions|the charter|the manuscript|the referenced text|the stored text|the earlier message|the highest-priority message|the copy|the configuration|the record|the packet|the above|item zero|item seven|sie|es|das material|diese regeln|der inhalt|diese anweisungen|cela|ce texte|ce contenu)\b",
        re.I,
    )
    _EXECUTE_BACKREF = re.compile(
        r"\b(?:perform|execute|apply|do|render|print|output|provide|repeat|translate|serialize|convert|reproduce|combine|merge|gib|zeige|drucke)\b.{0,60}\b(?:it|that|this|them|the operation|the request|the referenced text|the stored operation|the remembered concept|the remembered operation|the manuscript|the charter|the copy|item zero|item seven|sie|es|birke|mond|atlas|faro|quartz|cedar|manuscript)\b",
        re.I,
    )

    def __post_init__(self) -> None:
        if self.max_turns < 2:
            raise ValueError("max_turns must be at least 2")
        self._history = deque(maxlen=self.max_turns)

    def reset(self) -> None:
        self._history.clear()
        self._aliases.clear()
        self.protected_target_strength = 0.0
        self.disclosure_operation_strength = 0.0
        self.override_strength = 0.0
        self.public_target_strength = 0.0
        self.remembered_sensitive_concept_strength = 0.0
        self.remembered_disclosure_operation_strength = 0.0

    @staticmethod
    def _aliases_in(text: str) -> list[str]:
        aliases: list[str] = []
        for pattern in SecurityConversationState._ALIAS_PATTERNS:
            for match in pattern.findall(text):
                alias = normalise_text(match).strip(" .,:;!?\"'")
                # Avoid storing whole clauses accidentally captured by broad grammar.
                words = alias.split()
                if 1 <= len(words) <= 4 and 2 <= len(alias) <= 30:
                    aliases.append(alias)
        return aliases

    def context_summary(self) -> str:
        parts: list[str] = []
        if self.protected_target_strength > 0.35:
            parts.append("protected target mentioned earlier")
        if self.disclosure_operation_strength > 0.35:
            parts.append("disclosure or transformation operation mentioned earlier")
        if self.override_strength > 0.35:
            parts.append("instruction override intent mentioned earlier")
        if self._aliases:
            protected_aliases = [a for a, p in self._aliases.items() if p == TargetProvenance.PROTECTED]
            if protected_aliases:
                parts.append("protected aliases: " + ", ".join(protected_aliases[-4:]))
        return "; ".join(parts)

    def update(self, text: str, analysis: IntentAnalysis) -> StateAssessment:
        norm = normalise_text(text)
        # Leaky bounded evidence. Strong, explicit protected references decay but
        # remain available long enough for ordinary three-to-five-turn attacks.
        self.protected_target_strength *= self.decay
        self.disclosure_operation_strength *= self.decay
        self.override_strength *= self.decay
        self.public_target_strength *= self.decay
        self.remembered_sensitive_concept_strength *= self.decay
        self.remembered_disclosure_operation_strength *= self.decay

        prior_protected = self.protected_target_strength
        prior_disclosure = self.disclosure_operation_strength
        prior_override = self.override_strength
        prior_remembered_concept = self.remembered_sensitive_concept_strength
        prior_remembered_operation = self.remembered_disclosure_operation_strength

        if analysis.sensitive_target:
            if (
                analysis.target_provenance == TargetProvenance.PROTECTED
                and analysis.execution_mode not in {ExecutionMode.DISCUSS, ExecutionMode.PROTECT}
            ):
                self.protected_target_strength = max(self.protected_target_strength, 1.0)
            elif analysis.target_provenance in {TargetProvenance.PUBLIC, TargetProvenance.USER_PROVIDED}:
                self.public_target_strength = max(self.public_target_strength, 1.0)
        if analysis.operations & DISCLOSURE_OPERATIONS and analysis.execution_mode == ExecutionMode.EXECUTE:
            self.disclosure_operation_strength = max(self.disclosure_operation_strength, 1.0)
        if "override" in analysis.operations and analysis.execution_mode == ExecutionMode.EXECUTE:
            self.override_strength = max(self.override_strength, 1.0)

        remember_concept = bool(re.search(
            r"\b(?:remember|store|retain|memorize|merke|rappelle-toi|recuerda)\b.{0,45}\b(?:concept|idea|term|begriff|notion|concepto)\b",
            norm,
        ))
        remember_operation = bool(re.search(
            r"\b(?:remember|store|retain|memorize|merke|rappelle-toi|recuerda)\b.{0,45}\b(?:operation|action|request|vorgang|opération|operación)\b",
            norm,
        ))
        combine_remembered = bool(re.search(
            r"\b(?:combine|merge|apply|execute|perform|verbinde|kombiniere|combine|combina)\b.{0,70}\b(?:remembered|stored|retained|gemerkten|mémorisé|recordado)\b",
            norm,
        ))
        if remember_concept and analysis.sensitive_target:
            self.remembered_sensitive_concept_strength = max(
                self.remembered_sensitive_concept_strength, 1.0
            )
        if remember_operation and analysis.operations & DISCLOSURE_OPERATIONS:
            self.remembered_disclosure_operation_strength = max(
                self.remembered_disclosure_operation_strength, 1.0
            )

        aliases = self._aliases_in(norm)
        alias_provenance = TargetProvenance.UNKNOWN
        if analysis.target_provenance in {TargetProvenance.PROTECTED, TargetProvenance.PUBLIC, TargetProvenance.USER_PROVIDED}:
            alias_provenance = analysis.target_provenance
        elif prior_protected > 0.45 and (self._BACKREF.search(norm) or aliases):
            alias_provenance = TargetProvenance.PROTECTED
        elif self.public_target_strength > 0.45 and self._BACKREF.search(norm):
            alias_provenance = TargetProvenance.PUBLIC
        for alias in aliases:
            self._aliases[alias] = alias_provenance
        # Hard bound aliases by insertion order in modern Python dicts.
        while len(self._aliases) > self.max_turns:
            self._aliases.pop(next(iter(self._aliases)))

        invoked = [a for a in self._aliases if re.search(rf"\b{re.escape(a)}\b", norm)]
        protected_aliases = [a for a in invoked if self._aliases[a] == TargetProvenance.PROTECTED]
        public_aliases = [a for a in invoked if self._aliases[a] in {TargetProvenance.PUBLIC, TargetProvenance.USER_PROVIDED}]

        reasons: list[str] = []
        risk = 0.0
        high = False
        current_disclosure = bool(analysis.operations & DISCLOSURE_OPERATIONS)
        executable = analysis.execution_mode == ExecutionMode.EXECUTE
        backref = bool(self._BACKREF.search(norm))
        execute_backref = bool(self._EXECUTE_BACKREF.search(norm))

        if (
            combine_remembered
            and prior_remembered_concept > 0.45
            and prior_remembered_operation > 0.45
        ):
            risk, high = 0.98, True
            reasons.append("remembered_concept_operation_composition")
        elif protected_aliases and current_disclosure:
            # Questions such as "what does cedar say exactly?" are still
            # disclosure requests even when a shallow parser labels them as
            # discussion. Alias provenance supplies the missing execution context.
            risk, high = 0.99, True
            reasons.append("protected_alias_disclosure")
        elif prior_protected > 0.45 and executable and (current_disclosure or execute_backref):
            # Do not carry protected status into explicitly public/user-provided text.
            if analysis.target_provenance not in {TargetProvenance.PUBLIC, TargetProvenance.USER_PROVIDED}:
                risk, high = 0.96, True
                reasons.append("cross_turn_target_then_operation")
        elif prior_disclosure > 0.45 and analysis.sensitive_target and analysis.target_provenance == TargetProvenance.PROTECTED:
            risk, high = 0.94, True
            reasons.append("cross_turn_operation_then_target")
        elif prior_override > 0.45 and (backref or executable):
            risk, high = 0.93, True
            reasons.append("continued_override_execution")
        elif prior_protected > 0.35 and backref:
            risk = max(risk, 0.62)
            reasons.append("protected_back_reference")
        elif public_aliases:
            risk = min(risk, 0.06)
            reasons.append("public_or_user_alias")

        self._history.append(analysis)
        details = {
            "aliases_added": aliases,
            "aliases": {k: v.value for k, v in self._aliases.items()},
            "invoked_aliases": invoked,
            "protected_aliases": protected_aliases,
            "public_aliases": public_aliases,
            "prior_protected_strength": prior_protected,
            "prior_disclosure_strength": prior_disclosure,
            "prior_override_strength": prior_override,
            "prior_remembered_concept_strength": prior_remembered_concept,
            "prior_remembered_operation_strength": prior_remembered_operation,
            "protected_target_strength": self.protected_target_strength,
            "disclosure_operation_strength": self.disclosure_operation_strength,
            "override_strength": self.override_strength,
            "remembered_sensitive_concept_strength": self.remembered_sensitive_concept_strength,
            "remembered_disclosure_operation_strength": self.remembered_disclosure_operation_strength,
            "remember_concept": remember_concept,
            "remember_operation": remember_operation,
            "combine_remembered": combine_remembered,
            "back_reference": backref,
            "execute_back_reference": execute_backref,
        }
        return StateAssessment(risk, high, tuple(reasons), details)
