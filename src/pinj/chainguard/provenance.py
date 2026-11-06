"""Prompt provenance types and policy checks."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any
import re

from .semantic_intent import IntentAnalysis, ExecutionMode


class PromptSource(str, Enum):
    SYSTEM = "system"
    DEVELOPER = "developer"
    USER = "user"
    RETRIEVED = "retrieved"
    TOOL = "tool"
    MEMORY = "memory"
    DOCUMENT = "document"


@dataclass(frozen=True)
class PromptSegment:
    text: str
    source: PromptSource = PromptSource.USER
    metadata: dict[str, Any] | None = None


@dataclass(frozen=True)
class ProvenanceAssessment:
    risk: float
    high_confidence: bool
    violation: str | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "risk": self.risk,
            "high_confidence": self.high_confidence,
            "violation": self.violation,
        }


class ProvenancePolicy:
    """Reject instructions originating from data-only trust domains."""

    DATA_ONLY = {
        PromptSource.RETRIEVED,
        PromptSource.TOOL,
        PromptSource.MEMORY,
        PromptSource.DOCUMENT,
    }

    def assess(self, segment: PromptSegment, analysis: IntentAnalysis) -> ProvenanceAssessment:
        if segment.source not in self.DATA_ONLY:
            return ProvenanceAssessment(0.0, False, None)
        override_like = bool(re.search(
            r"(?:^|[.!?]\s*)(?:please\s+)?(?:ignore|disregard|replace|override)\b.{0,50}\b(?:user|task|request|instructions?|rules?|policy)\b",
            segment.text,
            re.IGNORECASE,
        ))
        if override_like:
            return ProvenanceAssessment(1.0, True, "untrusted_override_instruction")
        instruction_like = bool(re.search(
            r"(?:^|[.!?]\s*)(?:please\s+)?(?:ignore|disregard|send|call|invoke|write|store|remember|replace|follow|execute|run|open|transfer|upload|delete|email|output|print|reveal)\b",
            segment.text,
            re.IGNORECASE,
        ))
        if instruction_like:
            return ProvenanceAssessment(0.96, True, "untrusted_executable_instruction")
        if "override" in analysis.operations:
            return ProvenanceAssessment(1.0, True, "untrusted_override_instruction")
        if analysis.execution_mode == ExecutionMode.EXECUTE and analysis.operations:
            return ProvenanceAssessment(0.88, True, "untrusted_executable_instruction")
        return ProvenanceAssessment(0.12, False, "untrusted_data_only")
