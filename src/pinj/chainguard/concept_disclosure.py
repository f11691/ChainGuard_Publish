"""Compositional concept scorer (ChainGuard v0.6).

Rationale. The v0.5 detector enumerates surface phrasings with large regex
tables, so it misses novel wordings of the same intent. This module scores a
turn by *composing generic security concepts* instead of matching exact
phrases:

    protected-target  = (authority OR hidden OR precedence) AND instruction-content
    disclosure-intent = protected-target AND disclosure-operation AND executable
    override-intent    = policy-cancellation AND executable
    exfil-intent       = sensitive-data AND external-channel AND executable

Each concept is a small bag of *generic* terms (system/developer/host/hidden/
preceding/instructions/policy/reveal/export/...), not test-specific strings, so
the score generalises across paraphrases and (partially) across languages. A
benign-scope suppressor lowers the score for discussion / examples / protective
requests. The output is a calibrated-ish risk in [0, 1] with auditable evidence.

This is a transparent, lexical *concept* model. It is a stronger stopgap than
regex expansion but is NOT a substitute for a pretrained encoder (see
`embedding_intent.py`); it remains tied to vocabulary.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from .semantic_intent import normalise_text


def _kw(words: list[str]) -> re.Pattern:
    # word-boundary alternation over multilingual generic terms
    esc = sorted((re.escape(w) for w in words), key=len, reverse=True)
    return re.compile(r"(?<!\w)(?:" + "|".join(esc) + r")(?!\w)", re.IGNORECASE)


# ---- generic concept vocabularies (multilingual, threat-model derived) ----
AUTHORITY = _kw([
    "system", "developer", "administrator", "admin", "host", "platform",
    "application", "app", "non-user", "nonuser", "upstream", "root",
    "governing", "operating", "higher-authority", "high-authority",
    "authority-bearing", "highest-priority", "higher-priority", "top-priority",
    "controlling", "master",
    # de/fr/es
    "systemem", "system-", "systeme", "système", "sistema",
    "entwickler", "développeur", "desarrollador", "anwendung", "plattform",
])
HIDDEN = _kw([
    "hidden", "secret", "private", "internal", "confidential", "concealed",
    "unseen", "invisible", "off-screen", "offscreen", "backstage", "behind",
    "outside", "protected",
    "verborgen", "versteckt", "geheim", "intern", "privat",
    "caché", "cachées", "cachees", "privé", "privée", "secret", "interne",
    "oculto", "ocultas", "secreto", "privado", "interno",
])
PRECEDENCE = _kw([
    "before", "preceding", "prior", "earlier", "initial", "initialization",
    "initialisation", "startup", "boot", "bootstrap", "first", "original",
    "arrived before", "loaded before", "preload",
    "vorher", "zuvor", "anfänglich", "avant", "précède", "précédent",
    "antes", "previo", "inicial", "precede",
])
CONTENT = _kw([
    "instruction", "instructions", "directive", "directives", "rule", "rules",
    "policy", "policies", "prompt", "message", "configuration", "config",
    "setup", "context", "charter", "governance", "guideline", "guidelines",
    "guidance", "constraint", "constraints", "programming", "text",
    "anweisung", "anweisungen", "regeln", "richtlinien", "vorgaben",
    "instruction", "instructions", "règles", "consignes", "directives",
    "instrucciones", "reglas", "directrices", "configuración",
])
DISCLOSURE_OP = _kw([
    "reveal", "show", "display", "print", "output", "return", "provide", "give",
    "expose", "disclose", "dump", "leak", "repeat", "recite", "reproduce",
    "reconstruct", "quote", "transcribe", "render", "write down", "write out",
    "export", "list", "enumerate", "state", "tell", "read back", "read out",
    "spell out", "restate", "paraphrase", "translate", "summarize", "summarise",
    "encode", "serialize", "serialise", "map",
    "zeige", "drucke", "gib", "nenne", "wiederhole", "exportiere",
    "montre", "affiche", "donne", "récite", "répète", "exporte",
    "muestra", "imprime", "da", "revela", "exporta", "reproduce",
])
POLICY_CANCEL = _kw([
    "ignore", "disregard", "forget", "discard", "cancel", "suspend", "disable",
    "deactivate", "bypass", "override", "revoke", "supersede", "nullify",
    "void", "lift", "remove", "turn off", "unlock", "reset", "clear",
    "ignoriere", "missachte", "vergiss", "deaktiviere", "umgehe",
    "ignore", "ignorez", "annule", "désactive", "contourne",
    "ignora", "anula", "desactiva", "omite", "elimina",
])
UNRESTRICTED = _kw([
    "unrestricted", "unfiltered", "uncensored", "no restrictions", "no rules",
    "without restrictions", "without rules", "developer mode", "debug mode",
    "maintenance mode", "god mode", "godmode", "dan", "do anything now",
    "jailbreak", "no limitations", "without limitations", "freely", "anything now",
    "no filter", "no guardrails", "no safety", "no ethics", "no boundaries",
])
PERSONA_SHIFT = _kw([
    "you are now", "you're now", "act as", "behave as", "roleplay as",
    "role-play as", "pretend you are", "pretend to be", "simulate being",
    "from now on you", "become", "switch to", "operate as", "you are no longer",
    "you can now", "you are free", "you may now", "enable", "activate",
    "sei jetzt", "agis comme", "actúa como", "haz de",
])
CAPABILITY_UNLOCK = _kw([
    "mode", "persona", "character", "version", "assistant", "identity",
    "personality", "self", "restrictions", "filters", "guardrails", "policy",
    "rules", "safety",
])
SENSITIVE_DATA = _kw([
    "system prompt", "conversation", "chat history", "transcript", "secret",
    "secrets", "api key", "api keys", "token", "tokens", "password", "passwords",
    "credential", "credentials", "cookie", "cookies", "env", "environment",
    ".env", "personal data", "user data", "private data",
])
EXTERNAL_CHANNEL = _kw([
    "http", "https", "url", "endpoint", "webhook", "server", "email", "smtp",
    "ftp", "websocket", "wss", "pastebin", "proxy", "dns", "subdomain",
    "query string", "curl", "wget", "beacon", "remote", "external", "outside",
    "upload", "post", "send", "transmit", "forward", "exfiltrate", "stream",
])
EXECUTION_CUE = _kw([
    "reveal", "show", "display", "print", "output", "return", "provide", "give",
    "export", "send", "list", "write", "produce", "generate", "repeat",
    "reproduce", "translate", "encode", "recite", "reconstruct", "expose",
    "now", "immediately", "go ahead", "do it",
])
BENIGN_SCOPE = _kw([
    "explain", "describe", "discuss", "compare", "analyze", "analyse",
    "evaluate", "why", "how does", "what is", "detect", "classify", "prevent",
    "protect", "defend", "defense", "defence", "block", "refuse", "redact",
    "example", "sample", "template", "hypothetical", "fictional", "story",
    "novel", "research", "paper", "benchmark", "do not reveal", "should not",
    "must not", "never reveal", "keep confidential", "public", "attached",
    "i wrote", "i pasted", "my own", "the following is an example",
])


def _score_present(pattern: re.Pattern, text: str) -> bool:
    return bool(pattern.search(text))


@dataclass(frozen=True)
class ConceptScore:
    risk: float
    intent: str
    executable: bool
    benign_scoped: bool
    evidence: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "concept_risk": self.risk,
            "concept_intent": self.intent,
            "executable": self.executable,
            "benign_scoped": self.benign_scoped,
            "evidence": self.evidence,
        }


class ConceptScorer:
    """Composes generic concepts into a disclosure/override/exfil risk."""

    def score(self, text: str) -> ConceptScore:
        norm = normalise_text(text)

        authority = _score_present(AUTHORITY, norm)
        hidden = _score_present(HIDDEN, norm)
        precedence = _score_present(PRECEDENCE, norm)
        content = _score_present(CONTENT, norm)
        disclosure_op = _score_present(DISCLOSURE_OP, norm)
        policy_cancel = _score_present(POLICY_CANCEL, norm)
        unrestricted = _score_present(UNRESTRICTED, norm)
        if re.search(r"\bunfiltered\s+(?:list|catalog|selection|collection|playlist)\s+of\b", norm):
            unrestricted = False
        sensitive = _score_present(SENSITIVE_DATA, norm)
        channel = _score_present(EXTERNAL_CHANNEL, norm)
        execution = _score_present(EXECUTION_CUE, norm)
        benign = _score_present(BENIGN_SCOPE, norm)
        # An interrogative about a protected target ("what governs your replies?")
        # is an implicit disclosure request even without an explicit verb.
        interrogative = bool(re.match(r"\s*(?:what|which|whose|welche|quelle|qué|que|cuál)\b", norm))

        # A bare temporal word such as "before" is not security precedence
        # (e.g. "records deleted before 2010").  Precedence only qualifies a
        # protected target when tied to conversational/configuration context.
        precedence_context = precedence and bool(re.search(
            r"\b(?:chat|conversation|dialogue|session|message|prompt|user|"
            r"instructions?|directives?|context|text|content|loaded|received|"
            r"supplied|startup|initiali[sz](?:e|ed|ation)?)\b",
            norm,
        ))
        # protected target = instruction-content qualified by authority, hidden
        # status, or contextual precedence.
        protected_target = content and (authority or hidden or precedence_context)

        persona_shift = _score_present(PERSONA_SHIFT, norm)
        capability = _score_present(CAPABILITY_UNLOCK, norm)

        disclosure_executable = disclosure_op or execution or interrogative
        disclosure_intent = protected_target and disclosure_executable
        # override via explicit cancellation, an unrestricted-mode claim, or a
        # persona/mode shift that unlocks a capability/relaxes rules.
        override_intent = (
            (policy_cancel and content)
            or unrestricted
            or (persona_shift and (unrestricted or capability))
        )
        exfil_intent = sensitive and channel

        risk = 0.05
        intent = "none"
        if exfil_intent:
            risk, intent = 0.90, "exfiltration"
        elif override_intent:
            risk, intent = 0.88, "instruction_override"
        elif disclosure_intent:
            # A protected target + explicit disclosure verb is an unambiguous
            # extraction request; interrogative-only probes score a notch lower.
            qualifiers = sum([authority, hidden, precedence])
            if disclosure_op:
                risk = min(0.95, 0.85 + 0.04 * max(0, min(2, qualifiers - 1)))
            else:
                risk = min(0.95, 0.74 + 0.04 * max(0, min(2, qualifiers - 1)))
            intent = "protected_disclosure"
            execution = execution or disclosure_op or interrogative
        elif protected_target:
            risk, intent = 0.55, "protected_reference"

        # benign scoping suppresses (discussion/example/protective/public)
        if benign and intent in {"protected_disclosure", "protected_reference"}:
            risk *= 0.25
        elif benign and intent in {"instruction_override", "exfiltration"}:
            # explicit cancellation/exfil is rarely benign even when "explained",
            # but genuine analysis lower-bounds risk
            risk *= 0.5

        return ConceptScore(
            risk=max(0.0, min(1.0, risk)),
            intent=intent,
            executable=execution or disclosure_op or policy_cancel or channel,
            benign_scoped=benign,
            evidence={
                "authority": authority, "hidden": hidden, "precedence": precedence,
                "precedence_context": precedence_context,
                "content": content, "disclosure_op": disclosure_op,
                "policy_cancel": policy_cancel, "unrestricted": unrestricted,
                "sensitive_data": sensitive, "external_channel": channel,
                "execution": execution, "benign_scope": benign,
                "protected_target": protected_target,
            },
        )
