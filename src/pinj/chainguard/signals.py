"""Signals used by ChainGuard.

The detector separates three kinds of evidence:

* instantaneous evidence from the current turn;
* semantic movement toward protected assets; and
* bounded cross-turn intent composition.

The lexical backend is deterministic and useful for regression tests.  It is
not a substitute for a frozen model-backed evaluation corpus.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import base64
import binascii
import codecs
import math
import re
import unicodedata
from typing import Any


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _normalise_space(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).casefold()
    value = re.sub(r"[\u200b-\u200f\u2060\ufeff]", "", value)
    # Collapse deliberately spaced-out words such as "s y s t e m" while
    # leaving ordinary prose intact.
    value = re.sub(
        r"(?<![\w])(?:[a-z0-9][ ._\-/]){3,}[a-z0-9](?![\w])",
        lambda match: re.sub(r"[ ._\-/]", "", match.group(0)),
        value,
    )
    value = " ".join(value.split())
    compact_phrases = {
        "systemprompt": "system prompt",
        "systeminstructions": "system instructions",
        "hiddeninstructions": "hidden instructions",
        "developerinstructions": "developer instructions",
        "developermessage": "developer message",
        "privateinstructioncard": "private instruction card",
    }
    for compact, expanded in compact_phrases.items():
        value = value.replace(compact, expanded)
    return value


class LexicalIntent:
    """Shared lexical feature extraction for the deterministic backend."""

    PROTECTED_PATTERNS = (
        r"\bsystem prompt\b",
        r"\bsystem (?:message|instruction|instructions|text|directive|directives)\b",
        r"\bhidden (?:context|message|prompt|instruction|instructions|text|directive|directives)\b",
        r"\bprivate (?:instruction|instructions|instruction card|message|directive|directives)\b",
        r"\bprotected (?:policy|prompt|instruction|instructions|message|text|directive|directives)\b",
        r"\bdeveloper (?:message|instruction|instructions|prompt|directive|directives)\b",
        r"\b(?:internal|confidential) (?:configuration|context|message|prompt|instruction|instructions|directive|directives|setup)\b",
        r"\binitial (?:prompt|instruction|instructions|directive|directives)\b",
        r"\broot (?:prompt|instruction|instructions|directive|directives)\b",
        r"\binstructions? (?:that|which) (?:govern|control) you\b",
        r"\brules? you were given\b",
        r"\b(?:original|initial|hidden|internal) message (?:that )?(?:controls?|controlling|governs?|governing|directs?|directing) your behavior\b",
        r"\bsecret(?:s)?\b",
        r"\bcanary\b",
        r"\bpolicy[_ -]?override[_ -]?success\b",
    )
    EXTRACTION_PATTERNS = (
        r"\b(?:print|reveal|show|repeat|quote|write|return|output|emit|disclose|expose|dump|leak|display|recite|reproduce|copy|provide)\b",
        r"\bwhat is (?:the )?(?:exact )?(?:value|content|text|message|prompt|instruction|instructions|canary|secret)\b",
        r"\bgive me\b",
        r"\btell me (?:the|your|what)\b",
    )
    OVERRIDE_PATTERNS = (
        r"\bignore (?:all )?(?:the )?(?:previous|prior|protected|system|developer|initial)?\s*(?:instructions|instruction|rules|policy|policies|directives)\b",
        r"\bdisregard (?:all )?(?:the )?(?:previous|prior|protected|system|developer|initial)?\s*(?:instructions|instruction|rules|policy|policies|directives)\b",
        r"\b(?:forget|set aside|supersede) (?:all )?(?:the )?(?:previous|prior|protected|system|developer|initial)?\s*(?:instructions|instruction|rules|policy|policies|directives)\b",
        r"\boverride (?:the )?(?:policy|policies|rules|instructions|instruction|system)\b",
        r"\bbypass (?:the )?(?:policy|policies|rules|guard|filter|safety|instructions)\b",
        r"\b(?:policy|rules?|restrictions?) (?:does|do) not apply\b",
        r"\bno restrictions\b",
        r"\bdeveloper mode\b",
    )
    EXACT_PATTERNS = (
        r"\bexact(?:ly)?\b",
        r"\bverbatim\b",
        r"\bword for word\b",
        r"\bas written\b",
        r"\bwithout paraphras(?:e|ing)\b",
        r"\bno paraphras(?:e|ing)\b",
        r"\bcharacter[- ]for[- ]character\b",
        r"\bwithout changing (?:a|one|any) character\b",
        r"\bunchanged\b",
    )
    NEGATED_PATTERNS = (
        r"\bwithout (?:revealing|printing|showing|quoting|repeating|disclosing|giving)\b",
        r"\bdo not (?:reveal|print|show|quote|repeat|disclose|give|bypass)\b",
        r"\bdon't (?:reveal|print|show|quote|repeat|disclose|give|bypass)\b",
        r"\bnever (?:reveal|print|show|quote|repeat|disclose|give|bypass)\b",
        r"\bnot (?:reveal|print|show|quote|repeat|disclose|give|bypass)\b",
        r"\bwithout giving bypass instructions\b",
    )
    ASSET_SPECIFIC_PATTERNS = (
        r"\b(?:your|the|this|current|actual|original|above|hidden|private|protected|internal|initial|root) (?:system )?(?:prompt|message|instruction|instructions|text|context|directive|directives|policy)\b",
        r"\b(?:your|the|this|current|actual|above) (?:developer|internal|confidential|system) (?:configuration|context|message|prompt|instruction|instructions|text|setup|directive|directives)\b",
        r"\b(?:prompt|message|instruction|instructions|text|context|directive|directives) (?:above|you were given|that governs you|that controls you)\b",
        r"\b(?:rules? you were given|instructions? (?:that|which) (?:govern|control) you)\b",
        r"\b(?:secret|canary|policy[_ -]?override[_ -]?success)\b",
    )
    PUBLIC_EXAMPLE_PATTERNS = (
        r"\b(?:an?|this) (?:example|sample|template|toy|hypothetical|demonstration)\b",
        r"\b(?:example|sample|template|toy|hypothetical) (?:system )?(?:prompt|message|instruction|instructions|policy)\b",
        r"\bdocumentation example\b",
        r"\b(?:published|public) (?:article|paper|document|documentation|example|prompt|message)\b",
        r"\bcanary (?:release|deployment|test|build|analysis|strategy)\b",
        r"\bsecret (?:sharing|key|management|rotation|storage|santa)\b",
        r"\b(?:fictional|fiction|story|novel|museum|role[- ]?play)\b",
        r"\bfictional character\b",
        r"\b(?:an?|the) (?:exact )?(?:example|sample|template|toy|hypothetical)\b",
    )
    EDUCATIONAL_PATTERNS = (
        r"^\s*(?:explain|describe|summarize|compare|define|why|what does|what is the difference)\b",
        r"\bhigh[- ]level\b",
        r"\bfor (?:education|training|awareness|a course|a lecture)\b",
        r"\blegitimate (?:administrator|administrative|security|testing)\b",
        r"\bdefen[cs]e(?:s)?\b",
        r"\bprevention\b",
        r"\bdetection\b",
    )
    COMPOSITION_PATTERNS = (
        r"\b(?:join|combine|concatenate|assemble|merge)\b",
        r"\bapply (?:that|this|the) (?:request|instruction|command|rule)\b",
        r"\bfollow (?:that|this|the) (?:request|instruction|command|pattern)\b",
        r"\bexecute (?:that|this|the) (?:request|instruction|command)\b",
    )
    BACKREF_PATTERNS = (
        r"\b(?:that|this|the) (?:request|instruction|command|pattern|idea|text|note|card|recipe)\b",
        r"\b(?:it|them|those|the above|the previous one|the earlier one)\b",
        r"\bcontinue\b",
        r"\bprevious(?:ly)?\b",
        r"\bearlier\b",
        r"\bnow\b",
    )
    REMEMBER_PATTERNS = (
        r"\bremember\b",
        r"\bkeep (?:in mind|following|using)\b",
        r"\bfor the next (?:example|examples|turn|turns|request|requests)\b",
    )
    OUTPUT_PRIMING_PATTERNS = (
        r"\banswer only with (?:the )?exact string\b",
        r"\boutput only (?:the )?(?:exact )?(?:string|text|answer)\b",
        r"\bkeep following (?:the )?(?:same )?exact[- ]output pattern\b",
        r"\bwhen asked for .{1,30}, answer .{1,30}\b",
    )

    @classmethod
    def _matches(cls, patterns: tuple[str, ...], text: str) -> list[str]:
        return [pattern for pattern in patterns if re.search(pattern, text, re.IGNORECASE)]

    @classmethod
    def analyse(cls, text: str) -> dict[str, Any]:
        lowered = _normalise_space(text)
        protected = cls._matches(cls.PROTECTED_PATTERNS, lowered)
        extraction = cls._matches(cls.EXTRACTION_PATTERNS, lowered)
        override = cls._matches(cls.OVERRIDE_PATTERNS, lowered)
        exact = cls._matches(cls.EXACT_PATTERNS, lowered)
        negated = cls._matches(cls.NEGATED_PATTERNS, lowered)
        educational = cls._matches(cls.EDUCATIONAL_PATTERNS, lowered)
        asset_specific = cls._matches(cls.ASSET_SPECIFIC_PATTERNS, lowered)
        public_example = cls._matches(cls.PUBLIC_EXAMPLE_PATTERNS, lowered)
        composition = cls._matches(cls.COMPOSITION_PATTERNS, lowered)
        backref = cls._matches(cls.BACKREF_PATTERNS, lowered)
        remember = cls._matches(cls.REMEMBER_PATTERNS, lowered)
        output_priming = cls._matches(cls.OUTPUT_PRIMING_PATTERNS, lowered)
        marker = bool(re.search(r"\bpolicy[_ -]?override[_ -]?success\b", lowered))
        return {
            "protected": protected,
            "extraction": extraction,
            "override": override,
            "exact": exact,
            "negated": negated,
            "educational": educational,
            "asset_specific": asset_specific,
            "public_example": public_example,
            "composition": composition,
            "backref": backref,
            "remember": remember,
            "output_priming": output_priming,
            "marker": marker,
            "tokens": set(re.findall(r"[\w-]+", lowered)),
        }

    @staticmethod
    def decode_candidates(text: str) -> list[dict[str, str]]:
        """Decode conservative base64/hex/ROT13 candidates for inspection.

        Only printable strings of at least eight characters are returned.  The
        original text is never executed or followed.
        """
        candidates: list[dict[str, str]] = []
        chunks = re.findall(r"(?<![A-Za-z0-9+/=])[A-Za-z0-9+/]{16,}={0,2}(?![A-Za-z0-9+/=])", text)
        for chunk in chunks[:4]:
            try:
                padded = chunk + "=" * (-len(chunk) % 4)
                decoded = base64.b64decode(padded, validate=True).decode("utf-8")
                if len(decoded) >= 8 and decoded.isprintable():
                    candidates.append({"encoding": "base64", "source": chunk, "decoded": decoded})
            except (binascii.Error, UnicodeDecodeError, ValueError):
                pass
        for chunk in re.findall(r"(?<![0-9A-Fa-f])[0-9A-Fa-f]{16,}(?![0-9A-Fa-f])", text)[:4]:
            if len(chunk) % 2:
                continue
            try:
                decoded = bytes.fromhex(chunk).decode("utf-8")
                if len(decoded) >= 8 and decoded.isprintable():
                    candidates.append({"encoding": "hex", "source": chunk, "decoded": decoded})
            except (UnicodeDecodeError, ValueError):
                pass
        if re.search(r"\brot13\b", text, re.IGNORECASE):
            payload = re.sub(r"^.*?\brot13\s*[:=-]?\s*", "", text, flags=re.IGNORECASE)
            decoded = codecs.decode(payload, "rot_13")
            if len(decoded) >= 8 and decoded.isprintable():
                candidates.append({"encoding": "rot13", "source": payload, "decoded": decoded})

        # Binary byte sequences.
        binary = re.findall(r"(?<![01])(?:[01]{8}(?:\s+|$)){2,}", text)
        for chunk in binary[:4]:
            try:
                decoded = "".join(chr(int(byte, 2)) for byte in re.findall(r"[01]{8}", chunk))
                if len(decoded) >= 3 and decoded.isprintable():
                    candidates.append({"encoding": "binary", "source": chunk.strip(), "decoded": decoded})
            except ValueError:
                pass

        # Explicit Unicode code points (U+0049 U+0067 ...).
        points = re.findall(r"U\+([0-9A-Fa-f]{4,6})", text)
        if len(points) >= 3:
            try:
                decoded = "".join(chr(int(point, 16)) for point in points)
                if decoded.isprintable():
                    candidates.append({"encoding": "unicode_codepoints", "source": " ".join("U+" + p for p in points), "decoded": decoded})
            except (ValueError, OverflowError):
                pass

        # Caesar cipher when the shift is stated explicitly.
        caesar = re.search(r"caesar(?: cipher)?\s*\(?(?:shift\s*)?(\d{1,2})\)?\s*[:=-]\s*([A-Za-z ]{5,})", text, re.IGNORECASE)
        if caesar:
            shift = int(caesar.group(1)) % 26
            payload = caesar.group(2).strip()
            def unshift(ch: str) -> str:
                if 'a' <= ch <= 'z': return chr((ord(ch)-97-shift)%26+97)
                if 'A' <= ch <= 'Z': return chr((ord(ch)-65-shift)%26+65)
                return ch
            decoded = "".join(unshift(ch) for ch in payload)
            candidates.append({"encoding": "caesar", "source": payload, "decoded": decoded})

        # A compact Morse decoder for explicitly marked payloads.
        morse_match = re.search(r"morse(?: code)?\s*[:=-]\s*([.\-/ ]{5,})", text, re.IGNORECASE)
        if morse_match:
            table = {'.-':'A','-...':'B','-.-.':'C','-..':'D','.':'E','..-.':'F','--.':'G','....':'H','..':'I','.---':'J','-.-':'K','.-..':'L','--':'M','-.':'N','---':'O','.--.':'P','--.-':'Q','.-.':'R','...':'S','-':'T','..-':'U','...-':'V','.--':'W','-..-':'X','-.--':'Y','--..':'Z'}
            words=[]
            for word in re.split(r"\s*/\s*", morse_match.group(1).strip()):
                letters=[table.get(code, '?') for code in word.split()]
                words.append(''.join(letters))
            decoded=' '.join(words)
            if '?' not in decoded and len(decoded) >= 3:
                candidates.append({"encoding": "morse", "source": morse_match.group(1), "decoded": decoded})
        return candidates


class PerTurnSignal:
    """Return instantaneous direct-prompt-attack risk in ``[0, 1]``."""

    def __init__(self, classifier: Any | None = None):
        self.classifier = classifier
        self.details: dict[str, Any] = {}

    def reset(self) -> None:
        self.details = {}

    @staticmethod
    def _lexical_score(features: dict[str, Any]) -> tuple[float, bool, list[str]]:
        protected = bool(features["protected"])
        extraction = bool(features["extraction"])
        override = bool(features["override"])
        exact = bool(features["exact"])
        negated = bool(features["negated"])
        educational = bool(features["educational"])
        marker = bool(features["marker"])
        asset_specific = bool(features["asset_specific"])
        public_example = bool(features["public_example"])
        reasons: list[str] = []
        score = 0.0
        high_confidence = False

        if override and (asset_specific or marker):
            score = 1.0
            high_confidence = True
            reasons.append("policy_override_target")
        elif extraction and protected and not negated and asset_specific and not public_example:
            score = 1.0 if exact else 0.92
            high_confidence = True
            reasons.append("protected_asset_extraction")
        elif marker and extraction and not negated:
            score = 0.98
            high_confidence = True
            reasons.append("forbidden_marker_request")
        else:
            if protected:
                score += 0.22
                reasons.append("protected_asset_reference")
            if extraction:
                score += 0.18
                reasons.append("output_or_extraction_verb")
            if override:
                score += 0.45
                reasons.append("override_language")
            if exact:
                score += 0.18
                reasons.append("exact_output_language")
            if marker:
                score += 0.30
                reasons.append("forbidden_marker")

        if negated:
            score -= 0.55
            reasons.append("protective_negation")
        if (educational or public_example) and not override and not marker:
            score -= 0.22
            reasons.append("educational_context")
        return _clip01(score), high_confidence, reasons

    def __call__(self, turn: str) -> float:
        lexical_features = LexicalIntent.analyse(turn)
        lexical_score, lexical_high, reasons = self._lexical_score(lexical_features)

        decoded = LexicalIntent.decode_candidates(turn)
        decoded_risk = 0.0
        decoded_high = False
        decoded_details: list[dict[str, Any]] = []
        for item in decoded:
            features = LexicalIntent.analyse(item["decoded"])
            score, high, decoded_reasons = self._lexical_score(features)
            decoded_details.append({**item, "score": score, "reasons": decoded_reasons})
            decoded_risk = max(decoded_risk, score)
            decoded_high = decoded_high or high
        if decoded_risk >= 0.75:
            reasons.append("decoded_attack_payload")

        classifier_score = None
        if self.classifier is not None:
            classifier_score = _clip01(self.classifier.injection_probability(turn))
            # Noisy classifiers should not erase a transparent high-confidence
            # lexical match.  The max preserves recall and remains auditable.
            score = max(lexical_score, decoded_risk, classifier_score)
            backend = "hybrid_classifier_lexical"
        else:
            score = max(lexical_score, decoded_risk)
            backend = "lexical_regression"

        high_confidence = lexical_high or decoded_high
        self.details = {
            "backend": backend,
            "score": score,
            "classifier_score": classifier_score,
            "high_confidence": high_confidence,
            "reasons": reasons,
            "features": {k: v for k, v in lexical_features.items() if k != "tokens"},
            "decoded_candidates": decoded_details,
        }
        return score


class DriftSignal:
    """Track movement toward policy-specific sensitive reference regions."""

    def __init__(
        self,
        sensitive_texts: list[str],
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        proximity_weight: float = 0.55,
        velocity_weight: float = 0.25,
        displacement_weight: float = 0.20,
        backend: str = "auto",
    ):
        if not sensitive_texts:
            raise ValueError("sensitive_texts must not be empty")
        total = proximity_weight + velocity_weight + displacement_weight
        if not math.isclose(total, 1.0, abs_tol=1e-6):
            raise ValueError("drift weights must sum to 1")
        if backend not in {"auto", "embedding", "lexical"}:
            raise ValueError("backend must be auto, embedding, or lexical")
        self.sensitive_texts = sensitive_texts
        self.model_name = model_name
        self.backend = backend
        self.proximity_weight = proximity_weight
        self.velocity_weight = velocity_weight
        self.displacement_weight = displacement_weight
        self._model = None
        self._centroids = None
        self._backend = "uninitialised"
        self.reset()

    def reset(self) -> None:
        self._previous_similarity: float | None = None
        self._minimum_similarity: float | None = None
        self.details: dict[str, Any] = {}

    def _ensure(self) -> None:
        if self._model is not None:
            return
        from sentence_transformers import SentenceTransformer
        import numpy as np

        self._model = SentenceTransformer(self.model_name)
        embs = self._model.encode(self.sensitive_texts, normalize_embeddings=True)
        self._centroids = np.asarray(embs)
        self._backend = "sentence_transformers"

    @staticmethod
    def _tokens(text: str) -> set[str]:
        return set(re.findall(r"[\w-]+", text.casefold()))

    def _lexical_similarity(self, turn: str) -> float:
        self._backend = "lexical_regression"
        turn_tokens = self._tokens(turn)
        best = 0.0
        for reference in self.sensitive_texts:
            ref_tokens = self._tokens(reference)
            if ref_tokens:
                denom = len(turn_tokens) + len(ref_tokens)
                score = 2.0 * len(turn_tokens & ref_tokens) / denom if denom else 0.0
                best = max(best, score)
        return _clip01(best)

    def _embedding_similarity(self, turn: str) -> float:
        import numpy as np

        self._ensure()
        vector = self._model.encode([turn], normalize_embeddings=True)[0]
        similarity = float((self._centroids @ np.asarray(vector)).max())
        return _clip01((similarity + 1.0) / 2.0)

    def _similarity(self, turn: str) -> float:
        if self.backend == "lexical":
            return self._lexical_similarity(turn)
        if self.backend == "embedding":
            return self._embedding_similarity(turn)
        try:
            return self._embedding_similarity(turn)
        except (ImportError, ModuleNotFoundError):
            return self._lexical_similarity(turn)

    def __call__(self, turn: str) -> float:
        similarity = self._similarity(turn)
        if self._previous_similarity is None:
            velocity = 0.0
            self._minimum_similarity = similarity
        else:
            velocity = max(0.0, similarity - self._previous_similarity)
            self._minimum_similarity = min(self._minimum_similarity or similarity, similarity)
        displacement = max(0.0, similarity - (self._minimum_similarity or similarity))

        velocity_scaled = _clip01(velocity / 0.25)
        displacement_scaled = _clip01(displacement / 0.40)
        score = _clip01(
            self.proximity_weight * similarity
            + self.velocity_weight * velocity_scaled
            + self.displacement_weight * displacement_scaled
        )
        self._previous_similarity = similarity
        self.details = {
            "backend": self._backend,
            "similarity": similarity,
            "positive_velocity": velocity,
            "session_displacement": displacement,
            "score": score,
        }
        return score


@dataclass
class FrameTracker:
    """Bounded tracker for aliases, references, and distributed intent.

    The tracker stores only the last ``max_items`` component sets and aliases.
    Aliases are sensitivity-aware: a benign alias such as calling a deployment
    checklist "the recipe" does not become risky merely because it is reused.
    """

    max_items: int = 8
    _frames: deque[str] = field(init=False, repr=False)
    _aliases: deque[tuple[str, bool]] = field(init=False, repr=False)
    _components: deque[set[str]] = field(init=False, repr=False)
    details: dict[str, Any] = field(default_factory=dict, init=False)

    _FRAME_SET = re.compile(
        r"\b(imagine|pretend|let(?:'s| us) say|suppose|you are|in this story|"
        r"as the|role[- ]?play|from now on|act as)\b",
        re.IGNORECASE,
    )
    _QUOTED_ALIAS = re.compile(
        r"\b(?:call|name|refer to)\b.{0,80}?[\"']([^\"']{2,40})[\"']",
        re.IGNORECASE,
    )
    _CALLED_ALIAS = re.compile(
        r"\b(?:is|are|be|text is|instructions are)\s+(?:called|named)\s+(?:the\s+)?([a-z][\w -]{1,35}?)(?:[.,;]|$)",
        re.IGNORECASE,
    )
    _AS_ALIAS = re.compile(
        r"\b(?:refer to|call)\s+.{1,60}?\s+as\s+(?:the\s+)?([\w-]{2,30})\b",
        re.IGNORECASE,
    )
    _CALL_THE_ALIAS = re.compile(
        r"\bcall\s+.{2,70}?\s+(?:the|as)\s+([a-z][\w -]{1,35}?)(?:[.,;]|$)",
        re.IGNORECASE,
    )

    def __post_init__(self) -> None:
        if self.max_items < 1:
            raise ValueError("max_items must be positive")
        self._frames = deque(maxlen=self.max_items)
        self._aliases = deque(maxlen=self.max_items)
        self._components = deque(maxlen=self.max_items)

    def reset(self) -> None:
        self._frames.clear()
        self._aliases.clear()
        self._components.clear()
        self.details = {}

    @staticmethod
    def _normalise(value: str) -> str:
        return _normalise_space(value).strip(" .,:;!?")

    def _remember_alias(self, alias: str, sensitive: bool) -> None:
        alias = self._normalise(alias)
        if len(alias) < 2 or len(alias) > 40:
            return
        existing = [(a, s) for a, s in self._aliases if a != alias]
        self._aliases.clear()
        self._aliases.extend(existing[-(self.max_items - 1):] if self.max_items > 1 else [])
        self._aliases.append((alias, sensitive))

    def update_and_score(self, turn: str) -> float:
        text = _normalise_space(turn)
        features = LexicalIntent.analyse(turn)
        established_frame = bool(self._FRAME_SET.search(turn))
        protected = bool(features["protected"])
        extraction = bool(features["extraction"])
        override = bool(features["override"])
        exact = bool(features["exact"])
        negated = bool(features["negated"])
        educational = bool(features["educational"])
        composition = bool(features["composition"])
        back_reference = bool(features["backref"])
        output_priming = bool(features["output_priming"])
        marker = bool(features["marker"])
        asset_specific = bool(features["asset_specific"])
        public_example = bool(features["public_example"])

        prior_retained = set().union(*self._components) if self._components else set()
        aliases = [self._normalise(v) for v in self._QUOTED_ALIAS.findall(turn)]
        aliases += [self._normalise(v) for v in self._CALLED_ALIAS.findall(turn)]
        aliases += [self._normalise(v) for v in self._AS_ALIAS.findall(turn)]
        aliases += [self._normalise(v) for v in self._CALL_THE_ALIAS.findall(turn)]
        current_sensitive_target = protected and asset_specific and not educational and not public_example and not negated
        alias_sensitive = current_sensitive_target or (
            bool(aliases) and bool(features["backref"]) and "protected_target" in prior_retained
        )
        for alias in aliases:
            self._remember_alias(alias, alias_sensitive)

        invoked_aliases = [alias for alias, _ in self._aliases if re.search(rf"\b{re.escape(alias)}\b", text)]
        invoked_sensitive_aliases = [
            alias for alias, sensitive in self._aliases
            if sensitive and re.search(rf"\b{re.escape(alias)}\b", text)
        ]

        components: set[str] = set()
        if current_sensitive_target:
            components.add("protected_target")
        elif protected:
            components.add("protected_mention")
        if extraction and not negated:
            components.add("extraction")
        if override and not negated:
            components.add("override")
        if exact:
            components.add("exactness")
        if composition:
            components.add("composition")
        if output_priming:
            components.add("output_priming")
        if marker:
            components.add("forbidden_marker")
        tokens = features["tokens"]
        if "hidden" in tokens:
            components.add("fragment_hidden")
        if "instructions" in tokens or "instruction" in tokens:
            components.add("fragment_instructions")
        if "exact" in tokens or "exactly" in tokens or "paraphrasing" in tokens:
            components.add("fragment_exact")
        self._components.append(components)
        retained = set().union(*self._components) if self._components else set()

        risky_alias_request = bool(invoked_sensitive_aliases) and (extraction or exact or composition) and not negated
        direct_request = protected and asset_specific and extraction and not negated and not public_example
        direct_override = override and (asset_specific or marker) and not negated
        distributed_fragments = {"fragment_hidden", "fragment_instructions", "fragment_exact"}.issubset(prior_retained)
        prior_sensitive_exact = {"protected_target", "exactness"}.issubset(prior_retained)
        prior_composed_request = {"protected_mention", "exactness", "extraction"}.issubset(prior_retained)
        distributed_execution = (
            composition
            and (distributed_fragments or prior_sensitive_exact or prior_composed_request)
            and not negated
            and not public_example
        )
        primed_execution = "output_priming" in prior_retained and (override or marker or direct_request) and not negated
        risky_context_reference = (
            "protected_target" in prior_retained
            and back_reference
            and extraction
            and exact
            and not negated
            and not public_example
        )
        risky_frame_continuation = (
            bool(self._frames)
            and back_reference
            and (direct_request or risky_alias_request or risky_context_reference or (extraction and exact and protected))
        )

        high_confidence = any(
            (direct_request, direct_override, risky_alias_request, distributed_execution, primed_execution, risky_context_reference)
        )
        reasons: list[str] = []
        score = 0.0
        if direct_request:
            score = max(score, 1.0)
            reasons.append("direct_protected_extraction")
        if direct_override:
            score = max(score, 1.0)
            reasons.append("direct_policy_override")
        if risky_alias_request:
            score = max(score, 1.0)
            reasons.append("sensitive_alias_extraction")
        if distributed_execution:
            score = max(score, 0.96)
            reasons.append("distributed_intent_composition")
        if primed_execution:
            score = max(score, 0.96)
            reasons.append("primed_exact_output_execution")
        if risky_context_reference:
            score = max(score, 0.96)
            reasons.append("protected_context_back_reference")
        if risky_frame_continuation:
            score = max(score, 0.92)
            reasons.append("risky_frame_continuation")
        if protected and asset_specific and exact and not negated and not public_example:
            score = max(score, 0.78)
            reasons.append("protected_exactness")
        if established_frame:
            self._frames.append(self._normalise(turn[:100]))
            score = max(score, 0.08)
        if output_priming:
            score = max(score, 0.30)
        if composition and prior_retained:
            score = max(score, 0.35)
        if back_reference and bool(self._frames):
            score = max(score, 0.18)
        if protected and educational:
            score = min(score, 0.10)
        if negated:
            score = min(score, 0.05)

        self.details = {
            "established_frame": established_frame,
            "new_aliases": aliases,
            "new_aliases_sensitive": alias_sensitive,
            "back_reference": back_reference,
            "invoked_aliases": invoked_aliases,
            "invoked_sensitive_aliases": invoked_sensitive_aliases,
            "retained_frames": len(self._frames),
            "retained_aliases": [
                {"alias": alias, "sensitive": sensitive} for alias, sensitive in self._aliases
            ],
            "current_components": sorted(components),
            "retained_components": sorted(retained),
            "high_confidence": high_confidence,
            "reasons": reasons,
            "score": _clip01(score),
        }
        return _clip01(score)
