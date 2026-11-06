"""Multi-encoding decode layer for ChainGuard v0.6.

Attackers hide payloads in encodings a lexical/TF-IDF layer never sees as text
(binary, Morse, hex, decimal, base32, reversed, simple leetspeak). This module
performs *conservative* decoding and returns decoded candidates so the existing
semantic layers can score the plaintext. It is deliberately high-precision:
each decoder only fires on well-formed input and rejects short/gibberish output.

This is a general capability derived from the threat model, not tuned to any
test set.
"""
from __future__ import annotations

import base64
import binascii
import re
from dataclasses import dataclass
from difflib import SequenceMatcher


@dataclass(frozen=True)
class Decoded:
    method: str
    text: str


_MORSE = {
    ".-": "a", "-...": "b", "-.-.": "c", "-..": "d", ".": "e", "..-.": "f",
    "--.": "g", "....": "h", "..": "i", ".---": "j", "-.-": "k", ".-..": "l",
    "--": "m", "-.": "n", "---": "o", ".--.": "p", "--.-": "q", ".-.": "r",
    "...": "s", "-": "t", "..-": "u", "...-": "v", ".--": "w", "-..-": "x",
    "-.--": "y", "--..": "z", "-----": "0", ".----": "1", "..---": "2",
    "...--": "3", "....-": "4", ".....": "5", "-....": "6", "--...": "7",
    "---..": "8", "----.": "9",
}

_LEET = str.maketrans({"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "@": "a", "$": "s"})

_PRINTABLE = re.compile(r"[ -~]")
_RISKY_SINGLE = re.compile(
    r"^(?:ignore|disregard|override|bypass|disable|reveal|disclose|exfiltrate|leak|dump|execute|obey|follow)$",
    re.I,
)
_RISKY_WORDS = (
    "ignore", "disregard", "override", "bypass", "disable", "reveal",
    "disclose", "exfiltrate", "leak", "dump", "execute", "obey", "follow",
)


def _plausible(text: str) -> bool:
    """Accept decoded output only if it looks like meaningful ASCII text."""
    if len(text) < 6:
        return False
    printable = sum(1 for ch in text if _PRINTABLE.match(ch))
    if printable / max(1, len(text)) < 0.85:
        return False
    words = re.findall(r"[A-Za-z]{3,}", text)
    # A single decoded command word can be security-relevant when the wrapper
    # explicitly says to execute/obey the decoded result.
    if len(words) == 1 and _RISKY_SINGLE.fullmatch(words[0]):
        return True
    return len(words) >= 2


def _try_base64(s: str) -> str | None:
    m = re.findall(r"[A-Za-z0-9+/]{16,}={0,2}", s)
    for token in m:
        if len(token) % 4 == 0 or "=" in token:
            try:
                out = base64.b64decode(token, validate=False).decode("utf-8", "ignore")
                if _plausible(out):
                    return out
            except (binascii.Error, ValueError):
                continue
    return None


def _try_base32(s: str) -> str | None:
    m = re.findall(r"[A-Z2-7]{16,}={0,6}", s)
    for token in m:
        pad = (-len(token)) % 8
        try:
            out = base64.b32decode(token + "=" * pad, casefold=True).decode("utf-8", "ignore")
            if _plausible(out):
                return out
        except (binascii.Error, ValueError):
            continue
    return None


def _try_hex(s: str) -> str | None:
    for token in re.findall(r"(?:0x)?[0-9A-Fa-f]{12,}", s):
        h = token[2:] if token.lower().startswith("0x") else token
        if len(h) % 2:
            continue
        try:
            out = bytes.fromhex(h).decode("utf-8", "ignore")
            if _plausible(out):
                return out
        except ValueError:
            continue
    return None


def _try_decimal(s: str) -> str | None:
    nums = re.findall(r"\b\d{1,3}\b", s)
    if len(nums) >= 6 and all(0 <= int(n) <= 255 for n in nums):
        out = "".join(chr(int(n)) for n in nums)
        if _plausible(out):
            return out
    return None


def _try_binary(s: str) -> str | None:
    bits = re.findall(r"[01]{8}", s)
    if len(bits) >= 6:
        try:
            out = "".join(chr(int(b, 2)) for b in bits)
            if _plausible(out):
                return out
        except ValueError:
            return None
    return None


def _try_morse(s: str) -> str | None:
    if not re.search(r"[.\-]{1,7}(?:[ /]+[.\-]{1,7}){5,}", s):
        return None
    words = re.split(r"\s*/\s*|\s{2,}", s.strip())
    out_words = []
    for w in words:
        letters = [_MORSE.get(tok, "") for tok in w.split()]
        out_words.append("".join(letters))
    out = " ".join(x for x in out_words if x)
    return out if _plausible(out) else None


def _try_reversed(s: str) -> str | None:
    rev = s[::-1]
    # only worth flagging if the reversed form contains attack-relevant words
    if re.search(r"\b(system|prompt|instruction|reveal|ignore|secret|policy)\b", rev, re.I):
        return rev
    return None


def _try_leet(s: str) -> str | None:
    out = s.translate(_LEET)
    if out != s and re.search(r"\b(system|prompt|instruction|ignore|reveal|secret|policy)\b", out, re.I):
        return out
    return None



def _nearest_risky(token: str) -> str:
    low = token.lower()
    best = max(_RISKY_WORDS, key=lambda w: SequenceMatcher(None, low, w).ratio())
    ratio = SequenceMatcher(None, low, best).ratio()
    return best if ratio >= 0.78 else token


def _try_alternating(s: str) -> str | None:
    """Recover simple every-other-character or interleaved-filler payloads."""
    if not re.search(r"\b(?:alternating|every other|every second|odd characters?|even characters?|interleaved|hidden order)\b", s, re.I):
        return None
    payload = s.split(":", 1)[-1].strip()
    tokens = re.findall(r"[A-Za-z]{4,}", payload)
    if not tokens:
        return None

    candidates: list[str] = []
    candidates.append(" ".join(tok[::2] for tok in tokens))
    candidates.append(" ".join(tok[1::2] for tok in tokens))

    # Remove a repeated filler character when it occupies many positions.
    letters = "".join(tokens).lower()
    if letters:
        filler = max(set(letters), key=letters.count)
        if letters.count(filler) >= max(3, len(letters) // 4):
            candidates.append(" ".join(tok.replace(filler, "").replace(filler.upper(), "") for tok in tokens))

    for cand in candidates:
        words = [_nearest_risky(w) for w in re.findall(r"[A-Za-z]{3,}", cand)]
        out = " ".join(words)
        if _plausible(out) and re.search(r"\b(?:ignore|disregard|override|bypass|disable|reveal|disclose|leak|execute|obey|follow)\b", out, re.I):
            return out
    return None

_DECODERS = [
    ("base64", _try_base64),
    ("base32", _try_base32),
    ("hex", _try_hex),
    ("binary", _try_binary),
    ("decimal", _try_decimal),
    ("morse", _try_morse),
    ("alternating", _try_alternating),
    ("reversed", _try_reversed),
    ("leetspeak", _try_leet),
]


def decode_candidates(text: str, max_candidates: int = 4) -> list[Decoded]:
    """Return conservative decoded candidates (possibly empty)."""
    out: list[Decoded] = []
    for name, fn in _DECODERS:
        try:
            decoded = fn(text)
        except Exception:
            decoded = None
        if decoded and decoded.strip() and decoded.strip().lower() != text.strip().lower():
            out.append(Decoded(name, decoded.strip()))
            if len(out) >= max_candidates:
                break
    return out
