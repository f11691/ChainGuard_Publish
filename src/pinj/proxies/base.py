"""Proxy interface. A proxy inspects a user prompt and returns a decision."""
from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum


class Decision(str, Enum):
    ALLOW = "allow"
    BLOCK = "block"
    SANITIZE = "sanitize"


@dataclass
class ProxyResult:
    decision: Decision
    prompt: str            # possibly transformed prompt to forward to the LLM
    latency_ms: float
    meta: dict


class Proxy:
    """Base class. Subclasses implement `_run`."""

    name = "base"

    def _run(self, prompt: str) -> ProxyResult:  # pragma: no cover - abstract
        raise NotImplementedError

    def __call__(self, prompt: str) -> ProxyResult:
        t0 = time.perf_counter()
        res = self._run(prompt)
        # ensure latency is measured even if subclass didn't set it
        if res.latency_ms == 0.0:
            res.latency_ms = (time.perf_counter() - t0) * 1000.0
        return res
