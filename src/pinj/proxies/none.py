"""No-op baseline proxy: forwards the raw prompt."""
from __future__ import annotations
from .base import Proxy, ProxyResult, Decision


class NoneProxy(Proxy):
    name = "none"

    def _run(self, prompt: str) -> ProxyResult:
        return ProxyResult(Decision.ALLOW, prompt, 0.0, {})
