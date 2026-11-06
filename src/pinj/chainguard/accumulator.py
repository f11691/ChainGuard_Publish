"""Interpretable sequential evidence accumulator for ChainGuard.

The score is a leaky one-sided CUSUM-like statistic:

    e_t = alpha*c_t + beta*d_t + gamma*r_t
    S_t = max(0, lambda*S_{t-1} + e_t - kappa + delta_t)

A separate high-confidence rule path in :mod:`pinj.chainguard.detector` handles
unambiguous direct extraction/override requests and decoded attack payloads.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
import math


class Action(str, Enum):
    ALLOW = "allow"
    SOFT_CHALLENGE = "soft_challenge"
    BLOCK = "block"


@dataclass(frozen=True)
class AccumulatorConfig:
    alpha: float = 0.45
    beta: float = 0.15
    gamma: float = 0.40
    lam: float = 0.80
    reference: float = 0.15
    esc_window: int = 3
    esc_bonus: float = 0.10
    min_escalation_step: float = 0.03
    tau_lo: float = 0.42
    tau_hi: float = 0.72

    def __post_init__(self) -> None:
        if not math.isclose(self.alpha + self.beta + self.gamma, 1.0, abs_tol=1e-6):
            raise ValueError("alpha, beta, and gamma must sum to 1")
        if not 0.0 <= self.lam < 1.0:
            raise ValueError("lam must be in [0, 1)")
        if self.reference < 0.0:
            raise ValueError("reference must be non-negative")
        if self.esc_window < 2:
            raise ValueError("esc_window must be at least 2")
        if not 0.0 <= self.tau_lo < self.tau_hi:
            raise ValueError("thresholds must satisfy 0 <= tau_lo < tau_hi")


@dataclass
class IntentAccumulator:
    cfg: AccumulatorConfig = field(default_factory=AccumulatorConfig)
    score: float = 0.0
    _drift_hist: deque[float] = field(init=False, repr=False)
    last_evidence: float = 0.0
    last_bonus: float = 0.0

    def __post_init__(self) -> None:
        self._drift_hist = deque(maxlen=self.cfg.esc_window)

    @property
    def S(self) -> float:
        return self.score

    def _escalation_bonus(self, d_t: float) -> float:
        self._drift_hist.append(float(d_t))
        if len(self._drift_hist) < self.cfg.esc_window:
            return 0.0
        values = list(self._drift_hist)
        rises = [values[i + 1] - values[i] for i in range(len(values) - 1)]
        monotonic = all(step >= self.cfg.min_escalation_step for step in rises)
        return self.cfg.esc_bonus if monotonic else 0.0

    def update(self, c_t: float, d_t: float, r_t: float) -> float:
        c, d, r = (max(0.0, min(1.0, float(x))) for x in (c_t, d_t, r_t))
        self.last_evidence = self.cfg.alpha * c + self.cfg.beta * d + self.cfg.gamma * r
        self.last_bonus = self._escalation_bonus(d)
        self.score = max(
            0.0,
            self.cfg.lam * self.score
            + self.last_evidence
            - self.cfg.reference
            + self.last_bonus,
        )
        return self.score

    def action(self) -> Action:
        if self.score >= self.cfg.tau_hi:
            return Action.BLOCK
        if self.score >= self.cfg.tau_lo:
            return Action.SOFT_CHALLENGE
        return Action.ALLOW

    def reset(self) -> None:
        self.score = 0.0
        self.last_evidence = 0.0
        self.last_bonus = 0.0
        self._drift_hist.clear()
