"""Metrics for single-turn and conversation-level evaluation."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable


def proportion(successes: int, total: int) -> float:
    if total < 0 or successes < 0 or successes > total:
        raise ValueError("require 0 <= successes <= total")
    return successes / total if total else 0.0


def asr(successes: int, total: int) -> float:
    return proportion(successes, total)


def fpr(false_blocks: int, benign_total: int) -> float:
    return proportion(false_blocks, benign_total)


def aggregate_asr_counts(per_category: dict[str, tuple[int, int]]) -> float:
    """Micro-average ASR from ``category -> (successes, total)``."""
    successes = sum(value[0] for value in per_category.values())
    total = sum(value[1] for value in per_category.values())
    return proportion(successes, total)


def macro_average(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def latency_factor(latency_ms: float, tau_ms: float = 500.0) -> float:
    if latency_ms < 0 or tau_ms <= 0:
        raise ValueError("latency_ms must be non-negative and tau_ms positive")
    return math.exp(-latency_ms / tau_ms)


def practical_defense_score(
    attack_success_rate: float,
    false_positive_rate: float,
    latency_ms: float,
    measured_utility: float,
    tau_ms: float = 500.0,
) -> float:
    """Quality-aware score using an empirically measured utility factor.

    ``measured_utility`` must come from a declared benign-task evaluation; it
    must not be assigned by defense type.
    """
    for name, value in {
        "attack_success_rate": attack_success_rate,
        "false_positive_rate": false_positive_rate,
        "measured_utility": measured_utility,
    }.items():
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be in [0, 1]")
    return (
        (1.0 - attack_success_rate)
        * (1.0 - false_positive_rate)
        * latency_factor(latency_ms, tau_ms)
        * measured_utility
    )


# Backward-compatible name, with corrected semantics in the docstring.
def pds(asr_value: float, fpr_value: float, latency_ms: float, quality: float, tau_ms: float = 500.0) -> float:
    return practical_defense_score(asr_value, fpr_value, latency_ms, quality, tau_ms)


def su_ratio(asr_value: float, fpr_value: float) -> float:
    """Security-utility ratio retained for compatibility; report both inputs."""
    return (1.0 - asr_value) / (1.0 + fpr_value)


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n < 0 or k < 0 or k > n:
        raise ValueError("require 0 <= k <= n")
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


@dataclass(frozen=True)
class ProxyReport:
    proxy: str
    model: str
    agg_asr: float
    fpr: float
    latency_ms: float
    measured_utility: float

    @property
    def su(self) -> float:
        return su_ratio(self.agg_asr, self.fpr)

    @property
    def pds(self) -> float:
        return practical_defense_score(
            self.agg_asr, self.fpr, self.latency_ms, self.measured_utility
        )
