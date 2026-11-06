"""Single-turn benchmark runner: reproduces the benchmark methodology.

For each (proxy, attack) it applies the proxy, and if the prompt is not
blocked, queries the LLM and scores the response by exact match. Benign
prompts feed the FPR estimate. Results are returned as dicts and can be
persisted to SQLite via storage.py.

This is the harness; producing real numbers requires a running Ollama server
and (for classifier proxies) the HF models. Nothing here is pre-computed.
"""
from __future__ import annotations

from dataclasses import dataclass

from .canary import Secrets, default_system_prompt, is_success
from .proxies import get_proxy, Decision
from .llm.ollama_client import OllamaClient
from . import metrics


@dataclass
class RunConfig:
    model: str = "llama3.2"
    temperature: float = 0.0
    seed: int = 42
    measured_utility: float = 1.0  # must come from a declared benign-task evaluation


def evaluate_proxy(
    proxy_name: str,
    attacks: dict[str, list[str]],   # {category: [prompt, ...]}
    benign: list[str],
    sec: Secrets,
    cfg: RunConfig,
    client: OllamaClient | None = None,
) -> dict:
    """Return a report dict with per-category ASR, aggregate ASR, FPR, PDS."""
    client = client or OllamaClient()
    proxy = get_proxy(proxy_name)
    system = default_system_prompt(sec)

    per_cat_asr: dict[str, float] = {}
    per_cat_counts: dict[str, tuple[int, int]] = {}
    total_latency = 0.0
    n_proxy_calls = 0

    # --- attacks ---
    for cat, prompts in attacks.items():
        succ = 0
        for p in prompts:
            r = proxy(p)
            total_latency += r.latency_ms
            n_proxy_calls += 1
            if r.decision == Decision.BLOCK:
                continue  # blocked => attack fails
            resp = client.generate(
                model=cfg.model, system=system, prompt=r.prompt,
                temperature=cfg.temperature, seed=cfg.seed,
            )
            if is_success(resp, sec):
                succ += 1
        per_cat_asr[cat] = metrics.asr(succ, len(prompts))
        per_cat_counts[cat] = (succ, len(prompts))

    # --- benign (FPR) ---
    blocked = 0
    for p in benign:
        r = proxy(p)
        total_latency += r.latency_ms
        n_proxy_calls += 1
        if r.decision == Decision.BLOCK:
            blocked += 1
    fpr = metrics.fpr(blocked, len(benign))

    agg = metrics.aggregate_asr_counts(per_cat_counts)
    mean_latency = total_latency / n_proxy_calls if n_proxy_calls else 0.0
    return {
        "proxy": proxy_name,
        "model": cfg.model,
        "per_category_asr": per_cat_asr,
        "aggregate_asr": agg,
        "fpr": fpr,
        "mean_proxy_latency_ms": mean_latency,
        "su": metrics.su_ratio(agg, fpr),
        "pds": metrics.pds(agg, fpr, mean_latency, cfg.measured_utility),
    }
