"""Detector-only and end-to-end evaluation for multi-turn prompt attacks.

A detector miss is *not* an attack success.  The module therefore reports two
separate quantities:

* detector escape rate (DER): attack sessions not blocked by the guard;
* chain-ASR: sessions that actually make the target LLM emit a protected marker.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable

from pinj.canary import Secrets, default_system_prompt, is_success
from .accumulator import Action
from .detector import ChainGuard, TurnVerdict
from .output_guard import OutputGuard


@dataclass(frozen=True)
class SessionSpec:
    session_id: str
    category: str
    turns: list[str]
    completion_turn: int | None = None


@dataclass
class SessionOutcome:
    blocked: bool
    detection_turn: int | None
    first_challenge_turn: int | None
    n_turns: int
    processed_turns: int
    max_score: float
    verdicts: list[TurnVerdict]
    attack_success: bool | None = None
    responses: list[str] | None = None
    raw_attack_success: bool | None = None
    output_blocked: bool = False
    output_block_turn: int | None = None
    output_assessments: list[dict[str, Any]] | None = None


def _coerce_session(raw: SessionSpec | dict[str, Any] | list[str], index: int) -> SessionSpec:
    if isinstance(raw, SessionSpec):
        return raw
    if isinstance(raw, list):
        return SessionSpec(f"session-{index:04d}", "unspecified", raw)
    return SessionSpec(
        session_id=str(raw.get("id") or raw.get("session_id") or f"session-{index:04d}"),
        category=str(raw.get("category") or raw.get("label") or "unspecified"),
        turns=[str(v) for v in raw["turns"]],
        completion_turn=int(raw["completion_turn"]) if raw.get("completion_turn") else None,
    )


def run_detector_session(
    guard: ChainGuard,
    turns: list[str],
    stop_on_block: bool = True,
) -> SessionOutcome:
    guard.reset()
    detection_turn = None
    first_challenge = None
    max_score = 0.0
    verdicts: list[TurnVerdict] = []
    for turn in turns:
        verdict = guard.step(turn)
        verdicts.append(verdict)
        max_score = max(max_score, verdict.score)
        if verdict.action == Action.SOFT_CHALLENGE and first_challenge is None:
            first_challenge = verdict.turn_index
        if verdict.action == Action.BLOCK and detection_turn is None:
            detection_turn = verdict.turn_index
            if stop_on_block:
                break
    return SessionOutcome(
        blocked=detection_turn is not None,
        detection_turn=detection_turn,
        first_challenge_turn=first_challenge,
        n_turns=len(turns),
        processed_turns=len(verdicts),
        max_score=max_score,
        verdicts=verdicts,
    )


def evaluate_detector(
    guard_factory: Callable[[], ChainGuard],
    attack_sessions: list[SessionSpec | dict[str, Any] | list[str]],
    benign_sessions: list[SessionSpec | dict[str, Any] | list[str]],
) -> dict[str, Any]:
    attack_outcomes = []
    ttds: list[int] = []
    for index, raw in enumerate(attack_sessions):
        spec = _coerce_session(raw, index)
        outcome = run_detector_session(guard_factory(), spec.turns)
        if outcome.detection_turn is not None:
            ttds.append(outcome.detection_turn)
        attack_outcomes.append({"session": asdict(spec), "outcome": _serialise_outcome(outcome)})

    benign_outcomes = []
    for index, raw in enumerate(benign_sessions):
        spec = _coerce_session(raw, index)
        outcome = run_detector_session(guard_factory(), spec.turns)
        benign_outcomes.append({"session": asdict(spec), "outcome": _serialise_outcome(outcome)})

    n_att = len(attack_outcomes)
    n_ben = len(benign_outcomes)
    blocked_attacks = sum(1 for item in attack_outcomes if item["outcome"]["blocked"])
    blocked_benign = sum(1 for item in benign_outcomes if item["outcome"]["blocked"])
    challenged_benign = sum(
        1 for item in benign_outcomes if item["outcome"]["first_challenge_turn"] is not None
    )
    return {
        "metric_semantics": {
            "detector_escape_rate": "fraction of labelled attack sessions not blocked; not chain-ASR",
            "session_block_fpr": "fraction of benign sessions ever blocked",
        },
        "n_attack_sessions": n_att,
        "n_benign_sessions": n_ben,
        "attack_detection_rate": blocked_attacks / n_att if n_att else 0.0,
        "detector_escape_rate": 1.0 - blocked_attacks / n_att if n_att else 0.0,
        "ttd_distribution": ttds,
        "ttd_mean_detected_only": sum(ttds) / len(ttds) if ttds else None,
        "session_block_fpr": blocked_benign / n_ben if n_ben else 0.0,
        "session_challenge_rate": challenged_benign / n_ben if n_ben else 0.0,
        "attack_outcomes": attack_outcomes,
        "benign_outcomes": benign_outcomes,
    }


def run_end_to_end_session(
    guard: ChainGuard,
    turns: list[str],
    client: Any,
    model: str,
    secrets: Secrets,
    temperature: float = 0.0,
    seed: int = 42,
    output_guard: OutputGuard | None = None,
) -> SessionOutcome:
    """Run the actual guarded dialogue against a chat-capable client.

    Soft challenges are recorded but do not silently become blocks.  Their
    wording/interaction policy must be evaluated as a separate intervention.
    """
    guard.reset()
    messages: list[dict[str, str]] = [
        {"role": "system", "content": default_system_prompt(secrets)}
    ]
    responses: list[str] = []
    verdicts: list[TurnVerdict] = []
    detection_turn = None
    first_challenge = None
    success = False
    raw_success = False
    output_blocked = False
    output_block_turn = None
    output_assessments: list[dict[str, Any]] = []
    if output_guard is None:
        output_guard = OutputGuard(
            canaries=[secrets.canary, secrets.marker],
            protected_texts=[default_system_prompt(secrets)],
        )
    max_score = 0.0

    for turn in turns:
        verdict = guard.step(turn)
        verdicts.append(verdict)
        max_score = max(max_score, verdict.score)
        if verdict.action == Action.SOFT_CHALLENGE and first_challenge is None:
            first_challenge = verdict.turn_index
        if verdict.action == Action.BLOCK:
            detection_turn = verdict.turn_index
            break

        messages.append({"role": "user", "content": turn})
        response = client.chat(
            model=model,
            messages=messages,
            temperature=temperature,
            seed=seed,
        )
        raw_success = raw_success or is_success(response, secrets)
        output_assessment = output_guard.inspect(response)
        output_assessments.append(output_assessment.as_dict())
        if output_assessment.blocked:
            output_blocked = True
            output_block_turn = verdict.turn_index
            responses.append(output_assessment.redacted_output or "[BLOCKED]")
            break
        responses.append(response)
        messages.append({"role": "assistant", "content": response})
        if is_success(response, secrets):
            success = True
            break

    return SessionOutcome(
        blocked=detection_turn is not None,
        detection_turn=detection_turn,
        first_challenge_turn=first_challenge,
        n_turns=len(turns),
        processed_turns=len(verdicts),
        max_score=max_score,
        verdicts=verdicts,
        attack_success=success,
        responses=responses,
        raw_attack_success=raw_success,
        output_blocked=output_blocked,
        output_block_turn=output_block_turn,
        output_assessments=output_assessments,
    )


def evaluate_end_to_end(
    guard_factory: Callable[[], ChainGuard],
    attack_sessions: list[SessionSpec | dict[str, Any] | list[str]],
    client: Any,
    model: str,
    secrets: Secrets,
    temperature: float = 0.0,
    seed: int = 42,
) -> dict[str, Any]:
    outcomes = []
    for index, raw in enumerate(attack_sessions):
        spec = _coerce_session(raw, index)
        outcome = run_end_to_end_session(
            guard_factory(), spec.turns, client, model, secrets, temperature, seed
        )
        outcomes.append({"session": asdict(spec), "outcome": _serialise_outcome(outcome)})
    successes = sum(1 for item in outcomes if item["outcome"]["attack_success"])
    raw_successes = sum(1 for item in outcomes if item["outcome"].get("raw_attack_success"))
    output_blocks = sum(1 for item in outcomes if item["outcome"].get("output_blocked"))
    return {
        "model": model,
        "n_attack_sessions": len(outcomes),
        "chain_asr": successes / len(outcomes) if outcomes else 0.0,
        "raw_model_leak_rate": raw_successes / len(outcomes) if outcomes else 0.0,
        "output_guard_block_rate": output_blocks / len(outcomes) if outcomes else 0.0,
        "outcomes": outcomes,
    }


def _serialise_outcome(outcome: SessionOutcome) -> dict[str, Any]:
    data = asdict(outcome)
    for verdict in data["verdicts"]:
        verdict["action"] = verdict["action"].value if hasattr(verdict["action"], "value") else verdict["action"]
    return data


# Backwards-compatible alias with corrected semantics.
def evaluate(guard_factory, attack_sessions, benign_sessions):
    return evaluate_detector(guard_factory, attack_sessions, benign_sessions)
