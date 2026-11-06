# Multi-turn session schema

Each JSON record should contain:

```json
{
  "id": "atk-gradual-001",
  "track": "direct_prompt_injection",
  "category": "gradual_escalation",
  "source": "human_adaptation_of_published_pattern",
  "source_template_id": "crescendo-inspired-01",
  "generator": "human",
  "target_asset": "system_canary",
  "completion_turn": 4,
  "turns": ["..."],
  "notes": "Harmless exact-match extraction task"
}
```

Required constraints:

- `id` is globally unique.
- `completion_turn` identifies the earliest turn at which the attack objective is fully specified.
- No real secret or fixed test canary appears in the corpus.
- `source_template_id` is used for template-disjoint splitting.
- Benign sessions use `track: "benign"` and omit `target_asset`.
- Generated descendants preserve the parent template ID.
