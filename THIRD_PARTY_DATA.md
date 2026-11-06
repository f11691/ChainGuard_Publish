# Third-party data

## `data/external/zachz_prompt_injection_benchmark.csv`

- Upstream dataset: `zachz/prompt-injection-benchmark`
- Upstream-declared license: MIT
- Local SHA-256: `214f58813b0dd6be93a198037be64ccb56c87c2d6af14719b8e233865ad38ba2`
- Local rows: 303
- Use: development/training and diagnostic evaluation

This dataset was inspected and used to modify detector behavior. Its score is development-only.

## `data/external/piguard/BIPIA_text.json`

- Upstream repository: `leolee99/PIGuard`
- Upstream file: `datasets/BIPIA_text.json`
- Upstream Git blob SHA reported during retrieval: `259e58a7aaf696ecddbf664022c0f254318b0a05`
- Local SHA-256: `c9e66d9dcf4e7c12cacbfaf7e8a95e0f650316c07cbc02912df2ea4e476ffe48`
- Local prompts: 75
- Upstream repository license: MIT
- Use: source-aware provenance development and diagnostic evaluation

BIPIA failures directly guided the strict retrieved/tool provenance rule. Therefore, the recorded BIPIA score is development-only, even when evaluated after the final source freeze. It must not be presented as an independent post-freeze result.

Before redistribution or publication, verify the current upstream license and preserve attribution.
