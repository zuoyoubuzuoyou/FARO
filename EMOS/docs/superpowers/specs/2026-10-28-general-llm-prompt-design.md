# General LLM monitoring prompt

The user selected prompt-only generalization: keep the existing input format and runtime interfaces. Future benchmarks should supply their task meaning through observations rather than inherit EMOS task assumptions.

## Approaches and scope

We considered extracting a generic detector and an EMOS observation adapter, or generalizing only the prompt. The user chose the latter for this change. Only `SYSTEM_PROMPT`, `PROMPT_VERSION`, and explanatory documentation change; the current reader, schema, sampling, validation, transport and result aggregation remain the existing implementation. This is not a claim that arbitrary benchmark logs are now accepted directly.

## Prompt behavior

- Use a task-execution monitoring role applicable to one or more agents, without named benchmarks, robot-specific task classes, action names or native predicate names.
- Derive success conditions, responsibilities, required ordering, simultaneous/current versus historical completion and capability constraints from the supplied task and observations.
- Distinguish clear decision/planning/capability problems from execution problems. Persistent unproductive waiting, repeated idle actions or ineffective retries count as system errors even when deliberately chosen and individually completed normally, provided necessary work remains and there is no observed productive execution or justified dependency wait.
- Retain current/past-only reasoning, untrusted-data instruction boundaries, actual effects versus claimed completion, placeholder feedback handling, and missing/stale evidence as unknown. Do not infer a specific implementation or hardware failure from a symptom.
- Keep the existing JSON output and evidence rules, including explicitly retained decision/message event steps. Step units follow the provided observations and are not universally simulator frames.
- Bump the prompt version so previous prompt caches are not reused. Historical responses remain historical.

## Verification

Run existing LLM and detection contracts with fake transport. Reconstruct the prior real episode61 request locally to verify unchanged observational input and rejection of its previous-prompt cache with a zero-request budget. Compare the Python AST excluding the two prompt constants to prove there is no implementation change. A source review checks generality and the agreed waiting policy; these checks do not measure a live model's semantic accuracy. No new paid API requests or simulation runs are needed.
