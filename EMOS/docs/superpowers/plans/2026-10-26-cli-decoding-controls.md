# CLI Decoding Controls

**Requirement:** Extend the existing direct HSSD command with task-LLM `temperature=0.0` and API `seed=100`, consistently across discussion, planning and execution, without environment-variable selection.

**Design:** Add nullable `habitat_baselines.llm_temperature` and `habitat_baselines.llm_seed`. Pass them explicitly through MultiLLMPolicy, group_discussion and CrabAgent.init_agent into appended OpenAIModel constructor arguments. Merge non-None constructor settings into request kwargs before tracing and transport, retaining explicit per-call overrides. Unspecified controls remain omitted from requests. Include controls in replay configuration checks; legacy missing optional controls are compatible with current None only. The stability worker supplies the same Hydra options from its existing settings while retaining its audit guard.

**Constraints:** Preserve previous uncommitted work. No real API calls, simulations, installs, commits or unrelated parameter changes. Existing constructor positional arguments stay compatible. Provider-side deterministic execution is outside offline validation.

- [x] Model owner: RED missing constructor controls, then implement and verify four request/trace/legacy-instance tests.
- [x] Root: reproduce missing CLI/stability overrides, wire config to all model constructors and record configuration for replay.
- [x] Root: run relevant regression and actual Hydra composition, update command documentation and obtain scoped independent review.

**Workspace ruling:** Continue in the existing checkout with disjoint implementation owners as previously authorized.

**Verification:** CLI/worker RED had two missing-config errors and one missing-override failure. Replay RED had a missing-config-field error and failed to reject changed legacy seed. Model owner reproduced absent constructor arguments and an older live instance lacking new attributes before adding `getattr(..., None)`. Final relevant regression passed 123 tests in 6.791s (`/tmp/emos-step-trajectory-20261025/cli-decoding-regression.log`). The actual full HSSD command passed `--cfg job`, resolving `llm_temperature: 0.0` and `llm_seed: 100` (`cli-decoding-hssd.yaml`). Seven recorded requests in the CLI integration cover leader discussion, robot reflection, planning and execution with explicit request sources. Independent scoped review passed with no Important findings. `git diff --check` passed. No actual API calls or simulations were started.
