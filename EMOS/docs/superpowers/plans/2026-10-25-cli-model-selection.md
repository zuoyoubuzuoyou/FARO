# CLI Model Selection Implementation Plan

**Historical plan:** Environment-variable compatibility below was superseded by the user's 2026-10-26 instruction to remove it. Current implementation follows `2026-10-26-remove-llm-model-environment.md` and passes the model through configuration directly.

**Goal/spec:** The user can switch the EMOS task model in the existing Hydra command; the unconfigured model defaults to `gpt-5.6-sol`.

**Architecture:** Register nullable `habitat_baselines.llm_model`. When specified, the direct entry point sets `EMOS_LLM_MODEL` for the execution scope, including spawned workers, and restores the prior environment afterward. Existing `OpenAIModel` selection remains explicit constructor argument > command-configured environment > existing environment > fallback. Empty command values fail before running a trainer. Omitting the command option preserves existing environment-variable behavior.

**Constraints:** Work in the current checkout and preserve previous trajectory changes. No API calls, new simulations, dependency installation or commits. Temperature and seed are outside this request. A provider's acceptance of a model name cannot be verified offline.

- [x] Model owner: update existing default tests, observe RED; change fallback and verify existing constructor/environment precedence.
- [x] Root: write failing tests that real Hydra config passes the CLI choice into actual request construction across discussion/planning/execution, preserves omission/environment precedence and restores the environment on errors; implement config and entry-point bridge.
- [x] Root: document CLI/default precedence, verify `--cfg job` and focused model/trajectory/config regressions; independent scoped review.

**Workspace ruling:** Continue in this feature checkout as previously authorized; no new worktree or approval handoff. Independent model/default and config/entry edits have disjoint ownership.

**Evidence:** Existing model-default tests first failed twice with `gpt-5.5`, then all six passed after changing the fallback. New CLI tests first failed five times because Hydra had no registered `llm_model`; all eleven CLI/model tests then passed. Final relevant regression command (model, CLI, trajectory, stability, config, context, recovery-source and evaluator suites) passed 92 tests. The full HSSD command including `habitat_baselines.llm_model=gpt-5.6-sol` passed actual `--cfg job` composition. Logs: `/tmp/emos-step-trajectory-20261025/cli-model-{red,green,regression}.log` and `cli-model-hssd.yaml`. Independent scoped review found no Important issues. `git diff --check` passed. No API calls or new simulations were started.
