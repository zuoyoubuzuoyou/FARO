# Combined Source Repair Implementation Plan

> **For agentic workers:** Use test-first implementation with disjoint ownership; root owns integration and verification.

**Goal:** Detect and propose a constrained original-source repair in one model response, then validate/install/resume through the existing source runtime.

**Architecture:** A sampled-frame hook selects combined source requests; existing standalone/extension routes remain. A restricted source-validation profile applies at prepare and install. Shared monitoring instructions keep classification consistent.

**Tech Stack:** Existing Python recovery/controllers, AST source transactions, unittest and fake transports.

**Spec:** `docs/superpowers/specs/2026-10-28-combined-source-repair-design.md`.

## Constraints

User selected automatic application after validation. Preserve existing unrelated changes, credentials, historical data, task semantics and physics. Do not make live model or Habitat calls, reset the episode, change evaluation predicates, or commit code.

## Tasks

- [x] Extract shared generic monitoring policy without changing standalone binary requests.
- [x] Implement and test combined source request/output and sampled controller integration; reuse source application and exception resume.
- [x] Implement restricted automatic source checks and adversarial preparation/install tests.
- [x] Integrate interfaces, document exact behavior/limits/commands, and independently review the resulting changes.
- [x] Run focused and broad regressions; record source restoration, no extra requests/steps, and remaining validation limits.

## Execution results

- The user selected automatic application after validation. Default source+LLM now uses one `detect_and_repair` call; standalone boolean detection and explicit legacy routes remain compatible. Shared monitoring extraction leaves the standalone system prompt byte-identical.
- New tests: 21 combined, 12 restricted guards and 1 runtime host-pin test, all passing. Full discovery: 512 entries, 482 passing executed cases and no executed case failures; 30 pre-existing import errors (pytest28/mock2). Source integration and worker/rollback fixtures restore original files.
- Real model/policy parsing integration uses one fake repair response to install an original parser correction and Resume into the same live objects, retains history/model identity and actual attempt usage, retries before one physical-step boundary, then restores source bytes. No live API request or Habitat episode was run.
- Independent security review found receiver-prefix truncation and parser answer-fabrication bypasses. Regression tests were observed failing, then passed after complete-root capability checks, parser provenance, unchanged logging/control branches and installation baseline checks. Root replayed eight actual adversarial mutations without executing them; all rejected.
- Automatic scope is deliberately restricted to declared parent Resume and model-parser methods. Simulator/task/physics/observation/evaluation/worker source remains readable but not automatically editable. Rejected/unrepairable proposals do not widen permissions. The profile is not a general Python sandbox or proof of semantic correctness/resource bounds.
- Explicit developer-injected generate-only legacy objects retain their old transaction behavior for backwards compatibility; the configured online generator cannot choose that route. This preserves coordinated-worker rollback coverage without relaxing the actual automatic model path.
- Independent combined-flow review found no remaining blocker. The real Perception Hydra job configuration parses with recovery enabled/source/llm. `git diff --check` passes. Evidence is under `analysis_results/recovery/combined_source_20261028/`.
