# Environment Step Trajectory Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans. Steps use checkbox (`- [x]`) syntax.

**Goal:** Record one completed environment transition per `steps.jsonl` line while retaining ordered model/action evidence and reconstructable replay memory.

**Architecture:** A dependency-free storage/reader module groups the recorder's existing expanded events. The recorder keeps its live callbacks and ordering; consumers read a common expanded-event interface for V1/V2/V3. Large repeated payloads use checked references; pending evidence persists without inventing completed steps.

**Tech Stack:** Python 3.9+, standard library, unittest, existing Habitat/PyTorch runtime.

**Spec:** `docs/superpowers/specs/2026-10-25-step-trajectory-design.md` (implementation explicitly requested 2026-10-25).

## Global Constraints

- One returned environment step per physical JSONL line, numbered 1..N; initial state is separate.
- Keep original event sequence, actual native success, pre-autoreset state, missing/null/false distinctions and failed attempts.
- Preserve full actual requests/responses and decision-boundary grouped model/communication memory; deduplicate, never replace history with a hash without its content.
- No paid API calls, new simulations, dependency installation, historical-output mutation, automatic commits or pushes.
- Work in the current `rsi` workspace. Only pre-existing new file is the approved design; preserve other user edits.
- Plan-owned scratch/ledger: `/tmp/emos-step-trajectory-20261025/`. Independent implementation owners edit disjoint files; root handles integration.

## Review Focus

- Errors before the first env.step and process interruption retain evidence, but create zero completed steps.
- Commit-before-pending-cleanup crashes cannot duplicate replay events; partial final lines must not fabricate transitions.
- Hash/path/reference tampering and mixed episode identities raise explicit recording errors.
- Timeout/handoff and model-suggestion versus entered-skill distinctions survive expansion in original order.
- `state.task_progress` and merged metrics are not guaranteed equal; keep their separate meanings and original success.

## Shared Interfaces

New module: `habitat-mas/habitat_mas/utils/trajectory_storage.py` (stdlib only).

- `StepTraceStore(directory, identity)`: episode directory already has `episode.json`; create it from identity if absent for standalone conversion.
- `append(event: dict) -> None`: persist an expanded event; initial event is stored with episode metadata; incomplete records go to atomic pending storage. Validate identity.
- `commit_step() -> None`: persist exactly one pending `event_type=step` and associated ordered events, then clear committed pending evidence. No fake step when none returned.
- `write_summary(summary: dict) -> None`: keep small summary readable; reference repeated terminal state/large data without losing expansion.
- `close() -> None`: close resources, preserve uncommitted evidence; safe/idempotent.
- `iter_trace_events(path, *, include_pending=True)`: stream legacy V1/V2 JSONL as-is; expand V3 `steps.jsonl` plus initial/pending evidence, validating order, identities and references.
- `read_trace_summary(path) -> dict`: expanded summary for V2/V3 consumers.

Owner may settle private codec details, but public step rows must expose `schema_version=3`, `step`, `action`, `reward`, `done`, `metrics`, `state`, ordered `events` and replay context when present. Ref control must not collide with literal user/model JSON. Content must be redacted before persistent hashing/storage.

### Task 1: Step storage and compatibility reader

**Owner/files:** Agent storage owns new `trajectory_storage.py` and `test/test_step_trajectory_storage.py`.

- [x] Write failing tests using literal event fixtures: one line per returned step; exact expansion/order; startup/error-only/pending; pending cleanup crash duplication; identity mismatch, damaged refs, missing files and partial final line; literal reference-shaped data survives.
- [x] Observe failures, implement the shared interfaces, run tests until green. Keep JSONL prefix streaming; avoid loading entire episodes unnecessarily.
- [x] Deduplicate repeated step state/commands/metrics, request messages, response contents and history messages with references. Preserve differing values.
- [x] Report red/green commands and private schema decisions to `/tmp/emos-step-trajectory-20261025/task-1-report.md`.

### Task 2: Recorder integration

**Owner/files:** Agent recorder owns `habitat-mas/habitat_mas/utils/trajectory.py` and new `test/test_step_trajectory_recorder.py`.

- [x] Write failing integration tests for two returned steps, initial/terminal metadata, callbacks, exception-before-step, exception-after-step, timeout/handoff event ordering, repeated attempts and summary references.
- [x] Switch physical output to schema 3 `steps.jsonl`, using Task 1 store. Keep expanded hooks/callback payloads synchronous and detached.
- [x] Commit a nonterminal step after its action_execution/deferred action_end events; a terminal step after episode_end. Preserve pending errors on close.
- [x] Stop duplicate `run.json`; update file indexes to steps. Do not edit consumer code/tests owned by root.
- [x] Report tests and integration assumptions to `/tmp/emos-step-trajectory-20261025/task-2-report.md`.

### Task 3: Replay memory at real decision boundaries

**Owner/files:** Agent memory owns `habitat-mas/habitat_mas/agents/crab_agent.py` and new `test/test_step_replay_context.py`. Coordinate helper module only if needed; do not modify trajectory.py.

- [x] Write failing tests proving export/restore preserves grouped history, final placeholder messages, system/task/planner/start state, pending recovery instructions, ordered global message_pipe; snapshot detached and no model/network calls.
- [x] Export context after a real decision through `_record_action_output`, guarded by active trace action identity so disabled recording has no cost/side effects. Attach context to `agent_action` event for storage grouping.
- [x] Retain model/window/tool configuration evidence; restoration on existing objects must validate incompatible settings and preserve live clients, skills and simulator objects.
- [x] Test forced recovery and send_request-to-wait paths; report red/green to `/tmp/emos-step-trajectory-20261025/task-3-report.md`.

### Task 4: Consumer integration, documentation and offline verification

**Owner/files:** Root owns `detection/readers.py`, `scripts/run_episode_stability.py`, `scripts/trajectory_smoke.py`, existing trace-related test reader helpers, new consumer integration tests, docs.

- [x] Write failing tests that V3 normalizes into the same detector frames, stability inspection reads V3 and identifies incomplete/API failures, and older V1/V2 remain accepted.
- [x] Use unified reader for disk consumers; use expanded summary access; retain old behavior and phase/native metric checks. New file selection follows summary indexes with legacy fallback.
- [x] Update existing tests to read expanded events via the common reader rather than assuming physical event rows. Preserve all existing behavioral assertions.
- [x] Document actual output files and commands, plus current replay-execution limitation.
- [x] Convert real historical episode 78 into a fresh ignored output directory; verify 339 step lines, complete expanded ordered events and normalized frames; measure total size including sidecars.
- [x] Run all relevant trajectory/detection/recovery/stability suites; no real API/simulator launches. Fresh code review before completion, address correctness findings with reproducing tests.

## Execution ledger

- Design accepted by user's explicit implementation request. No further approval handoff is required.
- Ruling: implement in current feature checkout; parent Git metadata and local assets span outside an isolated child checkout, and user expects this branch's behavior updated. No unrelated edits/commits.
- Ruling: parallel disjoint implementation tasks under the higher-priority collaboration instruction; shared public storage interfaces above bind all tasks.
- Tasks 1–4: complete; see verification evidence below.


## Completion evidence (2026-10-25)

- 350 relevant tests passed with Habitat Python, using `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:test:habitat-lab:habitat-baselines:habitat-mas:/tmp/emos_detection_deps /home/dyc/.conda/envs/habitat/bin/python /tmp/emos-step-trajectory-20261025/run_integration.py`. Log: `/tmp/emos-step-trajectory-20261025/integration-final.log`. Source-patching integration tests restored production source byte-for-byte.
- Two independent reviews passed after reproducing and fixing callback isolation, invalid pending packets, malformed input classification, orphan payload growth and replay restore validation. Reports and RED/GREEN evidence are in the plan scratch directory.
- Historical episode 78: 339 physical rows, 3556 identical expanded events, 339 identical detector frames, matching expanded summary after file-index changes. Including config and every sidecar/media file, 6,734,411 -> 5,408,725 bytes (-19.69%). Accepted offline artifact: `analysis_results/step_format_validation_ep78_20261025_compact/`; original files untouched.
- Public usage and current replay-execution limitation documented in `docs/step_trajectories.md`. Static event identity remains explicit for first-release validation; spec updated accordingly.
- No new simulation/API requests, dependency installation, commits or pushes.
