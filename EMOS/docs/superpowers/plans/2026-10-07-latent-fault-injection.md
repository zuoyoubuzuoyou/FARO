# Habitat Latent Fault Injection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Implement AF-1 navigation no-op, PL-1 wrong subgoal, CF-1 message delay and PF-1 localization error in the actual Habitat runtime.

**Architecture:** Dependency-light specifications and window state drive four existing execution boundaries. Policy dispatch preserves live agents, message transport owns delayed delivery, and the task exposes an agent-specific estimated position consumed by oracle navigation and semantic descriptions. CLI and evaluator share one registry and preserve existing clean/dropout/hang behavior.

**Tech Stack:** Python 3.9, dataclasses, unittest, Hydra/OmegaConf, PyTorch, Habitat 0.3.1 and local ReplicaCAD assets.

**Spec:** `docs/superpowers/specs/2026-10-07-latent-fault-injection-design.md`

## Global Constraints

- Four faults only; automatic latent-fault detection/recovery is outside this injection task.
- Agents remain online; real object transforms and PDDL success stay authoritative.
- Zero-based simulator steps; finite windows are half-open, -1 lasts to episode end.
- Only one enabled fault per run; all fault state resets even for repeated episode IDs.
- Preserve existing CLI, prefix replay, JSONL and text traces; retain staged user documents.
- Work in the existing `other-fault` checkout as explicitly requested; do not create an alternate checkout or commit staged user files.
- User instruction “按设计继续实现” authorizes implementation. Execute inline and continuously; no further task-authorization check-in.

## Review Focus

- Replay prefix must not arm delay or modify navigation caches before its branch boundary.
- Delay to a later-dispatched agent must not deliver earlier than the same delay to an earlier agent.
- Policy `actions` may encode a high-level skill; only `take_actions` is a navigation command.
- Localization must invalidate cached targets on entry and expiry and remain agent-specific.
- Trigger without a nonzero command/message/actual navigation consumption must be reported as ineffective.

## Task 1: Fault specification, state and CLI

**Files:** Create `habitat-mas/habitat_mas/fault_injection/latent.py` and `test/test_latent_faults.py`; modify `fault_injection/launcher.py`, `fault_injection/cli.py`, `habitat-baselines/habitat_baselines/config/default_structured_configs.py`.

**Interfaces:** Four spec classes expose `from_config`, `validate`, `episode_matches`, `target_simulator_step`, `simulator_step_schedule_map`. `FaultWindow.advance(step)` emits entry/expiry; registry maps four names to config paths. CLI writes exact resolved fault parameters to plan.json.

- [x] Write tests for window edges/reset, invalid entities/offset/durations, four aliases and dry-run parameters; verify RED.
- [x] Implement specs/window plus Hydra subtrees and applicable CLI options; preserve existing fault_overrides callers.
- [x] Run `test_latent_faults` and launcher/clean regressions; expect all PASS.

## Task 2: Delayed message delivery

**Files:** Modify `agents/message_bus.py`; create `test/test_message_delay.py`.

**Interfaces:** `AgentMessageBus.begin_step(step, delay=None)` delivers due messages and configures the current window; `drain_delivery_events()` returns evidence; reset clears pending messages and IDs.

- [x] Verify messages sent at step 5 with delay 3 are invisible at steps 5–7 and delivered before policies at step 8, preserving send order; watch RED.
- [x] Implement pending queues, per-message IDs and send/delivery/drop events; preserve immediate send behavior when disabled.
- [x] Test expiry with pending delivery, target dropout, reset and preserved sender/content; expect PASS.

## Task 3: Live action and planning injection

**Files:** Create `fault_injection/latent_runtime.py`, `test/test_latent_runtime.py`; modify `agents/crab_agent.py`, `rl/multi_agent/multi_llm_policy.py`.

**Interfaces:** `LatentFaultRuntime` owns per-episode lifecycle/events and modifies individual policy outputs. `CrabAgent.replace_subtask(subtask)` updates system and planning inputs while retaining history. Runtime action masking uses agent-local action-space ranges, preserving stop and high-level history.

- [x] Test real tensor no-op masking with preserved policy/other-agent state and exact entity-token subtask replacement; watch RED.
- [x] Implement lifecycle hooks after episode reset, before individual planning and after policy.act; initialize with corrupt subtask at step zero, replan/interruption mid-episode.
- [x] Connect message begin-step before either policy; attach generic events to policy_info, preserving replay prefix and decision counters.
- [x] Run runtime, message, existing hang policy and target-validation tests; expect PASS.

## Task 4: Simulator localization estimate

**Files:** Modify `habitat-lab/habitat/core/environments.py`, `tasks/rearrange/rearrange_task.py`, `actions/oracle_nav_action.py`, `actions/habitat_mas_actions.py`, `habitat-mas/habitat_mas/tasks/habitat_mas_sensors.py`, `scene_graph/utils.py`; create `test/test_localization_fault.py`.

**Interfaces:** Task setter accepts agent/entity/offset/fault identity and step; estimated navigation lookup logs real and estimated positions; per-agent text context regenerates height/distance from the same estimate. Changes clear only affected oracle target caches.

- [x] Test navigation lookup offset, cache entry/expiry, untouched ground truth and peer estimates, semantic description consistency; watch RED.
- [x] Implement task/VectorEnv interface and position consumption in `_get_target_for_idx`; collect simulator evidence in dynamic snapshots.
- [x] Ensure task reset clears estimates and evidence, even after auto-reset; run focused tests and config/import checks, expect PASS.

## Task 5: Evaluator, traces and summaries

**Files:** Modify `rl/multi_agent/habitat_mas_evaluator.py`, `fault_injection/trajectory.py`; create `test/test_latent_evaluation.py`.

**Interfaces:** Evaluator selects any one registered spec, creates latent runtime, synchronizes per-step simulator estimate, records generic events and run summaries. Summary distinguishes trigger/application and reason for no application; JSONL/text include original and effective actions and message timing.

- [x] Test effective versus triggered-only counts, generic trace events and four-fault replay acceptance; watch RED.
- [x] Wire runtime and simulator evidence, finish pending messages per episode and run; preserve legacy recorder/event behavior.
- [x] Run all 159 baseline cases plus new cases and four real Hydra compositions; expect PASS.

## Task 6: Native Habitat verification and user documentation

**Files:** Create `test/native_latent_fault_smoke.py`, `docs/FAULT_INJECTION_ZH.md`; update plan checkboxes and specification status.

**Interfaces:** Fixed model supplies reproducible navigation, replanning and real send_request without API keys. Native runs emit inspectable JSONL with proof assertions for all four faults and both navigation agents.

- [x] Run paired native no-op/localization cases for agent_0 and agent_1, plus planning and message cases; require actual effective events, boundary behavior and unchanged real objects/goals.
- [x] Exercise one real clean-prefix replay per new fault; verify branch boundaries and missing-effect diagnostics.
- [x] Document commands, parameters, output evidence and native results; perform full-branch review and address Important/Critical findings with RED→GREEN tests.
- [x] Run final regression/compile/diff checks and audit every spec acceptance requirement before claiming completion.

## Execution ledger

Baseline: 159 existing relevant unittest cases passed before implementation. Native GPU visibility confirmed with elevated read-only nvidia-smi; original sandbox CUDA remains unavailable.

Ruling: Use the current `other-fault` branch and existing installed Habitat environment — the user expressly requested implementation in the current EMOS workspace; an additional worktree or dependency installation would introduce unnecessary state.

Implementation: all four specs, shared windows, CLI parameters, message queues, live policy action/planning hooks, per-agent localization and evaluator traces/summaries are wired. Original 159-case baseline plus new coverage: 185 cases PASS in the final regression run. Four real Hydra compositions passed.

Native failures diagnosed: actual policy uses a flattened action space while its high-level policy retains named channels; EMOS robots override target lookup in OracleNavDiffBaseAction. Both received reproducing tests and runtime fixes.

Review: PF expiry now publishes restored XYZ to later model decisions; abort finalization retains buffered events, pending-message outcomes and summaries; AF coordinate counting follows the real look-at activation condition; localization boundaries also clear the completed-navigation latch. Each reproduced failure before the corresponding fix.

Final verification: 185 unittest cases PASS, compileall PASS, git diff --check PASS, four Hydra compositions PASS, 14 native cases PASS. Evidence: fault_output/latent_native_validation_20261007_final/verification.json. Native checks also verify replacement target identity, recipient model visibility, biased/restored model XYZ, unaffected peer state and true object positions, actual repeated-episode reset, and exception/pending-message evidence. CF replay correctly reports ineffective because its live suffix sends no message; the separate CF native case is effective. Independent follow-up review reports no remaining Important/Critical defects. Workspace and staged user documents retained without commits.
