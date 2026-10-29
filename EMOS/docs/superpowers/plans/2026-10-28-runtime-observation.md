# Runtime Observation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans. Steps use checkbox syntax for tracking.

**Goal:** Record the current Perception task conditions and execution evidence and make both detectors consume them.

**Architecture:** Add read-only worker diagnostics to the existing trajectory snapshot; instrument navigation at its original calculation sites; extend recorder outcome/request correlation and the existing normalized detector schema. Preserve historical inputs and EMOS execution behavior.

**Tech Stack:** Existing Python Habitat/MAS code, standard library and existing NumPy, unittest, V3 storage.

**Spec:** `docs/superpowers/specs/2026-10-28-runtime-observation-design.md`

## Global Constraints

- No extra simulation steps, sensor-suite calls, skill termination calls or PDDL cache mutation for logging.
- Preserve EMOS's tool Success, model inputs, control vectors, rewards and task success semantics.
- Current values are bool/null with source, env_step and timing; never substitute latched stages or missing values with false.
- No source history or generated old trajectories are rewritten; preserve the existing scene_graph/utils.py change and its test.
- Work in the current writable checkout on branch rsi; no commit, push or external model requests are required.

## Review Focus

- Native root/reference offsets and strict versus inclusive thresholds.
- Observation missingness, stale navigation and episode resets.
- Observer exceptions must not affect original results or mask execution errors.
- Action end can precede returned env.step; correlation must not attach a later action's state.
- Reader and LLM filtering must retain current evidence without exposing future outcomes, annotation or injected faults.

## Task 1: Worker native current diagnostics

Files: `habitat-lab/habitat/core/environments.py`, new `habitat-lab/habitat/core/trajectory_diagnostics.py`, `test/test_trajectory_state.py`, new collector tests.

Produces the shared `current_goal`, per-agent `goal_coverage` and `detection` state defined in the spec; consumes returned observations and the navigation producer's step-stamped `_navigation_diagnostics`.

- [x] Write failing snapshot/collector tests for native frame, target mapping, current detections and missing data.
- [x] Run the tests and verify the missing diagnostics fail.
- [x] Implement detached observational collection and get_info integration without losing the terminal state before vector reset.
- [x] Verify tests, no extra sensor/cache/control calls, JSON serializability and legacy state capture.

## Task 2: Navigation observation producers

Files: `habitat-lab/habitat/tasks/rearrange/actions/oracle_nav_action.py`, `habitat-lab/habitat/tasks/rearrange/actions/habitat_mas_actions.py`, new navigation tests.

Produces `_navigation_diagnostics` with standard detector names, `env_step=task._cur_episode_step+1`, and timing `pre_control` on existing control branches. Collector reads these; no collector files are edited by this task.

- [x] Write failing tests for active, arrival, invalid/inactive and path-failure diagnostics and preserved commands.
- [x] Implement diagnostics from already computed state for generic, coordinate and MAS differential-base navigation.
- [x] Verify navigation results and existing state transitions remain unchanged.

## Task 3: Detector observation path

Files: `habitat-mas/habitat_mas/detection/readers.py`, `heuristics.py`, `llm.py`, detection tests.

Consumes the Task 1 state contract and recorder events from Task 4. Produces normalized current metrics and native-first goal assessment; legacy frames without new fields continue to use their former approximation.

- [x] Write failing tests for native false/true/null precedence, detection loss, fresh timing, LLM payload and end-to-end alarms.
- [x] Preserve new state, identifiers and structured action-outcome/feedback events through readers and LLM input sanitization.
- [x] Teach heuristic assessment to prefer current native predicates and distances.
- [x] Verify all detection tests, causality and forbidden-field filtering.

## Task 4: Outcome/feedback correlation and integration

Files: `habitat-mas/habitat_mas/utils/trajectory.py`, recorder and integration tests, `docs/step_trajectories.md`, `docs/system_error_detection.md`.

Produces observational `action_outcome` events with real action ID, controller result, signal step and observation step. `execution_feedback` links each next actual LLM request to the prior outcome and placeholder IDs; it never edits the request.

- [x] Write failing tests for deferred termination, superseded action, terminal/no-step endings and next-request correlation.
- [x] Implement outcome emission using the action's last actual returned observation and original controller status.
- [x] Add capture-to-storage-to-detector integration evidence and explain recording-only versus feedback repair.
- [x] Run relevant regression suites, independent code review, fix verified findings and report limits.

## Execution ledger

- Baseline: 106 detection tests pass. Existing unrelated scene-description edits are preserved.
- Ruling: user authorization to add the discussed items permits implementation without another approval round; the written spec records the agreed conversation.
- Ruling: use disjoint file ownership for parallel producer/collector/detector work; the root owns recorder integration and documentation.

- Task 1 complete: native root-distance/current detection capture, unknown reasons, schema compatibility and source metadata verified.
- Task 2 complete: navigation branches instrumented using existing calculations; original controls, returns and errors covered by regression tests.
- Task 3 complete: native evidence and outcome/request events reach heuristic and LLM; stale/future values and forbidden labels are filtered.
- Task 4 complete: action outcome and actual model-feedback correlation, source hashes, documentation and capture-to-storage-to-detector test added.
- Review fixes: same-frame terminal outcomes now precede the step event while action_end keeps its original ordering; stale/future dynamic diagnostics are cleared; native false/null is not overwritten by aggregate success. Fresh independent re-review approved, no blocking findings remain.
- Final verification: 455 unittest entries, 425 passed, zero test-case failures; 30 unrelated generic Habitat test modules could not import because pytest or mock is absent. Detection-only suite: 124 tests passed. All changed production sources parse and git diff --check passes.
- Evidence: analysis_results/detection/runtime_observation_20261028/{verification.json,example_diagnostics.json,test_results.log,code_review.md}. The sample is synthetic; no real simulator or model request was run, and no historical trajectory was altered.
