# Episode Stability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Execute the approved Manipulation 10-task, three-round stability pilot and report native success and outcome changes without hiding incomplete trials.

**Architecture:** A standard-library runner creates an immutable cohort and launches one fresh Habitat worker per task/round. The worker temporarily wraps the existing model request method to supply temperature/seed, without editing production sources. A separate standard-library report module validates and summarizes the fixed manifest.

**Tech Stack:** Python 3.9+, unittest, existing Habitat/OpenAI environment, GPU 1.

**Spec:** `docs/superpowers/specs/2026-10-25-episode-stability-design.md` (approved in chat 2026-10-25).

## Global Constraints

- Ten Manipulation episodes sampled from all 166 with sample seed 100; three identical rounds.
- Environment/simulator/model seed 100; model gpt-6-sol; temperature 0; recovery disabled; max episode steps 4000; wall timeout 1800 seconds; GPU 1.
- Preserve current dirty workspace and existing outputs. Only add experimental scripts, tests and documentation; do not edit production sources.
- Missing records and API/execution/process errors remain distinct from completed native S/F; halt further paid runs on errors.
- No cached model answers and no selective retry or outcome-based task selection.
- Work in the approved workspace on branch rsi: a clean worktree would omit the modified baseline under study. No commits or changes to existing user files are needed.

## Review Focus

- A service error after some simulator steps must not be promoted to native F.
- Missing state/request/terminal evidence must invalidate a trial even when the process exits 0.
- Duplicate or foreign task/round identities must fail aggregation rather than inflate coverage.
- Initial-state comparisons must exclude run identifiers/timestamps and describe recorded state only.
- A changed source/input hash or existing output directory must stop execution rather than mix experiments or overwrite evidence.

## Shared Manifest Interface

`manifest` is a dict with `settings`, `cohort`, `trials`, `source_hashes`, `input_hashes`, and `status`. Settings include `repetitions=3` and the explicit model/seed values above. Cohort records have `task`, `scene_id`, `episode_id` (string), and `dataset_file`. Trial records have the same identity plus `round` (1-based), `trial_id`, `trial_dir`, `status`, and `task_success` (bool or null). Initially status is `pending`; only `completed` with a real bool is a valid S/F result. Other statuses include `running`, `api_error`, `execution_error`, `process_error`, `timeout`, `recording_error`, `configuration_changed`.

### Task 1: Report fixed-cohort results

**Files:** Create `scripts/episode_stability_report.py`; test `test/test_episode_stability_report.py`.

**Interfaces:** `summarize(manifest: dict) -> dict`; `write_reports(manifest: dict, output: pathlib.Path) -> dict`. The latter writes `summary.json`, `rounds.csv`, `task_outcomes.csv` (UTF-8 BOM CSV) and returns the summary. No Habitat imports or network calls.

- [x] Write and run failing tests for literal S/S/S, F/F/F, S/F/S outcomes, pairwise directional flips, incomplete coverage, error-vs-F distinction, duplicate and foreign identities.
- [x] Implement validation and metrics. Report per-round planned/completed/success/failure/other status counts and completion coverage. Formal round success_rate is null if that round lacks full coverage; success_rate_range_pp is null unless all rounds are complete.
- [x] Compute pairwise 1–2, 1–3, 2–3 valid intersections, flip counts/rates and S→F/F→S with that same denominator; no valid pairs means null rates. Compute any-change fraction on the three-round complete intersection.
- [x] Include task outcome matrix and whether recorded initial-state hashes agree wherever supplied. Do not infer formal statistical stability from 10 episodes.
- [x] Run `PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s test -p 'test_episode_stability_report.py' -v`; expected all pass.

### Task 2: Runner, worker and evidence validation

**Files:** Create `scripts/run_episode_stability.py`; test `test/test_episode_stability_runner.py`.

**Interfaces:** Consumes Task 1 functions and shared manifest. Expose `select_episodes(episodes, count=10, sample_seed=100)`, `install_decoding_controls(model_class, settings)`, `inspect_trial(trial, settings, returncode, timed_out=False)`, and CLI `--prepare-only`, `--summarize-only`, `--output`, internal `--worker-spec`.

- [x] Write and run failing tests proving outcome-independent sampling, request parameter enforcement/conflict rejection, missing required evidence rejection, and native F/API error separation.
- [x] Implement cohort export and write all 30 planned trials before execution. Validate scene and robot data. Refuse pre-existing experiment output; summarize-only may read it. Save relevant source/config hashes including both new scripts and source dataset/robot hashes.
- [x] For each trial launch an independent worker process group; retain stdout/stderr log; enforce wall deadline and kill only that trial's process group on timeout. Write worker spec and resolved command metadata without secrets. On error stop subsequent trials while retaining pending entries.
- [x] Worker sets visible GPU and model before imports, loads existing dotenv/dependencies, wraps only temperature/seed, calls the original Habitat module with the approved overrides. No model parameter overrides outside this worker.
- [x] Inspect the original summary/events for exact identity, initial state, complete native metric, all request parameter records, and response metadata. Classify API and execution errors separately; capture tokens/elapsed/steps/paths. Check source/input hashes before and after each trial; stop on changes.
- [x] Run targeted tests plus existing model/trajectory integration regression tests; expected all pass. Perform prepare-only and config-only checks before real API-backed simulation.

### Task 3: Review and execute the pilot

**Files:** Experiment artifacts under a fresh `analysis_results/stability_manipulation10_20261025_*` directory; update this plan and spec status.

- [ ] Fresh independent review of both scripts/tests for the global constraints and Review Focus; fix important findings with reproducing tests.
- [ ] Launch the 30-trial run outside the sandbox on GPU 1, with authorization already given in chat. Monitor progress and report meaningful updates at most 60 seconds apart.
- [ ] If runtime infrastructure or required recording fails, stop as specified and preserve explicit partial metrics and failure evidence; never claim the pilot completed.
- [ ] Verify final manifest/summary counts and output files; report success rates, outcome flips, coverage and sample-size limits, or the precise blocker if incomplete.

## Execution Ledger

- Planning: design approved by user; implementation and pilot run authorized. No additional approval handoff is needed.
- Ruling: add-only experimental files in current dirty checkout rather than a clean worktree, because the approved baseline is this workspace's code snapshot. Existing source edits remain untouched.
- Ruling: parallelize the independent reporting module; root implements the runner against the shared interface. A fresh whole-change review follows integration.

- Implementation: reporting module and runner implemented; 19 targeted tests passed. Independent review found five issues; reproducing tests observed failures before fixes, then all passed.
- Regression: six relevant existing suites passed; broad discovery had 30 import errors due to missing pytest, preserved in experiment logs.
- Input monitoring: SHA256 of 21,106 configuration/input files; size/mtime inventory for 77,632 asset files. Binary mesh contents are not hashed.
- Ruling: isolate legacy relative log outputs with per-trial workspace directories and read-only-by-convention asset symlinks; no production source edit.
- Final: minor (deferred): absent provider token usage is summed as zero by the initial reporting inspector; raw response usage remains retained. Actual usage availability must be checked when reporting cost.

- Task 1: complete; Task 2: complete. Task 3: executing. Experiment: `analysis_results/stability_manipulation10_20261025_run1`. First two completed trials are native F (64, 4000 steps) and native S (78, 339 steps); no outcome-flip result yet.

- Final scope update: user requested one episode repeated three times. Original background batch was already completed before this message. Revalidated episode 78 all three rounds with the real trace inspector and an independent reviewer; extracted single-episode manifest, result, JSON/CSV summaries without new API/simulation. All three native S, 339 steps, identical recorded transitions; no result flips. Provider prompt caching exists and LLM dialogue lengths differ, so no blanket model determinism claim.
