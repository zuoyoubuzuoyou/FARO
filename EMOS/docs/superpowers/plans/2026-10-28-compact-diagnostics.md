# Compact Diagnostic Trajectories Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: use superpowers:subagent-driven-development or superpowers:executing-plans. Track work below.

**Goal:** Remove repeated static/state/envelope data without losing any trajectory evidence.

**Architecture:** Extend the existing V3 codec/store and keep public expanded reader interfaces. Add a fresh-directory compaction/verification command; use the completed4000-step episode as an offline acceptance case.

**Tech Stack:** Existing dependency-free Python storage, JSON/SHA256, unittest.

**Spec:** `docs/superpowers/specs/2026-10-28-compact-diagnostics-design.md`.

## Global Constraints

- No diagnostic-field deletion, subsampling, simulator/model calls, execution changes, historical-source overwrite or git commit/push.
- Preserve the existing dirty worktree and its runtime-observation implementation.
- Expanded events, states, step identity/order, timestamps, literal JSON and summary must be exact.
- Source input is read-only; output must be fresh; include payload/media bytes in measurements.
- Public readers remain `iter_trace_events`, `_Codec.decode`, `read_trace_summary`.

## Review Focus

- Context-dependent references escaping their step/episode or interpreting user JSON as control envelopes.
- Payload duplication/file-count or decoding-time regressions masked by a smaller steps file.
- Reused mutable values or stale diagnostics changing expanded data.
- Pending/final state and summaries referencing data unavailable after copying.
- Third-party/local report tools still inspecting raw event_type/agent_id/sequence before decoding.

### Task 1: Compact V3 storage

Own `habitat-mas/habitat_mas/utils/trajectory_storage.py`, new storage compaction tests, necessary existing storage-test assertions only.

- [x] Write and run RED tests for repeated camera/navigation state, metric reuse and exact event/summary expansion.
- [x] Implement selective content sharing and row-envelope reconstruction, preserving small fields and raw event_type/agent_id/sequence.
- [x] Cover literal/invalid/corrupt reference and pending semantics; run storage/recorder tests.

### Task 2: Safe offline repack and verification command

Own `scripts/compact_trajectory.py` and `test/test_compact_trajectory.py`; consume existing public storage APIs and the unchanged StepTraceStore constructor/append/commit/write_summary.

- [x] Write and run RED tests for new output, source immutability, expanded equality and rejection of same/existing/nested output.
- [x] Repack V3 row boundaries exactly, copy required metadata/media/config and payloads, and verify events and summary without model calls.
- [x] Report exact input/output file and directory bytes, file counts, preserved rows/events and hashes.

### Task 3: Integration and real-data acceptance

Root owns docs, plan ledger and analysis artifacts, performs independent review and source-preserving offline repack.

- [x] Compare every expanded real event, normalized detector frame and final summary for the 4000-step episode.
- [x] Measure total size, syntax and whitespace; run relevant trajectory/detection/recovery tests and broad discovery.
- [x] Address independent review, document new references and command, and report actual results.

## Execution ledger

- User explicitly approved the proposed lossless reduction; no additional approval round is needed.
- Work stays in the existing writable rsi checkout, preserving previous changes. Parallel tasks own disjoint files.
- Storage: 12 new contracts pass; final CLI: 14 tests pass; trajectory-pattern suite: 120 pass; detector suite: 124 pass. Broad discovery before the last two identity regressions: 479 entries, 449 passed, 30 module import errors due to missing pytest (28) and mock (2), no executed test failures. The final two regressions passed in the final CLI and trajectory suites. Syntax and git diff --check pass.
- Independent review found row-only evidence and extra identity evidence could be silently omitted by event replay. Added five RED/GREEN rejection regressions and exact typed JSON row-field comparisons. Reviewer independently confirmed safe refusal/cleanup and closed all blocking findings.
- Real completed source: 4000 rows, 40,398 row events plus the initial episode event. Every expanded row field and normalized frame matches exactly, final summary matches, public event streams match, and team_progress predictions match. All 271 original run files retain their initial SHA256 and inventory. No simulation or model request.
- steps.jsonl: 79,918,613 -> 56,222,845 bytes (29.65% reduction); episode including payloads/media: 82,551,138 -> 58,860,429 bytes (28.70% reduction), excluding the generated report. Payload file count: 247 -> 259, total episode files: 270 -> 282. Allocated file bytes: 83,431,424 -> 59,777,024.
- Verified compact copy: analysis_results/trajectory_compaction_20261028/run_20261028_020031/episodes/episode_0061. Evidence: analysis_results/trajectory_compaction_20261028/detector_equivalence.json and verification.json. The initial real conversion completed before the later CLI refusal guards were added; independent canonical JSON comparison of all 4000 expanded rows covers the actual accepted copy in full.
