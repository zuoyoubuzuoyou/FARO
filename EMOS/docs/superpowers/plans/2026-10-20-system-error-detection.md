# System-error detection implementation plan

> For agentic workers: implement independent components against the shared JSON interfaces, test each component, then perform an independent integration review.

**Goal:** causal heuristic and LLM detection over all390 historical episodes, with independent runtime-anomaly manual labels and episode-level precision/recall. User explicitly chose to annotate runtime anomalies separately before formal scoring; final failure-attribution classes must not be reused as labels.

**Spec:** [design](../specs/2026-10-20-system-error-detection-design.md).

**Architecture:** dependency-light normalized frame stream, pluggable stateful monitors, a resumable replay runner and independent evaluation. Existing experiments remain unchanged. OpenAI-compatible inference uses a separate detector client and the configured credentials; tests use fake transports only for API boundaries.

## Tasks

- [x] 1. Add schema1/schema2 streaming readers with whitelisted causal features; tests for initial context, step units, changing actions, missing fields and oracle/future leakage.
- [x] 2. Add repeat-action baseline and team-progress rules; tests for normal waiting/navigation, fulfilled vs unresolved roles, sustained stalls, missing diagnostics and false completion.
- [x] 3. Add structured LLM windows, strict JSON validation, append-only audit/cache, error and usage recording; tests first for cadence, causal requests, invalid output, resume and distinct detector contexts.
- [x] 4. Add reference loaders and episode scoring for runtime vs attribution labels, unknowns, success/control and failed-only scopes; tests for TP/FP/FN/TN, duplicate alarms, absent predictions and zero denominators.
- [x] 5. Add replay CLI/configs/CSV reports, read-only source protection and examples with exact time units. Run all related and prior recorder tests.
- [x] 6. Run causal replay on the four complete batches; verify bounded real LLM calls then full resumable inference, retain raw responses and model metadata, and repair incomplete episode predictions.
- [x] 7. Independently review scope, leakage, metrics denominators, label provenance and actual model-call coverage. Report measured predictions separately from unavailable formal runtime GT.
- [ ] 8. Obtain independent human runtime labels, compute formal P/R, and inspect false positives/negatives without post-hoc threshold tuning.

## Review focus

1. Current pose is not necessarily the exact predicate reference transform: keep geometric coverage marked approximate.
2. Latched stage/original error_detection metrics must never substitute for current progress or independent labels.
3. Failed-only metrics and all-episode metrics must retain explicit distinct denominators.
4. Partial API failures must not become detector negatives; unknown labels must not become model errors.
5. Cache reuse must require identical prompt/model/config/window and preserve complete request provenance.

## Verified execution status

- 2026-10-20: 175 related tests passed, zero failures/errors/skips (93 detection and82 prior recorder tests). Independent code review accepted the fixes for missing-observation handling, slow motion, schema2 feedback ordering, provider cache isolation and evidence validation.
- All390 episodes and695862 raw frames replayed offline. Both heuristic methods completed;1170 aggregate rows cover allthree methods. Source SHA256 matched all390 original splits (3,634,603,826 bytes).
- Outputs: `analysis_results/detection/emos_runtime_20261020/`. All390 independent runtime labels remain blank as requested. Formal precision/recall are unavailable, not zero or perfect.
- The initial offline run made no LLM API calls. The user subsequently authorized `gpt-6-sol` and refreshed invalid credentials. All four interface smoke windows returned valid responses. The full390 replay and two repair passes have finished under `analysis_results/detection/emos_runtime_gpt6sol_20261020/`; all1170 predictions are complete. Historical failed attempts and the original offline run remain separate.
- The zero-request-budget replay is not a real LLM result:262 episodes have outstanding query coverage and128 shorter-than-window episodes issue no query by the configured cadence. Keep this distinct from measured inference.
- Latest detection suite:97 tests passed after integer/float timeout cache normalization and causal decision/message evidence-step validation. Six cache entries were recovered from original raw replies without editing content, requests or evidence. Cumulative main-run audits contain409 request attempts and379 returned responses identifying `gpt-6-sol`; cached usage is excluded from fresh-response usage totals. The four smoke replies are recorded separately.
- The actual run has262 queried episodes and128 short episodes with no query, including native-failed Manipulation292 (392frames). Do not describe390 complete episode predictions as390 actual LLM inferences. Heuristic alerts142 and LLM alerts199 are predictions, not reference labels or S/F.
- Final artifact verification passes with zero errors/warnings. Task8 remains pending independent human labels; all390 labels are blank and formal P/R is null. The full numerical objective is not yet achieved.
