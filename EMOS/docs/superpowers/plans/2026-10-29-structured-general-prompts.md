# Structured General Prompts Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans for this prompt-only change. Track checks below.

**Goal:** Apply the supplied general-instruction and structured-placeholder requirements to detection and recovery.

**Architecture:** Keep current request serialization, input paths, filtering and output contracts; change prompt constants and publish their placeholder mapping.

**Tech Stack:** Python string constants, Markdown, existing unittest transport fakes, AST/request comparison.

**Spec:** `docs/superpowers/specs/2026-10-29-structured-general-prompts-design.md`.

## Constraints and review focus

Preserve the dirty worktree and existing APIs; no paid requests, simulator runs, dependencies or commits. Keep detection unprejudiced, source-input current observations admissible, initial/current assignments distinct, omission/null unknown, and no claim of a measured token budget or optimal filter.

## Task 1: Fixed instructions and explicit input templates

- [x] Read the existing detector, source/extension generators and context/filter adapters; save originals under `/tmp/emos-general-prompt-20261029`.
- [x] Run existing focused contracts before editing: 64 tests passed.
- [x] Revise only prompt assignments in `detection/llm.py`, `recovery/source_generator.py`, `recovery/generator.py`; bump prompt version.
- [x] Add `docs/llm_prompt_templates.md` and link it from detection/source/combined/extension docs.
- [x] Verify prompt-only AST changes, unchanged real user-message payloads and request options, and template mappings.
- [x] Run detection and recovery regression tests and review this change independently.

Prompt content is configuration, not a new parser or runtime feature; use the existing behavioral tests and independent content review rather than keyword-mirroring tests.

## Verification results

- Final detection/recovery regression: 316 tests passed, run through `/tmp/emos-general-prompt-20261029/run_regression.py` under the Habitat Python environment. The initial stdin-based invocation caused five multiprocessing worker errors because spawn could not reopen `<stdin>`; running a real main-guarded file resolved all five without product changes.
- All three Python modules have identical non-prompt ASTs to their saved session baselines. Representative detection, combined-source and extension requests have unchanged user messages and request options; the caller's source context remains unchanged.
- Both documented placeholder templates parse after JSON substitution and match existing field names; their local links and whitespace checks pass. Detector version is `runtime-system-error-v5-structured-general`.
- Independent review found one ambiguity: compatibility recovery can put detector output in `error_report`, while combined checks use a real exception or null. Prompt and guide now distinguish report provenance and actual facts. No other important findings.
- Evidence is in `/tmp/emos-general-prompt-20261029/verification.json`, baseline diffs and regression logs. No real API requests or simulator episodes were run; live-model accuracy and optimal trajectory selection remain unmeasured.

## Task 2: User-requested single detection-and-recovery instruction

- [x] Audit source generator references and compatibility response formats; save the current files under `/tmp/emos-unified-prompt-20261029`.
- [x] Replace concatenated role prompts with one general instruction; keep the older `generate()` output format as a small adapter using the same instruction.
- [x] Rewrite the guide around one main instruction and structured input; update the combined-path documentation.
- [x] Verify unchanged request data, parsers and control flow; run existing related contracts and independently review prompt semantics.

This follow-up changes prompt configuration and documentation, with no new runtime behavior or keyword-based tests.

### Follow-up verification

Focused baseline: 53 tests passed. Final source, combined, model-input, extension and independent-detector contracts: 75 tests passed. AST comparison confirms unchanged parsers, request construction and control flow after normalizing prompt assignments, the removed prompt import and prompt selection. Local fake transports confirm one request per call and identical user payload/options/results for legacy generation and all three combined outcomes. Template fields and links remain valid.

Independent review identified an omitted evidence boundary: suspicious or unexecuted code must not by itself establish a runtime anomaly. The unified instruction and guide now explicitly require visible runtime observations or an actual execution exception for classification, while allowing source to explain observed behavior. Final evidence: `/tmp/emos-unified-prompt-20261029/verification.json` and `regression-final.log`. No real API requests or simulator episodes were run.
