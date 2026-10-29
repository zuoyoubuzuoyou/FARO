# Boolean LLM Output Implementation Plan

> **For agentic workers:** Track test-first execution and shared consumer checks below.

**Goal:** Request only a strict system-error boolean from the model.

**Architecture:** Strict wire parser normalizes the boolean to existing internal statuses; unchanged consumers receive program-supplied step and detector metadata.

**Spec:** `docs/superpowers/specs/2026-10-28-binary-llm-output-design.md`.

## Constraints

Preserve all existing unrelated edits, input formats, historical requests and credentials. No network/API calls, simulation, commits or fabricated model explanation/evidence.

## Steps

- [x] Audit runner/recovery consumers and obtain RED boolean-output tests before production changes.
- [x] Simplify prompt and parser, bump version and retain explicit request/parse/budget failures.
- [x] Migrate transport fixtures and verify LLM, runner and recovery integration scopes.
- [x] Independently review implementation, update documentation and finish broader regression checks.

## Results

- User authorized the single-field boolean response. Prompt version is `runtime-system-error-v4-binary`; no input schema or classification-policy change was introduced.
- RED: old parser rejected minimal boolean replies in LLM, runner and recovery scopes. GREEN: 22 LLM, 13 runner and 2 recovery end-to-end tests pass. Obsolete rich-output/evidence validation tests were replaced by strict boolean, schema rejection and minimal-report compatibility tests.
- Full discovery: 478 entries, 448 passing cases, zero executed case failures; 30 pre-existing import errors from missing pytest (28) and mock (2). Full log is preserved with exact module names.
- Offline real-request check verifies unchanged input payload, no reuse of the previous rich-output cache, no false boolean on budget failure, and unchanged historical audit/cache bytes. No live LLM request or simulation.
- Independent review found no blocking regressions. Evidence: `analysis_results/detection/binary_output_20261028/verification.json` and `full_test_results.log`. Whitespace checks pass.
