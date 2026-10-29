# General LLM Prompt Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans for this small inline change. Track the checks below.

**Goal:** Make the monitoring instructions benchmark-neutral while preserving the current data and runtime contracts.

**Architecture:** Replace only the system prompt and its cache-version constant; derive task semantics from the provided task. Preserve all existing Python behavior.

**Tech Stack:** Python prompt constants, existing unittest contracts, local cache/request verification.

**Spec:** `docs/superpowers/specs/2026-10-28-general-llm-prompt-design.md`.

## Constraints

User explicitly chose prompt-only work. Preserve the dirty worktree, input format, API credentials, historical outputs and original trajectories. No new network requests, simulator runs or commits.

## Steps

- [x] Audit the prompt and record baseline request/AST; run existing LLM tests.
- [x] Replace benchmark-specific instructions with task-derived monitoring policy; bump the prompt version.
- [x] Update the documentation and independently review the new instructions.
- [x] Verify unchanged non-prompt AST and observation payload, old-cache isolation, existing detection contracts and whitespace.

## Execution results

- Version: `runtime-system-error-v3-general`. Python behavior outside the two prompt constants is unchanged, verified by matching AST hashes under Habitat Python 3.9.
- Baseline LLM contracts: 25 passed. Final detection contracts: 124 passed. Prompt content was independently reviewed; no blocking findings.
- Reconstructed the previous real episode61 request exactly before editing. After editing, its observational user message and request options remain identical, the prior prompt cache is not reused, and both historical request and cache bytes remain unchanged. Verification used a zero-request budget and a client with no network transport.
- This is a prompt-data change. Existing tests and the local request/cache check validate compatibility, not semantic accuracy of model judgments; no keyword-mirroring tests or paid model reruns were added.
- Evidence: `analysis_results/detection/prompt_generalization_20261028/verification.json` and `code_review.md`. No simulation, external API request, input-format change or commit.
