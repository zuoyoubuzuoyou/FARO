# Qwen Compatibility Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make the Spot–drone perception evaluation robust across the Qwen model family while preserving the existing GPT configuration and runtime behavior.

**Architecture:** Add an explicit `EMOS_LLM_BACKEND` compatibility profile and keep request-model selection independent through `EMOS_LLM_MODEL`. A single Qwen Hydra overlay removes perception-incompatible Spot skills, while small Qwen-only branches filter action tools, strengthen the numerical prompt/parser, and bound generated-Python repair. Existing GPT branches and the base YAML remain unchanged.

**Tech Stack:** Python, pytest, Hydra/OmegaConf, OpenAI-compatible Chat Completions, Habitat Baselines, subprocess.

## Global Constraints

- Preserve the existing uncommitted user changes in multi_llm_policy.py, models.py, and scene_graph/utils.py; edit only relevant hunks and inspect every diff. Do not create implementation commits because staging either overlapping file would also stage user-owned changes.
- Do not alter `multi_rearrange/llm_spot_drone_per.yaml`.
- Do not infer the backend from `OPENAI_BASE_URL` or a model name.
- Do not hard-code `qwen-3.5-9b`; every Qwen model uses the same `qwen` compatibility profile.
- Keep GPT prompts, parser semantics, unfiltered action pool, interpreter `run()` behavior, and unbounded planning loop exactly as they are today.
- Follow strict TDD for each behavior: add the focused test, observe the expected failure, implement the minimum code, then rerun the focused test.

### Task 1: Runtime profile and independent request-model selection

**Files:**

- Create: `habitat-mas/habitat_mas/utils/llm_backend.py`
- Create: `habitat-mas/habitat_mas/test/test_llm_backend.py`
- Modify: `habitat-mas/habitat_mas/utils/models.py`
- Create: `habitat-mas/habitat_mas/test/test_models_qwen.py`

1. Add tests proving that an unset or mixed-case `gpt` backend resolves to `gpt`, mixed-case/whitespace `qwen` resolves to `qwen`, and an unknown value raises `ValueError` containing the invalid value.
2. Run `pytest -q habitat-mas/habitat_mas/test/test_llm_backend.py` and confirm import/test failure because the helper does not exist.
3. Implement `get_llm_backend()` in the new shared module with explicit `gpt`/`qwen` validation.
4. Add model-wrapper tests with the OpenAI client patched at its external-construction boundary: unset `EMOS_LLM_MODEL` preserves constructor argument `gpt-5.5`; a non-empty value overrides only `self.model`; `EMOS_LLM_BACKEND=qwen` alone does not change the request model; `EMOS_LLM_MODEL` alone does not enable Qwen behavior.
5. Run the focused model tests and confirm they fail on the missing override.
6. Update `OpenAIModel.__init__` to resolve `self.backend = get_llm_backend()` and `self.model = os.getenv("EMOS_LLM_MODEL", "").strip() or model`. Log the selected backend and request model at initialization without printing credentials or URLs.
7. Rerun both focused files until green.
8. Inspect this task focused diff and record a checkpoint; do not stage or commit the overlapping models.py file.

### Task 2: Generic Qwen perception overlay

**Files:**

- Create: `habitat-baselines/habitat_baselines/config/multi_rearrange/llm_spot_drone_per_qwen.yaml`
- Create: `test/test_qwen_perception_config.py`

1. Add a Hydra composition test using `habitat_baselines.config.default.get_config` that loads both the base and Qwen configs. Assert the base Spot configuration still exposes `pick`/`place` skills as before, while the Qwen overlay adds `pick`, `pick_at_position`, `place`, `place_at_position`, and `reset_arm` to Spot's `ignore_skills`. Assert Drone's configuration is unchanged between the two.
2. Run `pytest -q test/test_qwen_perception_config.py` and confirm failure because the Qwen config is missing.
3. Add the one-file overlay inheriting `llm_spot_drone_per`, repeating the original cabinet ignore list and appending the five manipulation skills. Do not duplicate benchmark, PDDL, or skill-definition YAML.
4. Rerun the focused config test and additionally execute a command-line Hydra compose (`python -m habitat_baselines.run --config-name=multi_rearrange/llm_spot_drone_per_qwen.yaml --cfg job`) to catch defaults/package errors without launching simulation.
5. Inspect the overlay/config-test diff and record a checkpoint without committing.

### Task 3: Per-agent Qwen action filtering and safe forbidden-action fallback

**Files:**

- Modify: `habitat-baselines/habitat_baselines/rl/hrl/hl/llm_policy.py`
- Create: `test/test_qwen_llm_policy.py`

1. Add pure behavior tests for a small action-selection helper: GPT receives the original `ACTION_POOL` object unchanged; Qwen keeps `send_request` plus only action names present in `_skill_name_to_idx`; Qwen Spot without manipulation never receives `pick`/`place`/`reset_arm`; Qwen Drone receives only its real skills plus `send_request`.
2. Add a policy-level test showing a Qwen-produced forbidden `place` call with missing place arguments falls back to `wait` without calling `_parse_function_call_args`; this catches the original pre-validation `KeyError` path.
3. Run the focused file and confirm failures on unfiltered tools and pre-validation parsing.
4. Implement a module-level `get_llm_actions(action_pool, skill_names, backend)` helper and use it only during `LLMHighLevelPolicy` initialization. Preserve the exact unfiltered object in GPT mode.
5. In `get_next_skill`, check Qwen action validity before parsing action-specific arguments; retain the current GPT statement order and fallback behavior.
6. Rerun the focused tests and any existing high-level-policy tests.
7. Inspect the policy/test diff and record a checkpoint without committing.

### Task 4: Detailed Python execution boundary

**Files:**

- Modify: `habitat-mas/habitat_mas/utils/python_interpreter.py`
- Create: `habitat-mas/habitat_mas/test/test_python_interpreter_detailed.py`

1. Add real subprocess tests for `run_python_detailed`: successful stdout and zero return code; compile-time `SyntaxError` without launching; runtime exception captured separately in stderr; timeout after a short test-specific timeout; unsupported code type rejection if the public interface accepts a type; and temporary-file cleanup for success, exception, and timeout.
2. Add a characterization test proving legacy `run()` still returns its current combined string format so GPT behavior cannot drift.
3. Run the focused file and confirm the new interface is absent.
4. Add immutable `ExecutionResult(returncode, stdout, stderr, timed_out)` and a separate `run_python_detailed(code, timeout_seconds=10)` method. Compile before creating/launching a file, execute with `subprocess.Popen.communicate(timeout=...)`, kill and collect on timeout, and unlink in `finally`.
5. Do not modify `run()` or `run_file()` except for imports shared by the new method.
6. Rerun the focused tests and the habitat-mas test subset.
7. Inspect the interpreter/test diff and record a checkpoint without committing.

### Task 5: Qwen-only bounded numerical repair loop

**Files:**

- Modify: `habitat-mas/habitat_mas/utils/models.py`
- Extend: `habitat-mas/habitat_mas/test/test_models_qwen.py`

1. Build complete fake Chat Completions responses at the network boundary (message content, tool calls, choices, usage). Add Qwen planning tests for: first-attempt success followed by a final decision; syntax failure then repair; runtime failure then repair; timeout then repair; exactly three failed code attempts returning `{{no||...}}`; and no fourth API call/code execution after exhaustion.
2. Add Qwen action-stage tests showing zero tool calls returns a safe `wait`, multiple tool calls execute only the first, malformed first-call JSON returns `wait`, and an unknown tool name returns `wait` instead of escaping.
3. Add GPT characterization tests around the same seams, proving GPT still calls legacy `interpreter.run`, preserves its existing loop, and preserves normal first-tool-call output.
4. Run the focused tests and confirm Qwen cases fail while GPT characterization passes.
5. Extract the current GPT planning body without semantic changes only if needed for readability. Add a Qwen-only planning path with a maximum of three generated-code executions per `chat()` call. Use `run_python_detailed`, send concise syntax/runtime/timeout feedback, request a code-free final decision after success, and return conservative `{{no||Numerical verification failed after 3 attempts}}` on exhaustion.
6. Add Qwen-only action-response validation before indexing or JSON decoding. Do not swallow OpenAI endpoint exceptions.
7. Rerun focused and combined habitat-mas tests.
8. Inspect the model/test diff and record a checkpoint; do not stage or commit the overlapping models.py file.

### Task 6: Qwen prompt contract and response parser

**Files:**

- Modify: `habitat-baselines/habitat_baselines/rl/multi_agent/multi_llm_policy.py`
- Create: `test/test_qwen_discussion_compat.py`

1. Add prompt tests proving GPT prompt output is byte-for-byte equal to a checked-in literal fixture for representative inputs, while Qwen prompt includes the one-block, imports-first, standard-library-only, minimal-print, repair-only, and code-free-final-decision contract.
2. Add table-driven parser tests for Qwen: `{{yes}}`, `{yes}`, bare final-line `yes`, case variants, quoted/fenced variants, `{{no||reason}}`, `{no||reason}`, bare final-line `no||reason`, last-braced-marker wins, and reasoning-body occurrences do not count. Add GPT characterization cases showing strict lowercase single-brace behavior remains unchanged.
3. Run the focused file and confirm Qwen variants fail.
4. Append the Qwen prompt contract only when `get_llm_backend() == "qwen"`. Add a separate Qwen parser helper and dispatch from `parse_agent_response`; leave the existing GPT regex body intact.
5. Ensure both reflection call sites continue to use the same dispatching parser.
6. Rerun focused tests and the group-discussion test subset.
7. Inspect the discussion/test diff and record a checkpoint; do not stage or commit the overlapping multi_llm_policy.py file.

### Task 7: Full verification and real model smoke tests

**Files:**

- Modify only if a test exposes a defect in the scoped compatibility code.
- Record commands/results in the final handoff; do not commit generated logs or chat histories.

1. Run all new focused tests in one command and record the pass count.
2. Run existing targeted regression tests for the modified modules and `python -m compileall` on all changed Python files.
3. Compose both Hydra configs and diff the relevant sections to verify only Qwen Spot skill exclusions differ.
4. Start with the current local Qwen endpoint configuration, set `EMOS_LLM_BACKEND=qwen`, optionally set `EMOS_LLM_MODEL=qwen-3.5-9b` only if `/v1/models` confirms that ID, and run a short single-environment evaluation with the Qwen overlay. Use `set -o pipefail` with `tee` and retain the log path for analysis.
5. Search the Qwen smoke log for the prior failure signatures: `object_to_goal_distance_sensor`, `Traceback`, generated-code `SyntaxError`/`NameError`/`TypeError`/`ValueError`/`KeyError`/`IndexError`, `No match found in agent response`, and forbidden Spot `pick`/`place` execution. If a scoped defect remains, add a reproducing test before fixing it and repeat the smoke run.
6. Run a GPT smoke evaluation with `EMOS_LLM_BACKEND` and `EMOS_LLM_MODEL` unset and the original config. Confirm original GPT tools/prompt/parser behavior and absence of compatibility-branch regressions. If remote credentials are unavailable, report that exact external verification gap rather than claiming it passed.
7. Inspect `git diff --check`, `git status --short`, and focused diffs. Confirm `scene_graph/utils.py` and unrelated user hunks remain untouched.
8. Use superpowers:verification-before-completion before declaring success. Leave implementation changes uncommitted so the existing user hunks stay under user control.

## Completion Criteria

- One generic Qwen YAML supports future Qwen models without duplication.
- Qwen Spot cannot see or execute perception-incompatible manipulation actions.
- Qwen generated Python is compile-checked, timed out, repaired at most twice, and cannot terminate evaluation after exhaustion.
- Qwen decision variants parse deterministically without accepting incidental reasoning text.
- Original GPT config and all GPT-characterization tests remain unchanged and green.
- A real Qwen smoke log contains none of the previously observed fatal/error signatures, or any remaining external blocker is named with evidence.
