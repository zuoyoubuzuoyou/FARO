# Qwen Single-Pipe Assignment Compatibility Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the Qwen perception leader accept the observed braced single-pipe assignment format and return semantic correction feedback instead of aborting with false missing-agent errors.

**Architecture:** Extend only the pure Qwen assignment-entry parser with a non-overlapping braced single-pipe expression. Feed its results through the unchanged semantic validator and bounded retry loop; keep GPT and deterministic-fallback behavior untouched.

**Tech Stack:** Python 3.9, `re`, pytest, Habitat Baselines, OpenAI-compatible Chat Completions.

## Global Constraints

- Enable the behavior only through the existing Qwen assignment parser.
- Accept braced `{agent_id||subtask}` and `{agent_id|subtask}` without parsing either response twice.
- Preserve pipes inside subtask object IDs such as `any_targets|0`.
- Do not add braceless single-pipe parsing because an isolated object ID is ambiguous in that form.
- Keep all existing semantic checks and the five-attempt exhaustion exception.
- Do not synthesize or hard-code an assignment.
- Do not modify the GPT parser, prompts, call count, YAML, or runtime branch.
- Preserve `habitat-mas/habitat_mas/scene_graph/utils.py` and untracked logs exactly as found.
- Execute RED, GREEN, and full verification before committing production changes.

---

### Task 1: Parse Braced Single-Pipe Qwen Assignments

**Files:**
- Modify: `test/test_qwen_perception_semantics.py`
- Modify: `habitat-baselines/habitat_baselines/rl/multi_agent/qwen_perception_compat.py`

**Interfaces:**
- Consumes: `parse_qwen_assignment(text: str) -> dict[str, str]`
- Consumes: `validate_assignment(text: str, tasks: dict[str, str], robot_ids: tuple[str, ...], goal_objects: tuple[str, ...]) -> tuple[str, ...]`
- Preserves both signatures and extends accepted input syntax only.

- [ ] **Step 1: Add a failing parser behavior test**

Append this test after the existing complete-assignment test:

```python
def test_accepts_qwen_braced_single_pipe_assignment_format():
    response = (
        "{agent_0|Detect object any_targets|0}\n"
        "{agent_1|Detect object any_targets|1}"
    )

    tasks = parse_qwen_assignment(response)

    assert tasks == {
        "agent_0": "Detect object any_targets|0",
        "agent_1": "Detect object any_targets|1",
    }
    assert validate_assignment(response, tasks, ROBOT_IDS, GOAL_OBJECTS) == ()
```

This test catches removal or omission of the braced single-pipe parser. Its
expected dictionary and empty violation tuple are hand-derived from the literal
episode response shape.

- [ ] **Step 2: Add a failing episode-219 diagnostic test**

```python
def test_single_pipe_assignment_reports_semantics_instead_of_missing_agents():
    response = (
        "{agent_0|Detect and rearrange object any_targets|0 to its target location}\n"
        "{agent_1|Detect and rearrange object any_targets|1 to its target location}"
    )

    violations = validate_assignment(
        response,
        parse_qwen_assignment(response),
        ROBOT_IDS,
        GOAL_OBJECTS,
    )

    assert not any("missing agent IDs" in item for item in violations)
    assert (
        "manipulation actions are invalid for detection-only assignments: "
        "agent_0, agent_1"
    ) in violations
```

This test catches a regression where the parser again masks the real semantic
violation with a false structural violation.

- [ ] **Step 3: Run the two focused tests and observe RED**

Run:

```bash
PYTHONPATH=habitat-baselines:habitat-lab:habitat-mas \
pytest -q test/test_qwen_perception_semantics.py -k single_pipe
```

Expected: both tests fail because `parse_qwen_assignment()` returns an empty
dictionary for `{agent_id|subtask}`.

- [ ] **Step 4: Implement the non-overlapping single-pipe parser**

Add next to the existing assignment expressions:

```python
SINGLE_PIPE_ASSIGNMENT_RE = re.compile(
    r"\{([^{}|]+)\|(?!\|)([^{}]*)\}"
)
```

Update `_assignment_entries()` so already parsed forms are removed before the
next expression runs:

```python
def _assignment_entries(text: str) -> list[tuple[str, str]]:
    entries = ASSIGNMENT_RE.findall(text)
    remaining_text = ASSIGNMENT_RE.sub("", text)
    entries.extend(SINGLE_PIPE_ASSIGNMENT_RE.findall(remaining_text))
    remaining_text = SINGLE_PIPE_ASSIGNMENT_RE.sub("", remaining_text)
    entries.extend(BARE_ASSIGNMENT_RE.findall(remaining_text))
    return [
        (robot_id.strip(), subtask.strip())
        for robot_id, subtask in entries
    ]
```

The negative lookahead prevents the new expression from consuming a
double-pipe separator. The subtask group permits pipes so object IDs remain
intact.

- [ ] **Step 5: Run focused and semantics tests and observe GREEN**

Run:

```bash
PYTHONPATH=habitat-baselines:habitat-lab:habitat-mas \
pytest -q test/test_qwen_perception_semantics.py -k single_pipe

PYTHONPATH=habitat-baselines:habitat-lab:habitat-mas \
pytest -q test/test_qwen_perception_semantics.py
```

Expected: 2 focused tests pass, followed by the complete semantics file with no
failures.

- [ ] **Step 6: Run Qwen and GPT compatibility regressions**

Run:

```bash
PYTHONPATH=habitat-baselines:habitat-lab:habitat-mas pytest -q \
  test/test_qwen_discussion_compat.py \
  test/test_qwen_llm_policy.py \
  test/test_qwen_perception_config.py \
  test/test_qwen_perception_semantics.py \
  habitat-mas/habitat_mas/test/test_models_qwen.py \
  habitat-mas/habitat_mas/test/test_models_qwen_runtime.py
```

These files cover the Qwen-only branch and existing GPT characterizations.

- [ ] **Step 7: Run static verification**

```bash
python -m compileall -q \
  habitat-baselines/habitat_baselines/rl/multi_agent/qwen_perception_compat.py \
  test/test_qwen_perception_semantics.py

git diff --check -- \
  habitat-baselines/habitat_baselines/rl/multi_agent/qwen_perception_compat.py \
  test/test_qwen_perception_semantics.py
```

- [ ] **Step 8: Commit only the parser and regression tests**

```bash
git add \
  habitat-baselines/habitat_baselines/rl/multi_agent/qwen_perception_compat.py \
  test/test_qwen_perception_semantics.py

git diff --cached --check
git diff --cached --name-only
git commit -m "fix: parse qwen single-pipe assignments"
```

- [ ] **Step 9: Run a one-episode Qwen integration smoke**

First confirm the active environment identifies Qwen and the local endpoint:

```bash
env | grep -E '^(EMOS_LLM_BACKEND|EMOS_LLM_MODEL|OPENAI_BASE_URL)='
```

Then run:

```bash
set -o pipefail
python -u -m habitat_baselines.run \
  --config-name=multi_rearrange/llm_spot_drone_per_qwen.yaml \
  habitat_baselines.evaluate=True \
  habitat_baselines.num_environments=1 \
  habitat_baselines.test_episode_count=1 \
  2>&1 | tee "logs/llm_spot_drone_per_qwen_single_pipe_smoke_$(date +%Y%m%d_%H%M%S).log"
```

Expected: the command exits zero after one episode, the Qwen overlay guard lists
only perception actions, HTTP requests return 200, and the log contains no
`QwenAssignmentValidationError`, traceback, or broken pipe.

- [ ] **Step 10: Audit the final diff and completion criteria**

```bash
git status --short --branch
git diff HEAD~2 --name-only
git diff HEAD~2 -- \
  habitat-baselines/habitat_baselines/config/multi_rearrange/llm_spot_drone_per.yaml \
  habitat-mas/habitat_mas/scene_graph/utils.py
```

Confirm the GPT YAML has no diff, the pre-existing `utils.py` diff is unchanged,
and the exact episode-219 response is protected by the new regression test.
