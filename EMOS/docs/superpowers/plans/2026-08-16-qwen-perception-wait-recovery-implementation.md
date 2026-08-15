# Qwen Perception Wait Recovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Qwen execute the explicit Spot–drone perception goals without invalid `TARGET_*` assignments or indistinguishable fallback waits, while leaving the GPT experiment path unchanged.

**Architecture:** Add a focused Qwen perception compatibility module that extracts explicit PDDL detection goals, builds model-family prompt contracts, and validates leader assignments without synthesizing answers. Integrate bounded Qwen-only retries at leader and action protocol boundaries, give Qwen action agents a compact semantic context, validate peers/targets/premature waits, and preserve the existing GPT calls and history. Require the Qwen Hydra overlay and verify behavior on the three failing episodes at the original step limit.

**Tech Stack:** Python 3, pytest, Pydantic, Hydra/OmegaConf, OpenAI-compatible Chat Completions, Habitat Baselines, PyTorch.

## Global Constraints

- Enable every new behavior only when `EMOS_LLM_BACKEND=qwen`.
- Do not alter `habitat-baselines/habitat_baselines/config/multi_rearrange/llm_spot_drone_per.yaml`.
- Keep the existing GPT prompts, parser, context propagation, action tools, call count, and retry behavior unchanged.
- Do not hard-code an assignment, choose a robot for a goal, or repair Qwen output in host code; invalid Qwen output must be returned to Qwen for correction.
- Allow at most three total Qwen attempts at each leader/action decision: the initial response plus two corrections.
- Keep group discussion, reflection, and numerical reasoning enabled.
- Preserve `/home/dyc/Projects/FARO/EMOS/habitat-mas/habitat_mas/scene_graph/utils.py` exactly as found; it contains unrelated user work.
- Reuse the generic `multi_rearrange/llm_spot_drone_per_qwen.yaml` overlay for all Qwen variants.
- Run each TDD cycle as RED, GREEN, REFACTOR; never implement a behavior before observing its focused test fail for the expected reason.

---

### Task 1: Explicit Detection-Goal Vocabulary and Assignment Validator

**Files:**
- Create: `habitat-baselines/habitat_baselines/rl/multi_agent/qwen_perception_compat.py`
- Create: `test/test_qwen_perception_semantics.py`

**Interfaces:**
- Produces: `extract_detection_goal_objects(task_description: str) -> tuple[str, ...]`
- Produces: `extract_object_ids(text: str) -> tuple[str, ...]`
- Produces: `build_assignment_contract(robot_ids: tuple[str, ...], goal_objects: tuple[str, ...]) -> str`
- Produces: `parse_qwen_assignment(text: str) -> dict[str, str]`
- Produces: `validate_assignment(text: str, tasks: dict[str, str], robot_ids: tuple[str, ...], goal_objects: tuple[str, ...]) -> tuple[str, ...]`
- Produces: `build_assignment_correction(violations: tuple[str, ...], robot_ids: tuple[str, ...], goal_objects: tuple[str, ...]) -> str`

- [ ] **Step 1: Write extraction and contract tests**

```python
TASK = """Goal of this episode is the logical operation and of the following conditions:
0. The object 'any_targets|1' has been detected by any robot.
1. The object 'any_targets|0' has been detected by any robot."""

def test_extracts_only_explicit_detection_goal_objects():
    assert extract_detection_goal_objects(TASK) == (
        "any_targets|1",
        "any_targets|0",
    )

def test_qwen_contract_names_only_real_agents_and_goal_objects():
    contract = build_assignment_contract(
        ("agent_0", "agent_1"),
        ("any_targets|1", "any_targets|0"),
    )
    assert "agent_0, agent_1" in contract
    assert "any_targets|1, any_targets|0" in contract
    assert "TARGET_any_targets" in contract
    assert "do not invent" in contract.lower()
```

- [ ] **Step 2: Write assignment validation tests reproducing episodes 379, 272, and 185**

```python
@pytest.mark.parametrize(
    ("response", "expected_fragment"),
    [
        (
            "{agent_0||Nothing to do}"
            "{agent_1||Detect TARGET_any_targets|0 and TARGET_any_targets|1 "
            "with a second robot}",
            "not explicit goal objects",
        ),
        (
            "{agent_0||Detect TARGET_any_targets|1}"
            "{agent_1||Detect any_targets|1 and TARGET_any_targets|0}",
            "uncovered goal objects: any_targets|0",
        ),
        (
            "{agent_0||Nothing to do}{agent_1||Nothing to do}",
            "uncovered goal objects",
        ),
        (
            "{agent_0||Detect any_targets|0}"
            "{agent_1||Detect any_targets|1}"
            "{agent_2||Assist agent_1}",
            "invalid agent IDs: agent_2",
        ),
    ],
)
def test_rejects_observed_invalid_assignments(response, expected_fragment):
    tasks = parse_qwen_assignment(response)
    violations = validate_assignment(
        response,
        tasks,
        ("agent_0", "agent_1"),
        ("any_targets|1", "any_targets|0"),
    )
    assert any(expected_fragment in item for item in violations)

def test_accepts_model_generated_complete_assignment():
    response = (
        "{agent_0||Detect object any_targets|0}"
        "{agent_1||Detect object any_targets|1}"
    )
    tasks = parse_qwen_assignment(response)
    assert validate_assignment(
        response,
        tasks,
        ("agent_0", "agent_1"),
        ("any_targets|1", "any_targets|0"),
    ) == ()
```

- [ ] **Step 3: Run the focused tests and observe RED**

Run:

```bash
PYTHONPATH=habitat-baselines:habitat-lab:habitat-mas pytest -q test/test_qwen_perception_semantics.py
```

Expected: collection fails because `qwen_perception_compat` and its functions do not exist.

- [ ] **Step 4: Implement the pure compatibility module**

```python
DETECTION_GOAL_RE = re.compile(
    r"^\s*\d+\.\s+The object ['\"]([^'\"]+)['\"] "
    r"has been detected by any robot\.\s*$",
    re.MULTILINE,
)
OBJECT_ID_RE = re.compile(r"(?:TARGET_)?any_targets\|\d+")
ASSIGNMENT_RE = re.compile(r"\{([^{}|]+)\|\|([^{}]*)\}")

def extract_detection_goal_objects(task_description: str) -> tuple[str, ...]:
    return tuple(dict.fromkeys(DETECTION_GOAL_RE.findall(task_description)))

def extract_object_ids(text: str) -> tuple[str, ...]:
    return tuple(dict.fromkeys(OBJECT_ID_RE.findall(text)))

def parse_qwen_assignment(text: str) -> dict[str, str]:
    return {
        robot_id.strip(): subtask.strip()
        for robot_id, subtask in ASSIGNMENT_RE.findall(text)
    }
```

Implement `validate_assignment` as a pure function that reports invalid/missing/duplicate real IDs, goal objects absent from every non-wait subtask, and object IDs not present in `goal_objects`. It must never mutate `tasks`. Implement correction text that states violations, valid IDs, and required goals without recommending any mapping.

- [ ] **Step 5: Run the focused tests and observe GREEN**

Run the Step 3 command. Expected: all tests pass.

- [ ] **Step 6: Run formatting and diff checks**

```bash
python -m compileall -q habitat-baselines/habitat_baselines/rl/multi_agent/qwen_perception_compat.py
git diff --check -- habitat-baselines/habitat_baselines/rl/multi_agent/qwen_perception_compat.py test/test_qwen_perception_semantics.py
```

- [ ] **Step 7: Commit the isolated unit**

```bash
git add habitat-baselines/habitat_baselines/rl/multi_agent/qwen_perception_compat.py test/test_qwen_perception_semantics.py
git commit -m "feat: validate qwen perception assignments"
```

---

### Task 2: Qwen-Only Leader Retry and Robot-Reflection Semantics

**Files:**
- Modify: `habitat-baselines/habitat_baselines/rl/multi_agent/multi_llm_policy.py`
- Extend: `test/test_qwen_perception_semantics.py`
- Extend: `test/test_qwen_discussion_compat.py`

**Interfaces:**
- Consumes: all Task 1 functions.
- Produces: `request_qwen_assignment(leader: OpenAIModel, prompt: str, robot_ids: tuple[str, ...], goal_objects: tuple[str, ...], max_attempts: int = 3) -> tuple[str, dict[str, str]]`
- Produces: Qwen-only optional keyword parameters on `create_leader_prompt` and `create_robot_start_message`; calls without these parameters retain current GPT strings.

- [ ] **Step 1: Write bounded leader-retry tests**

```python
def test_qwen_leader_retries_invalid_assignment_and_accepts_model_correction():
    leader = Mock()
    leader.chat.side_effect = [
        "{agent_0||Nothing to do}{agent_1||Detect TARGET_any_targets|0}",
        "{agent_0||Detect any_targets|0}{agent_1||Detect any_targets|1}",
    ]
    response, tasks = request_qwen_assignment(
        leader,
        "initial prompt",
        ("agent_0", "agent_1"),
        ("any_targets|1", "any_targets|0"),
    )
    assert tasks == {
        "agent_0": "Detect any_targets|0",
        "agent_1": "Detect any_targets|1",
    }
    assert leader.chat.call_count == 2
    assert "uncovered goal objects" in leader.chat.call_args_list[1].args[0]

def test_qwen_leader_exhaustion_never_synthesizes_assignment():
    leader = Mock()
    leader.chat.return_value = "{agent_0||Nothing to do}{agent_1||Nothing to do}"
    with pytest.raises(QwenAssignmentValidationError):
        request_qwen_assignment(
            leader,
            "initial prompt",
            ("agent_0", "agent_1"),
            ("any_targets|1", "any_targets|0"),
        )
    assert leader.chat.call_count == 3
```

- [ ] **Step 2: Write GPT characterization and reflection-contract tests**

```python
def test_gpt_leader_and_robot_prompts_remain_exactly_unchanged(monkeypatch):
    monkeypatch.setenv("EMOS_LLM_BACKEND", "gpt")
    leader = create_leader_prompt("resume")
    robot = create_robot_start_message("task", "scene")
    assert hashlib.sha256(leader.encode()).hexdigest() == (
        "f9db97b5e3c99340501ab52f4fd273fef5c06b147b888307422a31817bc749d7"
    )
    assert hashlib.sha256(robot.encode()).hexdigest() == (
        "91561a1c94ecd933e174e461ddfda672317b73c7d719886db6c8e97ca8d8a15f"
    )

def test_qwen_robot_reflection_forbids_invented_detection_geometry(monkeypatch):
    monkeypatch.setenv("EMOS_LLM_BACKEND", "qwen")
    prompt = create_robot_start_message(
        "Detect any_targets|0",
        "scene",
        goal_objects=("any_targets|0",),
    )
    assert "Do not apply manipulation workspace" in prompt
    assert "Do not invent" in prompt
    assert "nearest navigable" in prompt
```

- [ ] **Step 3: Run focused tests and observe RED**

```bash
PYTHONPATH=habitat-baselines:habitat-lab:habitat-mas pytest -q \
  test/test_qwen_perception_semantics.py \
  test/test_qwen_discussion_compat.py
```

Expected: missing retry helper and missing Qwen semantic contract assertions.

- [ ] **Step 4: Integrate goal extraction and Qwen leader retries**

At the start of `group_discussion`, compute:

```python
robot_ids = tuple(robot_resume)
goal_objects = extract_detection_goal_objects(task_description)
```

Append `build_assignment_contract(robot_ids, goal_objects)` only to the Qwen leader prompt. Dispatch initial assignment and every reflection reassignment through `request_qwen_assignment` only for Qwen. Keep the current single `leader.chat()` plus `parse_leader_response()` statements in the GPT branch.

- [ ] **Step 5: Add the Qwen-only robot reflection contract**

Pass `goal_objects` to `create_robot_start_message` in both initial and reassignment reflection loops. Append the detection semantic rules only when the backend is Qwen and `goal_objects` is non-empty. Do not change `COMPUTE_PATH`, `COMPUTE_SPACE`, or GPT string concatenation.

- [ ] **Step 6: Run focused and existing discussion tests and observe GREEN**

Run Step 3 plus:

```bash
PYTHONPATH=habitat-baselines:habitat-lab:habitat-mas pytest -q test/test_qwen_discussion_compat.py
```

- [ ] **Step 7: Commit only scoped discussion changes**

```bash
git add habitat-baselines/habitat_baselines/rl/multi_agent/multi_llm_policy.py \
  test/test_qwen_perception_semantics.py test/test_qwen_discussion_compat.py
git commit -m "fix: retry invalid qwen task assignments"
```

---

### Task 3: Clean Qwen Action Context, Valid Peers, and Bounded Action Retry

**Files:**
- Modify: `habitat-mas/habitat_mas/utils/__init__.py`
- Modify: `habitat-mas/habitat_mas/agents/crab_agent.py`
- Modify: `habitat-mas/habitat_mas/utils/models.py`
- Modify: `habitat-baselines/habitat_baselines/rl/multi_agent/multi_llm_policy.py`
- Extend: `habitat-mas/habitat_mas/test/test_models_qwen.py`
- Create: `habitat-mas/habitat_mas/test/test_crab_agent_qwen.py`
- Extend: `test/test_qwen_llm_policy.py`

**Interfaces:**
- Extends `AgentArguments` with defaults: `scene_description: str = ""` and `peer_ids: tuple[str, ...] = ()`.
- Extends `CrabAgent.init_agent(..., scene_description: str = "", peer_ids: tuple[str, ...] = ())`.
- Produces: `QwenActionProtocolError(RuntimeError)`.
- Extends `OpenAIModel.chat(content: str, crab_planning: bool = False, action_validator: Callable[[str, dict[str, Any]], str | None] | None = None)`; GPT ignores `action_validator` because only Qwen passes it.
- Produces: `OpenAIModel._chat_qwen_action(request, action_validator, max_attempts=3) -> tuple[str, dict[str, Any]]`.

- [ ] **Step 1: Write action-protocol retry tests at the model boundary**

Use fake Chat Completions responses with `choices[0].message` and `usage.total_tokens`.

```python
def test_qwen_missing_tool_then_valid_tool_retries_instead_of_wait(qwen_model):
    qwen_model.client.chat.completions.create.side_effect = [
        completion(content="I will wait", tool_calls=None),
        completion(tool_calls=[tool_call("nav_to_obj", '{"target_obj":"any_targets|0"}')]),
    ]
    assert qwen_model.chat("act") == (
        "nav_to_obj",
        {"target_obj": "any_targets|0"},
    )
    assert qwen_model.client.chat.completions.create.call_count == 2

def test_qwen_action_retry_exhaustion_raises_diagnostic(qwen_model):
    qwen_model.client.chat.completions.create.return_value = completion(
        content="no function call",
        tool_calls=None,
    )
    with pytest.raises(QwenActionProtocolError, match="no tool call"):
        qwen_model.chat("act")
    assert qwen_model.client.chat.completions.create.call_count == 3

def test_gpt_action_call_count_and_parser_are_unchanged(gpt_model):
    gpt_model.client.chat.completions.create.return_value = completion(
        tool_calls=[tool_call("wait", "{}")]
    )
    assert gpt_model.chat("act") == ("wait", {})
    assert gpt_model.client.chat.completions.create.call_count == 1
```

- [ ] **Step 2: Write CrabAgent semantic validation and context tests**

```python
def test_qwen_action_context_omits_numerical_transcript(monkeypatch, actions):
    monkeypatch.setenv("EMOS_LLM_BACKEND", "qwen")
    agent = CrabAgent("agent_0", actions)
    agent.init_agent(
        "SpotRobot",
        TASK,
        "Detect any_targets|0",
        chat_history=[[{"role": "assistant", "content": "```python\n...\n```"}]],
        scene_description="scene description",
        peer_ids=("agent_1",),
    )
    flattened = repr(agent.llm_model.chat_history)
    assert "```python" not in flattened
    assert "Detect any_targets|0" in agent.llm_model.system_message["content"]
    assert "agent_1" in agent.llm_model.system_message["content"]

def test_qwen_rejects_unknown_peer_and_premature_wait(monkeypatch, actions):
    monkeypatch.setenv("EMOS_LLM_BACKEND", "qwen")
    agent = make_initialized_agent(actions, subtask="Detect any_targets|0")
    assert "real peer" in agent._validate_qwen_action(
        "send_request", {"request": "help", "target_agent": "agent_2"}
    )
    assert "unfinished" in agent._validate_qwen_action("wait", {})
    assert agent._validate_qwen_action(
        "nav_to_obj", {"target_obj": "any_targets|0"}
    ) is None
```

Add a GPT characterization asserting its supplied raw `chat_history` remains assigned exactly as before.

- [ ] **Step 3: Run focused tests and observe RED**

```bash
PYTHONPATH=habitat-baselines:habitat-lab:habitat-mas pytest -q \
  habitat-mas/habitat_mas/test/test_models_qwen.py \
  habitat-mas/habitat_mas/test/test_crab_agent_qwen.py \
  test/test_qwen_llm_policy.py
```

Expected: Qwen still silently returns wait, `AgentArguments` lacks context fields, and `CrabAgent` lacks validation.

- [ ] **Step 4: Implement pure Qwen action parsing and bounded retry**

Refactor `_parse_qwen_action_response` so it either returns one validated action tuple or raises `QwenActionProtocolError` with one of: no tool call, multiple tool calls, unknown tool, malformed JSON, non-object arguments, or Pydantic schema validation failure. It must not append a success tool result before validation.

Implement `_chat_qwen_action`:

```python
for attempt in range(max_attempts):
    response = self.client.chat.completions.create(...)
    try:
        action_name, parameters = self._parse_qwen_action_response(message)
        validator_error = action_validator(action_name, parameters) if action_validator else None
        if validator_error:
            raise QwenActionProtocolError(validator_error)
        self._append_success_tool_result(message.tool_calls[0])
        return action_name, parameters
    except QwenActionProtocolError as error:
        last_error = str(error)
        if attempt + 1 < max_attempts:
            request.append({"role": "user", "content": build_action_correction(last_error)})
raise QwenActionProtocolError(last_error)
```

Endpoint exceptions continue to propagate.

- [ ] **Step 5: Implement Qwen-only compact context and action validator**

In `CrabAgent.init_agent`, retain the GPT `chat_history` assignment unchanged. In Qwen mode, do not inject raw reflection history. Put task, final subtask, scene, valid peers, wait policy, and function-call requirement in the system message. Change the Qwen pre-action planning request to ask for a short ordered action list and prohibit `{{yes}}`/`{{no}}`.

Track assigned object IDs and previously returned `nav_to_obj` targets. Validate target membership, peer membership/self-request, and premature waits. Catch only `QwenActionProtocolError` in `CrabAgent.chat`, log `action_source=protocol_fallback reason=...`, and return the existing external wait. Log `action_source=model_wait` for a valid literal wait and `action_source=send_request` for a valid internal request.

- [ ] **Step 6: Thread semantic context through `AgentArguments`**

Populate `scene_description` and all other real agent IDs in every result from `group_discussion`. Pass them from `MultiLLMPolicy.act` into `CrabAgent.init_agent`. Defaults preserve every non-Qwen caller.

- [ ] **Step 7: Run focused tests and observe GREEN**

Run Step 3. Also run:

```bash
PYTHONPATH=habitat-baselines:habitat-lab:habitat-mas pytest -q \
  habitat-mas/habitat_mas/test/test_models_qwen_runtime.py \
  test/test_qwen_discussion_compat.py
```

- [ ] **Step 8: Commit the action protocol unit without user-owned scene graph changes**

```bash
git add habitat-mas/habitat_mas/utils/__init__.py \
  habitat-mas/habitat_mas/agents/crab_agent.py \
  habitat-mas/habitat_mas/utils/models.py \
  habitat-baselines/habitat_baselines/rl/multi_agent/multi_llm_policy.py \
  habitat-mas/habitat_mas/test/test_models_qwen.py \
  habitat-mas/habitat_mas/test/test_crab_agent_qwen.py \
  test/test_qwen_llm_policy.py
git commit -m "fix: recover qwen action protocol without silent waits"
```

---

### Task 4: Qwen Perception Overlay Guard and Configuration Equivalence

**Files:**
- Extend: `habitat-baselines/habitat_baselines/rl/multi_agent/qwen_perception_compat.py`
- Modify: `habitat-baselines/habitat_baselines/rl/multi_agent/multi_llm_policy.py`
- Extend: `test/test_qwen_perception_config.py`
- Extend: `test/test_qwen_perception_semantics.py`

**Interfaces:**
- Produces: `validate_qwen_perception_config(config_name: str, goal_objects: tuple[str, ...], exposed_actions: dict[str, tuple[str, ...]]) -> None`
- Raises: `QwenPerceptionConfigError` with the required overlay name when a detection-only Qwen run exposes `pick`, `place`, or `reset_arm`.

- [ ] **Step 1: Write fail-fast and valid-overlay tests**

```python
def test_qwen_perception_base_config_fails_fast():
    with pytest.raises(QwenPerceptionConfigError, match="llm_spot_drone_per_qwen"):
        validate_qwen_perception_config(
            "multi_rearrange/llm_spot_drone_per",
            ("any_targets|0",),
            {"agent_0": ("nav_to_obj", "pick", "place", "wait")},
        )

def test_qwen_overlay_perception_actions_are_accepted():
    validate_qwen_perception_config(
        "multi_rearrange/llm_spot_drone_per_qwen",
        ("any_targets|0",),
        {
            "agent_0": ("send_request", "nav_to_obj", "wait"),
            "agent_1": ("send_request", "nav_to_obj", "wait"),
        },
    )
```

- [ ] **Step 2: Run focused config tests and observe RED**

```bash
PYTHONPATH=habitat-baselines:habitat-lab:habitat-mas pytest -q \
  test/test_qwen_perception_config.py test/test_qwen_perception_semantics.py
```

- [ ] **Step 3: Implement and call the guard only for Qwen**

At first-step discussion, obtain `HydraConfig.get().job.config_name` and action names from each active policy's `llm_agent.actions`. Run the guard only when backend is Qwen and explicit detection goals were extracted. Log config name and exposed action names. GPT does not call the helper.

- [ ] **Step 4: Prove resolved configuration equivalence**

```bash
PYTHONPATH=habitat-baselines:habitat-lab:habitat-mas python -m habitat_baselines.run \
  --config-name=multi_rearrange/llm_spot_drone_per.yaml --cfg job > /tmp/emos-gpt-config.yaml
PYTHONPATH=habitat-baselines:habitat-lab:habitat-mas python -m habitat_baselines.run \
  --config-name=multi_rearrange/llm_spot_drone_per_qwen.yaml --cfg job > /tmp/emos-qwen-config.yaml
```

Use a read-only diff and confirm the only intended semantic difference is Qwen Spot's manipulation `ignore_skills`; inherited dataset, sensors, PDDL goal, agents, maximum steps, and metrics match.

- [ ] **Step 5: Run focused tests and commit**

```bash
git add habitat-baselines/habitat_baselines/rl/multi_agent/qwen_perception_compat.py \
  habitat-baselines/habitat_baselines/rl/multi_agent/multi_llm_policy.py \
  test/test_qwen_perception_config.py test/test_qwen_perception_semantics.py
git commit -m "fix: require qwen perception overlay"
```

---

### Task 5: Full Regression and Real Qwen Comparability Verification

**Files:**
- Modify only scoped files when a newly added reproducing test proves a defect.
- Do not commit generated logs, videos, images, or chat histories.

**Interfaces:**
- Consumes: completed Tasks 1–4.
- Produces: authoritative test output, resolved-config comparison, and runtime logs proving completion criteria.

- [ ] **Step 1: Run all focused compatibility tests**

```bash
PYTHONPATH=habitat-baselines:habitat-lab:habitat-mas pytest -q \
  habitat-mas/habitat_mas/test/test_llm_backend.py \
  habitat-mas/habitat_mas/test/test_models_qwen.py \
  habitat-mas/habitat_mas/test/test_models_qwen_runtime.py \
  habitat-mas/habitat_mas/test/test_python_interpreter_detailed.py \
  habitat-mas/habitat_mas/test/test_crab_agent_qwen.py \
  test/test_qwen_discussion_compat.py \
  test/test_qwen_llm_policy.py \
  test/test_qwen_perception_config.py \
  test/test_qwen_perception_semantics.py
```

Expected: all tests pass with no skips related to changed behavior.

- [ ] **Step 2: Run compile and worktree safety checks**

```bash
python -m compileall -q \
  habitat-mas/habitat_mas/utils \
  habitat-mas/habitat_mas/agents/crab_agent.py \
  habitat-baselines/habitat_baselines/rl/multi_agent \
  habitat-baselines/habitat_baselines/rl/hrl/hl/llm_policy.py
git diff --check
git status --short
git diff -- habitat-mas/habitat_mas/scene_graph/utils.py
```

Confirm the scene graph diff is the same unrelated user change found before execution.

- [ ] **Step 3: Verify the local server model ID without changing the compatibility profile**

Query `http://127.0.0.1:8000/v1/models`. Set `EMOS_LLM_MODEL` only to an ID actually returned; otherwise leave it unset so the current server alias `gpt-5.5` remains the request model. Keep `EMOS_LLM_BACKEND=qwen` in either case.

- [ ] **Step 4: Run the three failing episodes with the original episode step limit**

```bash
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY
unset http_proxy https_proxy all_proxy
export OPENAI_BASE_URL=http://127.0.0.1:8000/v1
export OPENAI_API_KEY=local-qwen
export EMOS_LLM_BACKEND=qwen
set -o pipefail

python -u -m habitat_baselines.run \
  --config-name=multi_rearrange/llm_spot_drone_per_qwen.yaml \
  habitat_baselines.evaluate=True \
  habitat_baselines.num_environments=1 \
  habitat_baselines.test_episode_count=3 \
  2>&1 | tee "logs/llm_spot_drone_per_qwen_comparable_$(date +%Y%m%d_%H%M%S).log"
```

Do not override `habitat.environment.max_episode_steps`; the resolved base value is part of the GPT experiment setting.

- [ ] **Step 5: Audit the real Qwen log**

Extract episode IDs, final task assignments, `llm_output`, `action_source`, per-stage goals, `pddl_success`, and `num_steps`. Assert from the log and saved raw histories:

- only `agent_0` and `agent_1` appear as assignment or request recipients;
- only explicit `any_targets|*` goal objects are assigned for detection;
- no action navigates to irrelevant `TARGET_any_targets|*` markers;
- unfinished goals cause valid `nav_to_obj` calls rather than model waits;
- protocol fallbacks, internal requests, and model waits are separately observable; and
- at least one episode has `pddl_success=1`.

If any assertion fails, add one focused failing test that reproduces the exact raw response before modifying code, then repeat Steps 1–5.

- [ ] **Step 6: Expand to a multi-episode Qwen sample**

Run at least 10 episodes with the identical Qwen command except `habitat_baselines.test_episode_count=10`. Record success rate, average steps, action counts, fallback counts, and token usage. Do not claim full-dataset performance from this sample.

- [ ] **Step 7: Prove GPT isolation**

Unset `EMOS_LLM_BACKEND` and `EMOS_LLM_MODEL`, compose the original GPT YAML, and rerun all GPT characterization tests. If a valid GPT credential is available, run the same episode count and order using the original command. If the configured key is rejected, record the endpoint error and state that configuration comparability is proven but empirical GPT score comparison remains externally unverified.

- [ ] **Step 8: Invoke verification-before-completion and perform the requirement audit**

Map each completion criterion in `docs/superpowers/specs/2026-08-16-qwen-perception-comparability-design.md` to a current test result, resolved config, runtime log line, or explicit credential limitation. Do not mark the goal complete while any runtime or GPT-isolation requirement lacks evidence.

- [ ] **Step 9: Commit final scoped fixes only if runtime testing required them**

Stage explicit scoped paths. Never stage generated logs/chat histories or `habitat-mas/habitat_mas/scene_graph/utils.py`.
