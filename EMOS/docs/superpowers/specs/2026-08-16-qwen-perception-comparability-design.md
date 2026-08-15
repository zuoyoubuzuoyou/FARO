# Qwen Perception Comparability and Wait Recovery Design

## Objective

Make the Qwen Spot–drone perception evaluation execute the intended detection task instead of degenerating into `wait`, while preserving the original GPT experiment path and keeping results scientifically comparable.

Comparability means that GPT and Qwen use the same dataset, episodes, sensors, agents, PDDL goals, environment dynamics, skill implementations, maximum episode steps, and evaluation metrics. Qwen may receive model-family-specific protocol instructions, output validation, and bounded retries. The implementation must not hard-code a task assignment or complete a planning decision on Qwen's behalf.

## Evidence and Root Cause

The authoritative failing run is:

`logs/llm_spot_drone_per_20260816_031502.log`

Its raw action histories are stored under:

`chat_history_output/2026-08-16/llm_spot_drone_per/FULL/{379,272,185}`

The histories prove that the observed actions were explicit tool calls rather than missing-tool-call parser fallbacks:

- Episode 379 assigned Spot `Nothing to do`. Qwen assigned the drone to `TARGET_any_targets|0/1`, although the PDDL goals required detection of `any_targets|0/1`, and referred to a nonexistent second drone. The drone also sent a request to nonexistent `agent_2`.
- Episode 272 sent both agents toward the irrelevant `TARGET_any_targets|1`; only one required goal object was eventually visited.
- Episode 185 assigned both agents `Nothing to do`, so all four following `wait` actions were consistent with the incorrect assignment.
- The 03:15 run used the base GPT YAML, so Spot's prompt still exposed manipulation actions. It did not use the Qwen overlay.
- Robot reflection invented missing sensor values and camera poses, applied manipulation reachability to detection tasks, and rejected feasible assignments. Full numerical transcripts were then copied into action context, producing tens of thousands of tokens and distracting the action model.
- `send_request` is represented as an external `wait` while its message is queued. Without source logging, this is indistinguishable from a model-selected wait.

The primary failure is therefore task-semantics and agent-protocol adaptation, not a lack of Qwen tool-call support.

## Constraints and Non-goals

### Constraints

- All new behavior in this design is enabled only when `EMOS_LLM_BACKEND=qwen`.
- The original `multi_rearrange/llm_spot_drone_per.yaml` remains unchanged.
- The existing GPT prompts, parser, context propagation, tool-call count, and retry behavior remain unchanged.
- The Qwen overlay remains model-family based and must work with later Qwen model variants; no model-size-specific file is added.
- The numerical capability stage remains enabled.
- A Qwen retry may reject malformed or semantically invalid output, but host code must not synthesize a valid assignment.

### Non-goals

- Do not improve Qwen's benchmark score with a deterministic round-robin planner.
- Do not remove group discussion, robot reflection, or numerical reasoning from the experiment.
- Do not change the Habitat detection predicate or oracle navigation implementation.
- Do not modify unrelated scene graph code or user-owned worktree changes.

## Selected Approach

Use a Qwen-only compatibility adapter with four boundaries:

1. an explicit perception-task vocabulary in discussion prompts;
2. structural and semantic validation of leader assignments with bounded model retries;
3. clean execution context plus valid-recipient enforcement in the action agent; and
4. observable, bounded recovery from malformed action responses.

This is preferred over prompt-only adaptation because the observed 9B model repeatedly violates identifiers and goal coverage even while returning syntactically valid text. It is preferred over deterministic post-processing because post-processing would replace model reasoning and make the resulting benchmark incomparable with GPT.

## Goal Vocabulary

Add a small pure helper that extracts goal object identifiers from the PDDL text goal. For the observed task it returns:

```text
any_targets|1
any_targets|0
```

The extraction source is only the explicit goal-condition lines, not every identifier appearing in the scene description. This keeps `TARGET_any_targets|*` location markers out of a pure detection task unless such a marker is itself explicitly named in a goal condition.

The Qwen leader prompt receives:

- the exact valid agent IDs from `robot_resume`;
- the exact goal object IDs extracted from the task description;
- a statement that only explicit goal-condition objects are required;
- a statement that `TARGET_any_targets|*` is a rearrangement destination marker, not an implicit detection goal;
- a prohibition on inventing agents, sensors, camera poses, FOV values, altitudes, or object requirements; and
- a requirement that every goal object be assigned to at least one real agent.

The Qwen robot-reflection prompt receives these rules:

- judge only the assigned goal objects;
- do not apply manipulation workspace or arm reachability to a detection-only subtask;
- use only capability values supplied in the robot resume;
- do not assume missing VFOV, camera pitch, flight altitude, or extra robots;
- navigation can reposition the sensor, so a current nearest-navigable-point distance is not proof that detection is impossible; and
- `Nothing to do` is valid only when the leader literally assigned it.

GPT receives none of these added instructions.

## Leader Assignment Validation

After each Qwen leader response, parse and validate the assignment before robot reflection.

Validation checks:

1. every key is one of the real agent IDs;
2. every real agent appears exactly once after parsing;
3. every extracted goal object appears in at least one non-`Nothing to do` subtask;
4. no object identifier absent from the goal conditions is assigned as a detection target; and
5. `Nothing to do` assignments do not leave any goal object uncovered.

If validation fails, append a concise correction message to the same leader conversation. The message lists only validation violations, valid IDs, and required goal objects. It does not suggest which robot should receive which object.

Allow at most three total leader attempts for each assignment request: the initial response plus two corrections. If all attempts remain invalid, raise a distinct assignment-validation error and stop that episode with a clear diagnostic. Do not silently convert the invalid assignment to `Nothing to do`, and do not synthesize a fallback assignment.

The same validator is applied after reflection-driven reassignment. A structurally invalid reassignment never replaces the last valid assignment.

## Action Context and Planning

The current implementation copies the entire numerical reflection transcript into each action model. In Qwen mode, replace that raw transcript with a compact semantic execution context containing:

- robot type and real robot ID;
- original PDDL task description;
- final assigned subtask;
- scene description; and
- valid peer IDs.

The final subtask preserves the result of group discussion; only intermediate code, execution errors, and repeated feasibility messages are omitted. GPT continues receiving the existing full history.

The Qwen pre-action planning prompt must request a short ordered list of exact action names and arguments. It must prohibit yes/no feasibility markers. The generated plan remains advisory and is not parsed into host-selected actions.

The Qwen action system prompt adds:

- select exactly one exposed tool;
- `wait` is allowed only when the assigned subtask is literally `Nothing to do`, all assigned goal objects have already been successfully visited, or no immediate action is possible while waiting for a real peer;
- for an unfinished detection subtask, navigate to an unvisited assigned goal object; and
- `send_request.target_agent` must be one of the listed real peer IDs.

## Action Validation and Retry

The Qwen action response is validated before being returned to `CrabAgent`:

- exactly one tool call is present;
- the tool exists in the agent's filtered action map;
- arguments decode to a JSON object and validate against the action schema;
- `nav_to_obj.target_obj` is an object in the assigned subtask;
- `send_request.target_agent` is a real peer and is not the sender; and
- an explicit `wait` obeys the Qwen wait policy above.

On failure, append a concise tool-protocol correction and ask Qwen again. Allow at most three total attempts per action decision. Do not silently map a missing tool call, malformed JSON, unknown tool, invalid recipient, or premature wait to `wait`.

If all attempts fail, return a diagnostic action failure to the policy. The policy may safely execute the existing wait skill for environment stability, but the log must label it `protocol_fallback` and include the reason. This fallback is not counted as a model-selected wait.

Valid `send_request` remains an internal action and maps to external wait while the message is delivered, but the log labels the source as `send_request`. A literal valid wait is labeled `model_wait`.

## Qwen Overlay Guard

The supported Qwen perception command uses:

`multi_rearrange/llm_spot_drone_per_qwen.yaml`

At Qwen startup, log the Hydra config name and each agent's exposed action names. If the pure-perception task is started with manipulation skills exposed, fail fast with a message directing the caller to the Qwen overlay. This prevents another run that accidentally uses the GPT YAML while leaving GPT behavior untouched.

## Error Handling

- Invalid leader output: bounded correction, then an explicit assignment-validation failure.
- Invalid robot decision marker: existing Qwen parser recovery, then reflection continues according to its bounded policy.
- Invalid action response: bounded correction; exhaustion is a labeled protocol fallback rather than an indistinguishable model wait.
- Invalid peer: correction before message delivery; no queue entry is created for an unknown agent.
- Infrastructure error such as an unreachable OpenAI-compatible endpoint: propagate as an infrastructure failure.
- Habitat skill failure after a valid action: preserve existing skill termination and replanning behavior.

## Testing Strategy

### Unit tests

- Extract only explicit goal-condition objects from the PDDL text.
- Accept valid assignments that cover all goals with real agents.
- Reject missing agents, uncovered goals, nonexistent agents, and irrelevant `TARGET_*` detection assignments.
- Verify Qwen leader retries after invalid output and accepts a subsequent valid model-generated assignment.
- Verify retry exhaustion never synthesizes an assignment.
- Verify Qwen clean action context omits numerical code and retains task, final subtask, scene, and peers.
- Verify invalid `agent_2`, malformed arguments, missing tools, and premature wait trigger Qwen retries.
- Verify valid navigation, legitimate wait, and valid `send_request` retain distinct sources.
- Verify GPT prompts, history propagation, action parser, and call count are byte-for-byte or behaviorally unchanged.
- Verify the Qwen overlay exposes only `send_request`, `nav_to_obj`, and `wait` for this task.

### Runtime verification

Run Qwen first on the same three episode IDs observed in the failing log, using the Qwen overlay and the original episode step limit. Confirm:

- final assignments contain only real agent IDs and explicit goal objects;
- no action targets an irrelevant `TARGET_any_targets|*` marker;
- no request targets `agent_2`;
- non-wait navigation actions are executed for unfinished goals;
- protocol fallbacks, model waits, and internal requests are separately counted; and
- at least one episode completes with `pddl_success=1` before expanding the run.

Then run a multi-episode Qwen sample with the same dataset, sensor, environment, deterministic evaluation, and metric configuration as GPT. A reduced smoke-test step cap is not acceptable evidence for result comparability.

For GPT, resolve the original YAML and run regression tests proving no Qwen prompt fragments, validators, retries, or filtered tools are active. If valid GPT credentials or a stored GPT baseline are available, run the same episode sample and compare metrics. Without a valid GPT endpoint or stored results, configuration comparability can be proven, but empirical score comparability cannot be claimed.

## Invocation

Qwen:

```bash
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY
unset http_proxy https_proxy all_proxy
export OPENAI_BASE_URL="http://127.0.0.1:8000/v1"
export OPENAI_API_KEY="local-qwen"
export EMOS_LLM_BACKEND="qwen"
# Optional when the server requires a concrete served model ID.
export EMOS_LLM_MODEL="qwen-3.5-9b"
set -o pipefail

python -u -m habitat_baselines.run \
  --config-name=multi_rearrange/llm_spot_drone_per_qwen.yaml \
  habitat_baselines.evaluate=True \
  habitat_baselines.num_environments=1 \
  2>&1 | tee "logs/llm_spot_drone_per_qwen_$(date +%Y%m%d_%H%M%S).log"
```

GPT remains:

```bash
export OPENAI_BASE_URL="https://api.labforge.cc/v1"
export OPENAI_API_KEY="openai_key"
unset EMOS_LLM_BACKEND EMOS_LLM_MODEL
set -o pipefail

python -u -m habitat_baselines.run \
  --config-name=multi_rearrange/llm_spot_drone_per.yaml \
  habitat_baselines.evaluate=True \
  habitat_baselines.num_environments=1 \
  2>&1 | tee "logs/llm_spot_drone_per_$(date +%Y%m%d_%H%M%S).log"
```

## Completion Criteria

This work is complete only when all of the following have direct evidence:

1. the Qwen overlay is required and manipulation tools are absent;
2. failing-log semantics are covered by tests;
3. Qwen cannot silently turn invalid leader/action responses into ordinary waits;
4. Qwen runtime produces goal-directed navigation without nonexistent agents or irrelevant target markers;
5. at least one full-step-limit Qwen episode reaches `pddl_success=1`;
6. GPT regression tests prove its configuration and runtime branch remain unchanged; and
7. the final handoff distinguishes configuration comparability from any empirical GPT score comparison that credentials or baseline data do not permit.
