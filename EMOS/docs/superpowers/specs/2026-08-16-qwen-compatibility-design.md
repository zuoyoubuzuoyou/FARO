# Qwen Compatibility Design for LLM Spot–Drone Perception

## Objective

Add a Qwen-specific compatibility path for the Spot–drone perception evaluation while preserving the current GPT configuration and runtime behavior.

The Qwen path must:

- keep numerical Python generation and execution enabled;
- prevent perception-only agents from executing manipulation skills;
- expose only executable tools to each robot;
- recover safely from malformed generated Python;
- tolerate Qwen's observed final-answer formatting variants; and
- never abort the Habitat evaluation because one numerical check fails.

The GPT path must remain the default and retain its current configuration, prompts, tool list, parser, and numerical execution behavior.

## Constraints and Non-goals

### Constraints

- The caller currently selects the serving backend through `OPENAI_BASE_URL` and `OPENAI_API_KEY`.
- The local Qwen server accepts the existing request model name `gpt-5.5`; changing the request model name is not required for this work.
- The compatibility profile must support the Qwen model family rather than one specific Qwen model. Switching between Qwen model sizes or releases must not require another YAML file or a code change.
- Numerical reasoning must remain model-generated Python, so replacing it with a deterministic host calculator is outside this design.
- Qwen compatibility must be explicitly enabled. It must not be inferred from a localhost URL or request model name.

### Non-goals

- Do not change the base `multi_rearrange/llm_spot_drone_per.yaml` configuration.
- Do not change GPT prompts, response parsing, action exposure, retry behavior, or interpreter output.
- Do not add missing place sensors to the perception benchmark. Manipulation is not part of its PDDL goal.
- Do not refactor unrelated multi-agent planning or Habitat skill code.

## Selected Approach

Use an explicit runtime profile plus one Hydra overlay:

- `EMOS_LLM_BACKEND` selects the compatibility profile. An unset value means `gpt`; `qwen` enables the compatibility path.
- `EMOS_LLM_MODEL` optionally selects the request model ID independently of the compatibility profile.
- `multi_rearrange/llm_spot_drone_per_qwen.yaml` inherits the existing perception configuration and removes manipulation skills only for Qwen runs.
- Existing Python files receive small conditional Qwen branches. Their GPT branches retain the current behavior.

This approach is preferred over URL detection because a URL does not reliably identify the served model. It is preferred over duplicating the model and policy classes because duplication would make later fixes diverge.

The overlay name uses the model-family label `qwen`, not a versioned name such as `qwen-3.5-9b`. All supported Qwen variants share the same compatibility behavior unless future evidence demonstrates a real protocol difference.

## Runtime Profile

Add one shared helper that reads and validates `EMOS_LLM_BACKEND`:

```python
def get_llm_backend() -> str:
    backend = os.getenv("EMOS_LLM_BACKEND", "gpt").strip().lower()
    if backend not in {"gpt", "qwen"}:
        raise ValueError(f"Unsupported EMOS_LLM_BACKEND: {backend}")
    return backend
```

The helper will be used by the model wrapper, multi-agent discussion code, and LLM high-level policy. The selected backend must be logged once near initialization.

The request model ID is resolved separately. Existing callers continue to provide the current `model="gpt-5.5"` default. `OpenAIModel` overrides that request field only when `EMOS_LLM_MODEL` contains a non-empty value:

```python
configured_model = os.getenv("EMOS_LLM_MODEL", "").strip()
self.model = configured_model or model
```

`EMOS_LLM_MODEL` never enables Qwen compatibility by itself. Conversely, `EMOS_LLM_BACKEND=qwen` enables Qwen compatibility even when the server requires the request model alias to remain `gpt-5.5`. This separation is required because an OpenAI-compatible server may expose arbitrary model aliases.

Behavior matrix:

| Behavior | Backend unset / `gpt` | Backend `qwen` |
| --- | --- | --- |
| Base Hydra config | Existing file | Qwen overlay |
| Request model ID | Existing `gpt-5.5` default | `EMOS_LLM_MODEL`, or existing alias if unset |
| Manipulation skills | Existing behavior | Removed for perception |
| `ACTION_POOL` | Existing full pool | Filtered by actual skills |
| Numerical prompt | Existing prompt | Qwen-specific contract |
| Code execution loop | Existing loop | Bounded validation and repair |
| Final response parser | Existing strict parser | Qwen-compatible parser |

## Qwen Hydra Overlay

Add one file:

`habitat-baselines/habitat_baselines/config/multi_rearrange/llm_spot_drone_per_qwen.yaml`

It inherits `llm_spot_drone_per` and replaces only Spot's `ignore_skills` list:

```yaml
# @package _global_

defaults:
  - llm_spot_drone_per
  - _self_

habitat_baselines:
  rl:
    policy:
      agent_0:
        hierarchical_policy:
          ignore_skills:
            - open_cab
            - open_fridge
            - close_cab
            - close_fridge
            - pick
            - pick_at_position
            - place
            - place_at_position
            - reset_arm
```

The original four ignored cabinet skills are repeated because Hydra replaces the list rather than appending to it.

## Qwen Tool Exposure

In Qwen mode only, construct the CrabAgent action list from the agent's actual `_skill_name_to_idx` mapping:

```python
executable_action_names = set(self._skill_name_to_idx)
llm_actions = [
    action
    for action in ACTION_POOL
    if action.name == "send_request"
    or action.name in executable_action_names
]
```

`send_request` is preserved explicitly because CrabAgent handles it internally and it is not a Habitat skill.

In GPT mode, pass the existing unfiltered `ACTION_POOL` unchanged.

The runtime validity check remains as defense in depth: an action absent from `_skill_name_to_idx` falls back to `wait`. In the Qwen path, validity should be checked before parsing action-specific arguments so a malformed forbidden action cannot raise a dictionary-key error.

## Numerical Execution Recovery

### Existing GPT behavior

The existing `SubprocessInterpreter.run()` return format and the existing planning-stage loop remain unchanged for GPT.

### Qwen-only detailed execution

Add a separate detailed interpreter method rather than changing `run()`:

```python
@dataclass
class ExecutionResult:
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool
```

The Qwen method will:

1. accept Python only;
2. compile the generated source before launching a subprocess;
3. execute it with a 10-second timeout;
4. capture return code, stdout, and stderr separately;
5. always delete its temporary file; and
6. return an `ExecutionResult` instead of embedding stderr in a free-form string.

### Repair policy

A numerical interaction may make at most three total code attempts: the initial attempt plus two repairs.

- Syntax failure: send only the syntax-error type, line number, and message back to Qwen.
- Runtime failure: send only the exception tail and request a correction of that error.
- Timeout: tell Qwen that the program exceeded 10 seconds and prohibit loops in the repair.
- Success: return stdout and request the final feasibility decision without another code block.
- Exhaustion: return a conservative structured negative response explaining that numerical verification failed. This lets the leader reassign the subtask while the Habitat evaluation continues.

The retry counter is reset for every independent robot reflection request.

## Qwen Numerical Prompt Contract

Append the following behavioral contract only in Qwen mode:

1. Output exactly one Python code block when numerical verification is needed.
2. Make the code self-contained and put all imports first.
3. Use only Python standard-library modules.
4. Do not place Markdown, `{{yes}}`, `{{no}}`, or natural-language conclusions inside Python.
5. Print only required values using `name: value` lines.
6. Keep the calculation minimal; do not redefine values or create unused helper functions.
7. After successful execution, output only `{{yes}}` or `{{no||reason}}` and do not emit another code block.
8. After failed execution, correct only the reported error.

Prompt changes are supplementary. Parser and runtime validation remain authoritative.

## Qwen Response Parsing

Keep the current parser untouched for GPT. Add a Qwen-only parser that accepts:

- `{{yes}}`, `{yes}`, `yes`, and case variants;
- `{{no||reason}}`, `{no||reason}`, and `no||reason`; and
- surrounding whitespace, quotes, or Markdown fences.

Parsing order:

1. Search for braced decision markers and use the last match.
2. If none exists, inspect only the final non-empty line for a bare decision.
3. If neither form is valid, preserve the existing safe fallback of `no` with the original text as its reason.

Restricting bare decisions to the final line prevents words such as "yes" in the reasoning body from being mistaken for the result.

## Error Boundaries

The following Qwen-originated failures must remain local to the current reflection or action:

- malformed Python;
- Python runtime exception;
- numerical timeout;
- malformed yes/no response;
- unavailable or forbidden action; and
- multiple tool calls when only one is permitted.

None of these conditions may propagate out of the LLM wrapper and terminate the vector environment. Infrastructure errors such as an unreachable LLM endpoint remain real errors and are not hidden by this compatibility layer.

## Invocation

Qwen:

```bash
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY
unset http_proxy https_proxy all_proxy
export OPENAI_BASE_URL="http://127.0.0.1:8000/v1"
export OPENAI_API_KEY="local-qwen"
export EMOS_LLM_BACKEND="qwen"
# Optional: set this only when the server requires a concrete model ID.
export EMOS_LLM_MODEL="qwen-3.5-9b"

python -u -m habitat_baselines.run \
  --config-name=multi_rearrange/llm_spot_drone_per_qwen.yaml \
  habitat_baselines.evaluate=True \
  habitat_baselines.num_environments=1
```

GPT:

```bash
export OPENAI_BASE_URL="https://api.labforge.cc/v1"
export OPENAI_API_KEY="openai_key"
unset EMOS_LLM_BACKEND
unset EMOS_LLM_MODEL

python -u -m habitat_baselines.run \
  --config-name=multi_rearrange/llm_spot_drone_per.yaml \
  habitat_baselines.evaluate=True \
  habitat_baselines.num_environments=1
```

Use `set -o pipefail` when either command is piped through `tee`.

## Testing Strategy

### Profile isolation

- Unset profile resolves to `gpt`.
- Explicit `qwen` resolves to `qwen`.
- Unknown values fail at initialization.
- Two different `EMOS_LLM_MODEL` values under the `qwen` profile select different request model IDs while exercising the same compatibility branches.
- A Qwen server that requires the alias `gpt-5.5` works with `EMOS_LLM_MODEL` unset.
- Setting `EMOS_LLM_MODEL` without setting `EMOS_LLM_BACKEND=qwen` changes only the request model field and does not activate Qwen compatibility.
- GPT snapshots confirm the existing prompt, parser, `ACTION_POOL`, and interpreter output are unchanged.

### Configuration

- Compose the original GPT config and confirm its Spot skill configuration is unchanged.
- Compose the Qwen overlay and confirm manipulation skills are absent.

### Tool filtering

- Qwen Spot and drone receive only tools backed by their skill mappings, plus `send_request`.
- GPT receives the current full `ACTION_POOL`.
- A forbidden Qwen action falls back to `wait` before argument parsing.

### Numerical execution

- Valid code succeeds on the first attempt.
- A syntax error is repaired successfully.
- A runtime error is repaired successfully.
- Three failed attempts produce a safe negative decision without raising.
- An infinite loop times out and does not leak a temporary file.

### Parsing

- All supported Qwen yes/no variants parse correctly.
- A reasoning paragraph containing "yes" but without a final decision falls back safely.
- GPT continues using the current strict parser.

### Integration

- A short Qwen perception evaluation never exposes or executes `pick`, `place`, or `reset_arm`.
- A short GPT evaluation composes and starts with the original configuration and behavior.

## Acceptance Criteria

- The original GPT command and YAML require no changes.
- With `EMOS_LLM_BACKEND` unset, all Qwen compatibility branches are inactive.
- Changing from one Qwen model to another requires only changing or unsetting `EMOS_LLM_MODEL`; it does not require a new config file or code change.
- The Qwen overlay and compatibility code contain no version-specific Qwen model name.
- Qwen numerical execution remains enabled.
- A Qwen-generated Python error cannot terminate the Habitat evaluation.
- Qwen cannot select manipulation tools in the perception configuration.
- Qwen formatting variants no longer produce avoidable `No match found` messages.
- The implementation adds exactly one production configuration file and does not duplicate benchmark, PDDL, skill, or policy definitions.
