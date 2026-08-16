# Qwen Single-Pipe Assignment Compatibility Design

## Objective

Prevent Qwen perception evaluation from aborting when a Qwen model emits a
braced leader assignment with one separator pipe, such as
`{agent_0|Detect any_targets|0}`, while preserving the existing semantic
validation, retry policy, and GPT behavior.

This is a compatibility extension of the approved Qwen perception adapter. It
does not change task allocation policy or synthesize an assignment.

## Evidence and Root Cause

The authoritative failure is
`logs/llm_spot_drone_per_qwen_smoke_20260816_163641.log`, episode 219. The local
Qwen endpoint returned HTTP 200, but all five leader responses used the
single-pipe form `{agent_id|subtask}`. The Qwen assignment parser recognizes
only `{agent_id||subtask}` and braceless double-pipe lines, so validation
incorrectly reported both real agents as missing and eventually raised
`QwenAssignmentValidationError`.

The responses also contained the invalid manipulation verb `rearrange`.
Existing log evidence shows that the model usually removes manipulation
language after receiving the corresponding semantic violation. Because the
single-pipe responses never reached semantic parsing, every correction named
the wrong structural problem instead.

The final `BrokenPipeError` is a cleanup consequence of the primary exception.
The ReplicaCAD `v3_sc4` navmesh diagnostics are unrelated and non-fatal for the
current dataset.

## Constraints

- Change only the Qwen assignment compatibility parser and its tests.
- Accept both braced `{agent_id||subtask}` and `{agent_id|subtask}` forms.
- Keep the single-pipe compatibility unambiguous by requiring outer braces and
  using only the first separator between the agent ID and subtask.
- Preserve pipes inside object IDs such as `any_targets|0`.
- Continue applying all existing agent-ID, goal-coverage, duplication,
  irrelevant-object, and manipulation checks after parsing.
- Continue to raise `QwenAssignmentValidationError` after five genuinely
  invalid model attempts. Do not create a deterministic fallback assignment.
- Do not modify the GPT parser, GPT prompts, GPT call count, or the original GPT
  YAML.
- Preserve the unrelated user change in
  `habitat-mas/habitat_mas/scene_graph/utils.py`.

## Selected Approach

Add a second braced assignment expression for exactly one separator pipe. The
expression rejects `||` at the separator position so existing double-pipe
responses are not parsed twice. `_assignment_entries()` will collect the
existing double-pipe entries first, remove them from the remaining text, then
collect single-pipe entries before applying the existing braceless double-pipe
line parser.

This is preferred over changing only the prompt because episode 219 ignored the
existing `||` instruction five consecutive times. It is preferred over a host
fallback assignment because a fallback would replace model reasoning and break
the approved comparison constraints.

## Data and Error Flow

1. Qwen returns `{agent_0|...}` and `{agent_1|...}`.
2. `parse_qwen_assignment()` extracts both real agent IDs and their full
   subtasks, including object IDs containing pipes.
3. `validate_assignment()` reports the real semantic issue, such as forbidden
   `rearrange`, instead of reporting missing agents.
4. `request_qwen_assignment()` sends the existing concise correction to Qwen.
5. A corrected pure-detection assignment proceeds normally. A response that
   remains semantically invalid for five attempts still fails explicitly.

GPT continues through `parse_leader_response()` and never calls the Qwen parser.

## Testing

Add focused tests using literal episode-219 response shapes:

- a valid two-agent single-pipe detection response parses to the expected task
  dictionary and passes validation;
- a single-pipe response containing `rearrange` reports manipulation violations
  without reporting missing agents;
- double-pipe and braceless formats continue to pass their existing tests;
- existing exhaustion behavior still raises instead of synthesizing a task.

Run the focused semantics test first in RED and GREEN phases, then run all Qwen
compatibility tests and the full local regression suite. Finally run a
one-episode Qwen smoke evaluation for integration health and exercise the exact
episode-219 assignment text through the regression test.

## Completion Criteria

The fix is complete when:

1. the literal single-pipe assignment from episode 219 is parsed as two real
   agent entries;
2. its semantic manipulation error is observable and missing-agent errors are
   absent;
3. all Qwen and GPT regression tests pass;
4. a Qwen one-episode smoke run finishes without a configuration, API, parser,
   or protocol exception; and
5. the diff contains no change to the GPT YAML or the user's scene-graph edit.
