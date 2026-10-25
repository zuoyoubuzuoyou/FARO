# LLM code policy recovery for EMOS

The requested outcome is an online loop: detect a runtime system error, assemble a fixed structured prompt, generate revised Python recovery logic, and resume the same simulation. Later revisions receive the previous code and observed outcomes. This improves runtime error handling; it does not update model weights or establish automatic improvement in task success.

## Design

Use a versioned recovery extension rather than rewriting imported simulator modules while they execute. The LLM supplies a complete `recover(context, memory)` function. Its branches and persistent JSON memory decide when to resume/replan an agent or dispatch a configured skill. The host supplies the missing Resume interface, validates every command before applying any, and executes skills through their existing controller lifecycle. Simulator poses, held objects, agent models, conversation histories, rollout tensors and completed work persist.

The fixed context contains `schema_version`, `modification_instruction`, `error_report`, `task`, `planner_plan`, `recent_history`, `runtime`, `code`, `recovery`, and `available_api`. History means the last ten complete chat turns per agent. Missing information is explicit. Source inputs come only from a fixed repository allowlist covering simulation, evaluator, control, agent and model calls. Labels, final outcomes and injected fault markers are not recovery evidence. Source and context are redacted before persistence or transmission.

Generated functions return `{commands: [...], memory: {...}}`. Commands are `resume` (interrupt and select a new skill), `replan` (add explicit corrective instructions), or `action` (select an available skill with validated arguments). No generated function receives simulator objects, clients, filesystem handles or credentials. Python is AST-validated and runs in a separate process with restricted builtins, environment, time and resource budgets. This is a deliberately restricted Python API, not an arbitrary Python security sandbox.

The online adapter consumes detached trace events, reuses the incremental detector reader and runs recovery outside logger callbacks after a returned transition and before the next action. It supports EMOS's existing single-environment multi-agent policy. Recovery is opt-in and forces physical state capture. Terminal frames cannot be resumed after VectorEnv has already automatically reset; record their outcome without applying commands to the new episode.

Each generation attempt stores its prompt, raw response, code, validation/application result and version. Active code runs on subsequent sampled observations. Invalid candidates leave the previous version active. Later runtime failures roll back to the preceding validated version, or the original loop. Fixed cooldown and iteration budgets prevent unbounded generation. Outcome records distinguish a command being applied, observable progress and final task completion; neither syntax validity nor motion proves successful recovery.

## Acceptance

- Default evaluation behavior is unchanged when disabled.
- A real controller test detects a stalled fake environment, executes generated Python, changes the next action, and advances the same environment without reset.
- Fixed prompt fields contain actual plans, recent turns, source and prior revisions.
- Unknown agents/tools, invalid arguments, forbidden code, timeouts and malformed outputs cannot partially apply a command batch.
- Per-episode state is isolated; terminal frames never mutate the next episode.
- Tests use local transport replacements and CPU fixtures. Real model/simulator recovery effectiveness requires a subsequent experiment and is not inferred from those tests.
