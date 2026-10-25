# Code policy recovery implementation plan

Implement the user's specified recovery flow in the current clean `rsi` workspace. Preserve existing experiments and use no real model calls during implementation. The design is [the recovery specification](../specs/2026-10-23-code-policy-recovery-design.md).

1. Core (root): write failing tests for fixed context, generated Python execution, restricted operations, revision rollback, cooldown, progress feedback and budgets. Implement pure-stdlib `habitat_mas.recovery` configuration, prompt/generator, Python worker and controller. Run unittest tests with the Habitat Python environment.
2. Policy adapter (parallel): test and implement retained planner output, history/tool exports, whole-batch validation, forced action queue and explicit Resume/replan. Preserve normal skill lifecycle, agent history and simulator state. Fix the zero-action episode-start heuristic to use lifecycle state.
3. Online integration (parallel): test and implement a detached event callback, public incremental frame reader, per-episode online bridge, nested eval configuration and evaluator hook between returned transitions and next policy actions. Require state capture when enabled; never apply recovery to a terminal frame.
4. Integration and review (root plus fresh reviewer): run core, policy, online and existing detection/trajectory regression tests; inspect interface compatibility and fix findings. Add a deterministic recovery demo and usage documentation with accurate limits and measured test evidence.

Shared interfaces: `RecoveryConfig`; `RecoveryController(config, output_dir, detector=None, generator=None).observe(frame, policy, terminal=False)`; policy `get_recovery_context(history_turns=10)`, `validate_recovery_commands(commands)` and `apply_recovery_commands(commands)`; `OnlineRecovery.on_event(payload)` and `.advance(policy, dones)`.

Review focus: actual Resume before low-level act, gripper preservation via standard skills, invalid multi-command atomicity, code process limits, prompt redaction/no future labels, active-version rollback, no code/model call in trace callback, no cross-episode state, no automatic claim of successful recovery.
