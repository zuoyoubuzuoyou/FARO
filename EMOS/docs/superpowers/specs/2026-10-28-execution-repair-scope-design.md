# Execution repair scope

The user authorized automatic repairs to waiting/interruption, navigation completion, and logical control-state recovery, retaining the existing validate/apply/resume workflow and the prohibition on cheating. This extends the previously approved mechanism; it does not authorize a new simulation run or API request.

Use a versioned execution profile rather than removing validation. Simply adding class names would still fail the existing expression validator; allowing arbitrary action methods would also permit physics changes. The selected implementation uses narrow editable AST regions in existing execution methods and explicitly declared Resume control writes.

Editable regions:

- WaitSkillPolicy._is_skill_done: a read-only predicate over the existing elapsed steps, requested wait duration and batch indices; preserve the timing dependency and prohibit constant completion.
- SkillPolicy.should_terminate: interrupt combination only, preserving postconditions, timeout configuration, diagnostics and actual action execution.
- OracleNavPolicy._is_skill_done: gate the real completion sensor with existing control timing to reject stale completion; never synthesize positive sensor results.
- OracleNavAction.step, OracleNavCoordinateAction.step and OracleNavDiffBaseAction.step: completion predicates and completion bookkeeping only. Preserve geometry sources, target selection, existing thresholds and every motion/controller subtree. Allow filling in a missing skill_done=True only inside an existing at_goal branch. Cached-completion suppression may be disabled so real geometry is checked again.
- Resume: request high-level control and clear declared transient skill counters/previous-running flags through bounded iteration of the existing skills. Do not change skill arguments, selected targets, histories, observations or physical state.

Use one validator at prepare and install, a new validation profile, and allowlisted navigation worker RPC transactions. Preserve source hashes, exact signatures, current-instance method installation, reverse rollback, checkpoints, inference mode, and existing model/API behavior. Environment, task, simulator, sensors, metrics, physics, movement generation and capability definitions remain protected.

These are supported edit regions, not permission to rewrite every statement in those methods or a proof of arbitrary Python safety. Passing validation and Resume does not establish recovered task progress. Existing native hangs, crashes or completed/reset episodes remain outside the recovery boundary.

Verification must exercise real source methods and tensors, permitted predicate repairs, blocked constant success/threshold inflation/physical writes, worker installation on the original instance, and coordinated rollback. No production task fix is hardcoded as part of opening permissions.
