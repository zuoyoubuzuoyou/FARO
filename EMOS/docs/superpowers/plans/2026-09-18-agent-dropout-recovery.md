# Minimal agent dropout recovery

**Goal:** dropout → detect → identify unfinished goals → reassign → replan → execute.

**Design:** Add an opt-in `habitat_baselines.error_recovery.enabled` path to the
existing single-environment evaluator. Detection still uses membership telemetry.
A recovery coordinator reads current PDDL predicate truth values and existing
subtask descriptions. Keep surviving owners; transfer orphaned goals to the
least-loaded online agent (agent name breaks ties). Maintaining currently satisfied
goals also counts toward load. Exact entity tokens associate
the original free-text assignments with goals; ambiguous/missing ownership is
explicitly recorded and treated as unassigned. This is a baseline, not a capability
or reachability optimizer. Replan all survivors from current physical state and
remaining goals; never reset the simulator. Currently satisfied goals are retained
as execution constraints, since moving can invalidate perception/proximity goals.

Only conjunctions of atomic goals are supported. Agent-specific goals retain their
agent constraint. Unfinished goals bound to failed agents, objects held by failed
agents, and loss of the entire team produce explicit recovery failure and retain
the existing crash termination. Hang and message-loop behavior stays unchanged.
Successful recovery acknowledges only the detected dropout; the watchdog keeps
checking surviving agents. Episode reset clears recovery state.

Implementation and verification run in the existing `error-recovery` branch using
`/home/dyc/.conda/envs/habitat/bin/python`, without new dependencies.

- [x] Add behavioral tests for pending-goal extraction, assignment, watchdog
  acknowledgement, live dispatch, and continuation after dropout.
- [x] Implement dependency-light recovery coordinator and fresh simulator context.
- [x] Wire recovery into policy dispatch, agent replanning, evaluator configuration,
  event JSONL and trajectory output.
- [x] Run focused unittest regressions and import/config smoke checks; document
  launch command and distinguish verified continuation from simulator success.

Acceptance: a completed goal is never transferred; a failed agent never executes;
the survivor receives the pending goal and replans in the same simulator step;
subsequent steps continue without repeated recovery or spurious hang; repeated
episode IDs reset state; disabled recovery preserves detection-only termination.

## Validation

- 118 unittest cases passed in the habitat Python 3.9 environment, including 24
  recovery cases plus detection, dropout, hang, replay, launcher, trajectory and
  target-validation regressions. Modified Python files compile; Hydra composes
  the real Spot/Drone config with recovery enabled.
- Native Habitat, episode 295, fixed test model: inject agent_1 dropout at step 2;
  transfer `is_detected(any_targets|1)` from agent_1 to agent_0; replan agent_0 and
  select a new navigation skill at step 2; continue through step 29. No subsequent
  decisions from agent_1 and no watchdog crash. Exit code 0; final pddl_success=0
  at the deliberately limited 30-step horizon. This verifies the loop, not model
  quality or successful episode completion.
- The real-model attempt reached goal extraction and task reassignment but stopped
  before planning because the process had no OPENAI_API_KEY. No real-model recovery
  success rate has been measured.
- Read-only independent review found ambiguous-owner and transient-goal issues;
  both were corrected and verified with the reviewer's counterexamples.
