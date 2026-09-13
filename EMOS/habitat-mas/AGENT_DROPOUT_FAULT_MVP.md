# Agent Dropout Structure Fault MVP

推荐使用[统一错误注入命令行](FAULT_INJECTION_CLI.md)，通过 `--fault`、`--episode` / `--random-episodes` 和 `--step` / `--trajectory` 指定注入。

This MVP injects a permanent, observable, logical agent dropout using either a
zero-based Habitat `simulator_step` or a zero-based, per-agent
`high_level_decision` step.

After the trigger, the selected agent:

- no longer runs its execution-stage LLM or low-level policy;
- leaves the available team roster; peers see the membership update at their
  next execution-stage LLM decision;
- discards its unread inbox and rejects new requests with `agent_unavailable`;
- keeps its policy state frozen;
- sends an all-zero no-op action to Habitat.

Failed `send_request` calls feed the transport error back to the sending LLM
within the same decision, allowing it to choose another action or teammate.
Correction attempts are bounded to two retries, followed by a wait fallback.
All fault transitions happen before any individual policy runs in that step,
including when the dropped recipient appears later in the dispatch order.
The all-wait shortcut considers only remaining members. Episode reset restores
membership and clears both messages and departure state.

Unlike Dropout, Hang retains membership and accepts requests into an unread
queue without notifying peers of a departure or returning an unavailable error.
Both faults keep the physical robot in the simulator and suppress its actions.
Dropout events now have `silent=False` and record member removal and rejected
messages in `effects`; Hang remains `silent=True`.

The physical robot entity remains in the simulator so the fixed multi-agent
action and observation spaces do not change. Other agents and simulator
physics continue normally. The initial group discussion occurs before
`simulator_step=0`, so a step-0 dropout means that assignment completes but
the selected agent never starts executing its assigned task.

## Clean run

```bash
env -u ALL_PROXY -u all_proxy python -u -m habitat_baselines.run \
  --config-name=multi_rearrange/llm_spot_drone_per.yaml \
  habitat_baselines.evaluate=True \
  habitat_baselines.num_environments=1 \
  habitat_baselines.test_episode_count=1 \
  habitat.environment.iterator_options.shuffle=False \
  habitat_baselines.fault_injection.structure.agent_dropout.enabled=False \
  habitat_baselines.fault_injection.structure.agent_dropout.record_trajectory=True \
  habitat_baselines.fault_injection.structure.agent_dropout.output_dir=fault_output \
  habitat_baselines.fault_injection.structure.agent_dropout.run_label=clean
```

## Faulty run: simulator-step trigger

This example drops `agent_1` immediately before Habitat executes simulator
step 40 of episode 15:

```bash
env -u ALL_PROXY -u all_proxy python -u -m habitat_baselines.run \
  --config-name=multi_rearrange/llm_spot_drone_per.yaml \
  habitat_baselines.evaluate=True \
  habitat_baselines.num_environments=1 \
  habitat_baselines.test_episode_count=1 \
  habitat.environment.iterator_options.shuffle=False \
  habitat_baselines.fault_injection.structure.agent_dropout.enabled=True \
  habitat_baselines.fault_injection.structure.agent_dropout.record_trajectory=True \
  habitat_baselines.fault_injection.structure.agent_dropout.episode_id=15 \
  habitat_baselines.fault_injection.structure.agent_dropout.agent=agent_1 \
  habitat_baselines.fault_injection.structure.agent_dropout.trigger_domain=simulator_step \
  habitat_baselines.fault_injection.structure.agent_dropout.simulator_step=40 \
  habitat_baselines.fault_injection.structure.agent_dropout.output_dir=fault_output \
  habitat_baselines.fault_injection.structure.agent_dropout.run_label=dropout_sim_step
```

This can interrupt an already-running Navigate/Pick/Place skill. The event also
records how many high-level decisions had completed at simulator step 40.
Its readable trace uses `Agent_N_Execution_Override`, because no new
high-level action was requested at that physical step:

```text
Simulator_Step: 40
Fault: StructureFault/AgentDropout injected at PRE_ACTION simulator step 40 for agent_1; 1 high-level decisions had completed.
Agent_1_Execution_Override: AgentDropoutNoOp[]
Agent_1_Observation: The running low-level skill was interrupted; policy and LLM were not called, and a zero no-op action was sent.
```

### Different simulator steps for a batch of episodes

Use `simulator_step_schedule` when each episode needs its own injection step.
Its format is `episode_id@step;episode_id@step`. It replaces the single
`episode_id` and `simulator_step` settings, and only listed episodes are
injected. Evaluation fails if any scheduled injection is not reached.

The following schedule covers the reproducible ten-episode run selected by
`habitat.seed=20260901` and `shuffle=True`. Its injection steps were drawn with
random seed `20260902` from steps 1--40; episode 96 has only simulator step 0,
so its injection point is necessarily 0.

```bash
env \
  -u ALL_PROXY -u all_proxy \
  -u HTTP_PROXY -u http_proxy \
  -u HTTPS_PROXY -u https_proxy \
  python -u -m habitat_baselines.run \
  --config-name=multi_rearrange/llm_spot_drone_per.yaml \
  habitat_baselines.evaluate=True \
  habitat_baselines.num_environments=1 \
  habitat_baselines.test_episode_count=10 \
  habitat.seed=20260901 \
  habitat.environment.iterator_options.shuffle=True \
  habitat_baselines.eval.video_option=[] \
  habitat_baselines.fault_injection.structure.agent_dropout.enabled=True \
  habitat_baselines.fault_injection.structure.agent_dropout.record_trajectory=True \
  habitat_baselines.fault_injection.structure.agent_dropout.agent=agent_1 \
  habitat_baselines.fault_injection.structure.agent_dropout.trigger_domain=simulator_step \
  'habitat_baselines.fault_injection.structure.agent_dropout.simulator_step_schedule="233@18;27@40;49@24;96@0;77@2;279@37;28@6;354@35;126@23;178@2"' \
  habitat_baselines.fault_injection.structure.agent_dropout.output_dir=fault_output \
  habitat_baselines.fault_injection.structure.agent_dropout.run_label=dropout_sim_random10
```

## Faulty run: high-level-decision trigger

This example drops `agent_1` immediately before its zero-based decision step 2
(its third high-level action) of episode 15. The actual simulator step at which
that decision boundary is reached is recorded in the event and text trace.
This trace contains a replacement `Agent_N_Action`, because the fault fires
exactly when a new high-level action would otherwise have been generated:

```text
Simulator_Step: 87
Fault: StructureFault/AgentDropout injected at PRE_HIGH_LEVEL_DECISION for agent_1 decision step 2 (simulator step 87).
Agent_1_Decision_Step: 2
Agent_1_Action: AgentDropoutNoOp[]
Agent_1_Observation: Policy and LLM were not called; a zero no-op action was sent.
```

```bash
env -u ALL_PROXY -u all_proxy python -u -m habitat_baselines.run \
  --config-name=multi_rearrange/llm_spot_drone_per.yaml \
  habitat_baselines.evaluate=True \
  habitat_baselines.num_environments=1 \
  habitat_baselines.test_episode_count=1 \
  habitat.environment.iterator_options.shuffle=False \
  habitat_baselines.fault_injection.structure.agent_dropout.enabled=True \
  habitat_baselines.fault_injection.structure.agent_dropout.record_trajectory=True \
  habitat_baselines.fault_injection.structure.agent_dropout.episode_id=15 \
  habitat_baselines.fault_injection.structure.agent_dropout.agent=agent_1 \
  habitat_baselines.fault_injection.structure.agent_dropout.trigger_domain=high_level_decision \
  habitat_baselines.fault_injection.structure.agent_dropout.decision_step=2 \
  habitat_baselines.fault_injection.structure.agent_dropout.output_dir=fault_output \
  habitat_baselines.fault_injection.structure.agent_dropout.run_label=dropout_decision_step
```

The outputs are:

```text
fault_output/clean_trajectory.jsonl
fault_output/dropout_sim_step_trajectory.jsonl
fault_output/dropout_sim_step_events.jsonl
fault_output/dropout_decision_step_trajectory.jsonl
fault_output/dropout_decision_step_events.jsonl
fault_output/clean_trace-episode_15.txt
fault_output/dropout_sim_step_trace-episode_15.txt
fault_output/dropout_decision_step_trace-episode_15.txt
```

The `*_trace-episode_*.txt` files are readable per-simulator-step trajectories.
Every step lists every agent's effective high-level action. A newly selected
action has status `new_high_level_decision`; on subsequent physical steps it
is repeated with status `continuing`. A dropped agent is shown as
`AgentDropoutNoOp[]` on every remaining step. The files follow the
benchmark-style `Task`, `Agent_N_Action`, and `Agent_N_Observation` layout and
include `Simulator_Step` so the fault location is unambiguous. EMOS' execution
policy returns tool calls rather than an explicit natural-language chain of
thought, so the trace records real actions and feedback and does not fabricate
a `Thought` field.

The JSONL files are the precise source for programmatic analysis. Each
`simulator_step` event contains the raw `joint_env_action` and a resolved
`agent_step_actions` mapping with the effective action and status of every
agent. Each step also contains a `simulator_state` snapshot captured after
that step's environment action and before any automatic episode reset. The
snapshot records every agent's position, rotation, held object, current
high-level action, and dropout state, plus each rigid object's unique simulator
ID, handle, goal alias, and position. The readable trace prints the same fields
under `Agent_N_Position`, `Agent_N_Rotation`, `Agent_N_Held_Object`,
`Agent_N_Current_High_Level_Action`, `Agent_N_Dropout_State`, and
`Object_Position`.

The simulator-step trajectory contains, starting at step 40:

```json
{
  "agent_1_dropout_active": true,
  "agent_1_dropout_since_simulator_step": 40,
  "agent_1_dropout_event": {
    "fault_type": "StructureFault",
    "fault_subtype": "AgentDropout",
    "trigger_domain": "simulator_step",
    "phase": "PRE_ACTION",
    "simulator_step": 40,
    "logical_agent_active": false,
    "simulator_agent_removed": false
  }
}
```

The decision-step event instead contains fields such as:

```json
{
  "trigger_domain": "high_level_decision",
  "phase": "PRE_HIGH_LEVEL_DECISION",
  "agent": "agent_1",
  "decision_step": 2,
  "simulator_step": 87
}
```

The one-shot event appears only at the trigger step; `dropout_active` remains
true on every later step. If the configured episode, agent, or step is never
reached, evaluation raises `FaultInjectionError` rather than silently
producing a mislabeled faulty run.

Membership removal is injected directly, not inferred by a failure detector.
Peers can adapt to membership and transport feedback through their existing LLM
policy. There is no automatic task reassignment, recovery within an episode,
temporary outage, or physical removal of the robot.
