# Object Localization Error MVP

This MVP injects a one-shot perception fault into one EMOS LLM agent. It
changes only that agent's symbolic `scene_description` immediately before a
selected high-level decision. Habitat simulator state and every other agent's
context remain unchanged.

`decision_step` is zero-based and counts only occasions on which the selected
agent requests a new high-level skill. It is intentionally different from the
low-level Habitat simulator step. The emitted event records both counters.

## Export clean and faulty trajectories

Run the same episode once without injection and once with injection. Use
different `run_label` values so neither JSONL file overwrites the other.

Clean run:

```bash
python -u -m habitat_baselines.run \
  --config-name=multi_rearrange/llm_spot_drone_per.yaml \
  habitat_baselines.evaluate=True \
  habitat_baselines.num_environments=1 \
  habitat_baselines.test_episode_count=1 \
  habitat.environment.iterator_options.shuffle=False \
  habitat_baselines.fault_injection.object_localization.enabled=False \
  habitat_baselines.fault_injection.object_localization.record_trajectory=True \
  habitat_baselines.fault_injection.object_localization.output_dir=fault_output \
  habitat_baselines.fault_injection.object_localization.run_label=clean
```

Faulty run (replace the episode, object, agent, decision, and false location
with values valid for the selected dataset):

```bash
python -u -m habitat_baselines.run \
  --config-name=multi_rearrange/llm_spot_drone_per.yaml \
  habitat_baselines.evaluate=True \
  habitat_baselines.num_environments=1 \
  habitat_baselines.test_episode_count=1 \
  habitat.environment.iterator_options.shuffle=False \
  habitat_baselines.fault_injection.object_localization.enabled=True \
  habitat_baselines.fault_injection.object_localization.record_trajectory=True \
  habitat_baselines.fault_injection.object_localization.episode_id=15 \
  habitat_baselines.fault_injection.object_localization.agent=agent_1 \
  habitat_baselines.fault_injection.object_localization.decision_step=0 \
  habitat_baselines.fault_injection.object_localization.object_name=051_large_clamp_:0000 \
  habitat_baselines.fault_injection.object_localization.replacement_height=0.2 \
  habitat_baselines.fault_injection.object_localization.replacement_horizontal_distance=4.0 \
  habitat_baselines.fault_injection.object_localization.output_dir=fault_output \
  habitat_baselines.fault_injection.object_localization.run_label=faulty
```

The two raw trajectories are written to:

```text
fault_output/clean_trajectory.jsonl
fault_output/faulty_trajectory.jsonl
```

Each file contains ordered events:

- `trajectory_start`: run mode and label;
- `episode_start`: episode, scene, task text, robot resumes, and clean scene
  description;
- `simulator_step`: the joint low-level action, robot/object ground-truth
  state before the action, current high-level skills, any new LLM decisions,
  reward, and Habitat metrics;
- `episode_end`: final reward and metrics.

On the selected faulty decision, `policy_info` additionally contains both
`agent_N_decision_event` and `agent_N_fault_event`. The latter stores the
correct fact under `fact_before` and injected fact under `fact_after`.

Simulator step 0 is the first call to `envs.step`; its
`world_state_before` is the initial state before that action. The terminal
step's final task metrics are recorded even though Habitat's vectorized
environment may already have reset its physical simulator for the next
episode.

## MP3D: replace semantic region/floor

```bash
python -u -m habitat_baselines.run \
  --config-name=multi_rearrange/llm_multi_agent_mobility.yaml \
  habitat_baselines.evaluate=True \
  habitat_baselines.num_environments=1 \
  habitat_baselines.test_episode_count=1 \
  habitat_baselines.fault_injection.object_localization.enabled=True \
  habitat_baselines.fault_injection.object_localization.episode_id=272 \
  habitat_baselines.fault_injection.object_localization.agent=agent_1 \
  habitat_baselines.fault_injection.object_localization.decision_step=2 \
  habitat_baselines.fault_injection.object_localization.object_name=cup_0 \
  habitat_baselines.fault_injection.object_localization.replacement_region=bedroom_3 \
  habitat_baselines.fault_injection.object_localization.replacement_floor=1 \
  habitat_baselines.fault_injection.object_localization.run_label=PF_wrong_location_001
```

The injector recognizes facts in this form:

```text
The object "cup_0" is located in kitchen_1 on 0 floor.
```

## HSSD: replace exposed geometric localization

HSSD's current scene text does not contain semantic regions. It exposes object
height and horizontal distance to a navigable point, which can be corrupted as
follows:

```bash
python -u -m habitat_baselines.run \
  --config-name=multi_rearrange/llm_spot_drone_per.yaml \
  habitat_baselines.evaluate=True \
  habitat_baselines.num_environments=1 \
  habitat_baselines.test_episode_count=1 \
  habitat.environment.iterator_options.shuffle=False \
  habitat_baselines.fault_injection.object_localization.enabled=True \
  habitat_baselines.fault_injection.object_localization.agent=agent_0 \
  habitat_baselines.fault_injection.object_localization.decision_step=0 \
  habitat_baselines.fault_injection.object_localization.episode_id=15 \
  habitat_baselines.fault_injection.object_localization.object_name=032_knife_:0000 \
  habitat_baselines.fault_injection.object_localization.replacement_height=1.9 \
  habitat_baselines.fault_injection.object_localization.replacement_horizontal_distance=3.4 \
  habitat_baselines.fault_injection.object_localization.run_label=PF_wrong_geometry_001
```

Every successful injection appends one audit event to:

```text
fault_output/<run_label>_fault_events.jsonl
```

The event contains the episode, target agent, decision and simulator steps,
the original facts, injected facts, and `ground_truth_preserved: true`. If the
configured object or location field is absent, evaluation stops with a
`FaultInjectionError`; this prevents a run from being mislabeled as faulty
when no mutation occurred.
