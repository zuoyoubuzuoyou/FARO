# EMOS Fault Injection MVP

This implementation covers all eight first-stage examples from `研究内容.md`.
It preserves the Habitat simulator and scene graph as ground truth and injects
faults only into the agent-visible observation, assignment, action, or
verification layer.

| ID | Subtype | Injection stage | MVP effect |
| --- | --- | --- | --- |
| F1 | `WrongObjectLocation` | observation | Replaces/adds a false object location for one agent. |
| F2 | `ObjectRecognitionError` | observation | Renames one perceived object. |
| F3 | `WrongSubgoal` | assignment | Replaces one agent's assigned subgoal. |
| F4 | `MissingSubgoal` | assignment | Removes the configured subgoal or assigns `Nothing to do`. |
| F5 | `MessageDelayOrStale` | communication context | Adds an explicitly delayed/stale teammate message. |
| F6 | `DuplicateAssignment` | assignment | Copies `source_agent`'s assignment to another agent. |
| F7 | `ActionNoOpOrFalseSuccess` | action tensor | Executes Habitat `wait` instead and optionally reports success to the next planner call. |
| F8 | `MissingOrFalseVerification` | stop/verification control | Centrally forces all stop actions for a false positive, or suppresses them for false-negative/missing verification. |

## Run one fault

First validate all eight injection paths without starting Habitat or contacting
an external LLM:

```bash
python -m habitat_mas.fault_injection.smoke \
  --output-dir fault_injection_smoke_output
```

The command exits nonzero unless all F1–F8 events are injected and logged.

To test the actual Habitat dataset, simulator, scene-description sensor,
MultiLLMPolicy, and stop-action pipeline without contacting an external LLM,
run the deterministic mock integration with F8:

```bash
python -u -m habitat_mas.fault_injection.emos_mock_run \
  --config-name=multi_rearrange/llm_height_per.yaml \
  habitat_baselines.evaluate=True \
  habitat_baselines.num_environments=1 \
  habitat_baselines.test_episode_count=1 \
  'habitat_baselines.eval.video_option=[]' \
  habitat.dataset.fault_injection.enabled=True \
  habitat.dataset.fault_injection.schedule_path=habitat-mas/config/fault_injection/f8_false_verification_episode_21.json \
  habitat.dataset.fault_injection.log_dir=fault_injection_mock_integration_output
```

This runner is only an integration check. Use the regular
`habitat_baselines.run` command below for experiments with the configured LLM.

The deterministic HSSD example starts at episode `21`. Run one schedule at a
time so different faults do not mask one another:

```bash
env -u ALL_PROXY -u all_proxy python -u -m habitat_baselines.run \
  --config-name=multi_rearrange/llm_height_per.yaml \
  habitat_baselines.evaluate=True \
  habitat_baselines.num_environments=1 \
  habitat_baselines.test_episode_count=1 \
  'habitat_baselines.eval.video_option=[]' \
  habitat.dataset.fault_injection.enabled=True \
  habitat.dataset.fault_injection.schedule_path=habitat-mas/config/fault_injection/object_localization_episode_21.json \
  habitat.dataset.fault_injection.log_dir=fault_injection_output
```

Replace `object_localization_episode_21.json` with one of:

```text
f2_object_recognition_episode_21.json
f3_wrong_subgoal_episode_21.json
f4_missing_subgoal_episode_21.json
f5_stale_message_episode_21.json
f6_duplicate_assignment_episode_21.json
f7_action_noop_episode_21.json
f8_false_verification_episode_21.json
```

On injection the console prints, for example:

```text
[FaultInjection] episode=21 step=0 observer=agent_1 faults=F1_wrong_location_001
```

Each event is atomically written to
`fault_injection_output/<episode_id>/<fault_id>.json`. The record contains:

```json
{
  "fault_id": "F1_wrong_location_001",
  "fault_type": "PerceptionFault",
  "fault_subtype": "WrongObjectLocation",
  "episode_id": "21",
  "injected_at_step": 0,
  "faulty_agent": "agent_1",
  "affected_object": "002_master_chef_can_:0000",
  "affected_subtask": "detect_002_master_chef_can",
  "severity": "medium",
  "ground_truth_state": "...",
  "agent_observed_state": "...",
  "expected_recovery": "refresh_observation_or_ask_teammate",
  "status": "injected"
}
```

## Timing and strictness

Observation, communication-context, and assignment faults must use
`injected_at_step: 0`, matching EMOS's episode-initialization planning stage.
F7/F8 control faults can target later evaluator steps. `episode_id` is optional;
`episode_ids` accepts a list.

Object and assignment matching is strict by default. A missing scheduled target
stops the run with a descriptive error instead of silently producing a clean
episode. Set `habitat.dataset.fault_injection.strict=False` only when skipped
events are intentional.

## Troubleshooting

If OpenAI client creation reports that `socksio` is missing, reinstall the
updated Habitat-MAS requirements or run the local API endpoint without inherited
SOCKS proxy variables, as shown above. The dependency is declared in
`habitat-mas/requirements.txt`.

The released `height_per.json` uses the historical robot name
`SpotRobot_head_arm`, while its resume file is named `SpotRobot_head_jaw.json`.
The sensor resolves this alias explicitly and raises a descriptive error for any
other missing resume instead of silently dropping an active agent.

## Compare complete baseline and fault trajectories

The console log contains high-level LLM actions but not all per-step robot
poses. Set `trajectory_log_dir` to record one entry for every Habitat step. A
record contains both agents' world position and heading, the action sent to the
environment, policy/skill information, reward, metrics, and the done flag.

Run episode 21 without fault injection:

```bash
env -u ALL_PROXY -u all_proxy python -u -m habitat_baselines.run \
  --config-name=multi_rearrange/llm_height_per.yaml \
  habitat_baselines.evaluate=True \
  habitat_baselines.num_environments=1 \
  habitat_baselines.test_episode_count=1 \
  'habitat_baselines.eval.video_option=[]' \
  habitat.dataset.fault_injection.enabled=False \
  habitat.dataset.fault_injection.trajectory_log_dir=trajectory_output/baseline
```

Run the same episode with F1 enabled:

```bash
env -u ALL_PROXY -u all_proxy python -u -m habitat_baselines.run \
  --config-name=multi_rearrange/llm_height_per.yaml \
  habitat_baselines.evaluate=True \
  habitat_baselines.num_environments=1 \
  habitat_baselines.test_episode_count=1 \
  'habitat_baselines.eval.video_option=[]' \
  habitat.dataset.fault_injection.enabled=True \
  habitat.dataset.fault_injection.schedule_path=habitat-mas/config/fault_injection/object_localization_episode_21.json \
  habitat.dataset.fault_injection.log_dir=fault_injection_output \
  habitat.dataset.fault_injection.trajectory_log_dir=trajectory_output/fault
```

The two runs produce `trajectory_output/baseline/episode_21.json` and
`trajectory_output/fault/episode_21.json`. Compare and align every step with:

```bash
python -m habitat_mas.fault_injection.compare_trajectories \
  trajectory_output/baseline/episode_21.json \
  trajectory_output/fault/episode_21.json \
  --output-dir trajectory_output/comparison
```

The comparison writes a complete per-step `trajectory_comparison.csv`, a
machine-readable `trajectory_comparison.json`, and prints per-agent path length,
first divergent step, mean/max position divergence, and final divergence.

For a causal comparison, keep the episode, simulator seed, model, prompts, and
decoding settings identical. An external LLM can still produce different
answers across repeated calls, so LLM nondeterminism must be reported separately
from the injected fault's effect.
