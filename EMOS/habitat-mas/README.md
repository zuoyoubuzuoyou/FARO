# Habitat-MAS Package

Habitat-MAS is a Python package for Multi-Agent Systems in Habitat virtual environments.

## Table of Contents
- [Habitat-MAS Package](#habitat-mas-package)
  - [Table of Contents](#table-of-contents)
  - [Installation](#installation)
    - [Prerequisites](#prerequisites)
    - [Install habitat-mas](#install-habitat-mas)
  - [Usage](#usage)
    - [Download Data](#download-data)
      - [Download Base Dataset](#download-base-dataset)
      - [Download MP3D Dataset](#download-mp3d-dataset)
      - [Download Habitat-MAS Dataset](#download-habitat-mas-dataset)
    - [Run the demo](#run-the-demo)
    - [Run Habitat-MAS](#run-habitat-mas)
      - [Perception](#perception)
      - [Mobility](#mobility)
      - [Manipulation](#manipulation)
      - [Rearrange](#rearrange)
    - [Dataset Structure](#dataset-structure)
    - [Dataset Generation](#dataset-generation)

## Installation

### Prerequisites

Please make sure you have installed the [habitat-sim](https://github.com/facebookresearch/habitat-sim/tree/v0.3.1), [habitat-lab](../README.md) and [habitat-baselines](../habitat-baselines/) following the installation guide in the previous step. 

**Note**: If you try to install the habitat suites on a Linux server without GUI, you may need to install headless version of the habitat-sim:

```sh
# Make sure nvidia-smi is available on your linux server
# $ sudo apt list --installed | grep nvidia-driver
# > nvidia-driver-xxx ...

conda install habitat-sim withbullet headless -c conda-forge -c aihabitat
```

### Install habitat-mas

To install the package, you can use the following command under habitat-lab root directory:

```sh
pip install -e habitat-mas
```

## Usage

### Download Data

#### Download Base Dataset
The dataset used in the demo is the same as [Habitat-3.0 Multi-Agent Training](../habitat-baselines/README.md#habitat-30-multi-agent-training). You can download the dataset by running the following command:

```sh
python -m habitat_sim.utils.datasets_download --uids hssd-hab hab3-episodes habitat_humanoids hab_spot_arm hab3-episodes ycb hssd-hab hab3_bench_assets rearrange_task_assets
```
#### Download MP3D Dataset
Since the mobility and rearrange tasks utilize the scene data of [Matterport3D](https://niessner.github.io/Matterport/), it is necessary to download the scene data of mp3d-habitat first according to the instructions in [here](https://github.com/facebookresearch/habitat-sim/blob/main/DATASETS.md#matterport3d-mp3d-dataset).

In short, you should first fill and sign a form to get `download_mp.py`, then run the following command in Python 2.7:
```sh
python download_mp.py --task habitat
```
Remeber to rename the folder name from 'mp3d-habitat' to 'mp3d', the path of MP3D dataset will be `EMOS/data/scene_datasets/mp3d/...`.

#### Download Habitat-MAS Dataset
Besides, you should:
Download the robot configuration and episodes data from [Here](https://drive.google.com/drive/folders/1YVoCg2-tGkKWrdej4km6Abxsop0wS9XJ?usp=drive_link), extract and merge it into EMOS like `EMOS/data/...`.

The folder should look like this:
```
EMOS
├── data
│   ├── robots
│       ├── dji_drone
│           ├── meshes
│           ├── urdf
│       ├── hab_fetch
│       ├── hab_spot_arm
|           ├── meshesDae
│           ├── urdf
|               ├── hab_spot_onlyarm_dae.urdf
│               ...
│       ├── hab_stretch
│       ├── spot_data
│           ├── spot_walking_trajectory.csv
│       ├── robot_configs
│           ├── hssd
│             ├── hssd_eval.json
│           ├── mp3d
│             ├── mobility_eval.json
│           ├── replica_cad
│             ├── two_agent_perception_eval.json
│             ...
│       ...
│   ├── datasets
│       ├── replica_cad
│           ├── single_agent_eval.json.gz
│           ...
│       ├── hssd
│           ├── hssd_eval.json.gz
│           ...
│       ├── mp3d
│           ├── mobility_eval.json.gz
│           ...
...
```

### Run the demo

The demo is adapted from [Habitat-3.0 Social Rearrangement](../habitat-baselines/README.md#social-rearrangement). You can run the demo by running the following command:

```sh
# Under the habitat-lab root directory
python -u -m habitat_baselines.run \
  --config-name=social_rearrange/llm_spot_drone.yaml \
  habitat_baselines.evaluate=True \
  habitat_baselines.num_environments=1
```

### Run Habitat-MAS

To run our episodes in Habitat-MAS, you should first set your API key in [`habitat-mas/habitat_mas/utils/models.py`](https://github.com/SgtVincent/EMOS/blob/8f4348d73fcf605ebfbeee13ff897359723b5f1c/habitat-mas/habitat_mas/utils/models.py) to run EMOS.

For each task, you could run the following command:

#### Perception
```sh
python -u -m habitat_baselines.run \
  --config-name=multi_rearrange/llm_spot_drone_per.yaml \
  # --config-name=multi_rearrange/llm_height_per.yaml \
  habitat_baselines.evaluate=True \
  habitat_baselines.num_environments=1
```

#### Mobility
```sh
python -u -m habitat_baselines.run \
  --config-name=multi_rearrange/llm_spot_fetch_mobility.yaml \
  habitat_baselines.evaluate=True \
  habitat_baselines.num_environments=1
```

#### Manipulation
```sh
python -u -m habitat_baselines.run \
  --config-name=multi_rearrange/llm_fetch_stretch_man.yaml \
  habitat_baselines.evaluate=True \
  habitat_baselines.num_environments=1
```

#### Rearrange
```sh
python -u -m habitat_baselines.run \
  --config-name=multi_rearrange/llm_spot_drone_rearrange.yaml \
  habitat_baselines.evaluate=True \
  habitat_baselines.num_environments=1
```

### Structured run trajectories

`HabitatMASEvaluator` records run configuration and execution facts by default.
Recording does not change prompts, API parameters, policy inputs, submitted
commands, success predicates, existing fallback choices or exception handling.
No detector or system-error label is run. Human annotations belong in a
separate file keyed by `run_id` and `evaluation_id`; neither events nor task
failure are ground-truth system-error labels.

```text
runs/<run_id>/
  config.json
  run.json                       # same manifest, retained for compatibility
  episodes/<episode_id>--<evaluation_id>/
    episode.json
    events.jsonl
    summary.json
    media/
```

Each evaluation/attempt gets a UUID, including repeated episodes and equal IDs
in different scenes. Episode directory names are sanitized; the original IDs
remain in the records. Paths in episode records are relative to that episode
directory. Events are flushed after every line; JSON snapshots use temporary
files and replacement. A hard kill may leave a partial last line or no summary;
earlier flushed lines are still usable. File flushing is not a checkpoint or
an exact replay guarantee. Storage failures are reported and recording is best
effort; logging errors do not replace task errors or choose recovery actions.

Use the existing evaluation command (the configured task and API account are
unchanged):

```sh
python -u -m habitat_baselines.run \
  --config-name=multi_rearrange/llm_spot_drone_per.yaml \
  habitat_baselines.evaluate=True \
  habitat_baselines.num_environments=1 \
  habitat_baselines.eval.trajectory_dir=runs
```

Disable with `habitat_baselines.eval.save_trajectories=False`. Existing optional
video recording remains controlled by `habitat_baselines.eval.video_option`;
its output directory is recorded in the config snapshot. The new `media/`
directory holds referenced decision images, not continuous video.

**Configuration and initial conditions.** `config.json.metadata` contains the
benchmark/dataset/task category when declared, safe unresolved composed config,
Hydra source/choices, Git commit and dirty status, SHA256 of relevant source,
prompt and configuration files, agent identities/types/count, seeds, step/time
limits, existing termination settings and the original success-measure config.
`episode.json` adds the episode/scene identity, task/scene context and initial
robot state. Runtime `config.json.models_used` records actual module/agent model
IDs, system-prompt hashes, explicitly passed decoding parameters and distinct
server response model IDs, plus exposed API client timeouts/retry limits.
`config.json.episodes` indexes the original scene/episode IDs and unique attempt
directories. Declared model config is kept separate from observed
requests. Unsent provider defaults are `null`, with reasons; no model settings
are silently injected. Credential keys, known secret values and credential URL
parts are redacted. The environment, API keys and authentication headers are
not dumped.

**Common event fields.** Schema version is `2`; `event` and `step` are retained
aliases for `event_type` and `env_step`.

| Field | Meaning |
| --- | --- |
| `event_id`, `sequence`, `timestamp` | Event UUID, consecutive per-episode order, UTC wall-clock time |
| `run_id`, `evaluation_id`, `episode_id`, `scene_id` | Run, unique episode attempt, dataset episode and scene identity |
| `agent_id` | Agent responsible for the event; null for aggregate environment/episode events |
| `env_step` | 1-based `env.step` transition to which the event belongs; a decision before transition 1 and its returned state both use 1; initial state uses 0 |
| `decision_step` | Per-agent top-level chat count, independent of simulation steps; nested Crab/model calls share the count, discussion tool/API rounds stay in the same decision |
| `action_id` | One execution decision/controller action across multiple simulation transitions, including a decision aborted before controller entry |
| `phase`, `call_id` | Discussion/planning/execution stage; UUID of one concrete model API call |
| `elapsed_seconds`, `duration_seconds` | Monotonic time since episode start; duration of the measured operation |
| `missing_fields` | Reasons for unavailable or unsupported fields, stored as null rather than invented values |

`num_env_steps` in the summary counts returned simulator transitions. An
exception after submission can therefore have `env_step=1` and zero returned
steps. `sequence` defines ordering across decisions, submissions, returns and
termination. A terminal event can refer to the last attempted transition.
Discussion and initial planning decisions also increment the per-agent count;
`decisions_by_phase` separates them from execution decisions. A `call_id` is
needed to distinguish multiple API calls within a single discussion decision.

| Events | Facts recorded |
| --- | --- |
| `episode_start`, `task_assignment` | Task/scene context, original initial state, assigned subtasks |
| `decision_start`, `decision_end`, `model_config` | Agent decision boundary, phase, elapsed time, exact passed settings and unknown-default sources |
| `llm_request`, `llm_response`, `llm_error` | Actual input/request kwargs, complete SDK response/message, requested and response model IDs, usage, call status, API duration and original error |
| `llm_parse`, `skill_parse`, `tool_selection`, `skill_selection` | Raw tool arguments, parse outcome, mapped skill/arguments, existing fallback flag/reason and rethrow/continue handling |
| `discussion_parse`, `termination_request` | Existing discussion regex results/fallback and the original all-agent-wait stop request |
| `agent_tool_decision`, `agent_action` | Original suggested action and existing transformations such as robot injection or send-request to wait |
| `communication_send`, `communication_receive` | Sender, recipient, original/rendered message, queued/consumed/self-suppressed status and emission timestamp |
| `action_start`, `skill_action`, `skill_termination_signal`, `action_end` | Entered skill and parameters, actual controller output, original done/timeout/interruption signals, final controller status/result |
| `env_action_submit`, `action_submit`, `action_execution`, `step` | Aggregate/per-agent command, returned reward/done/metrics, decoded Habitat action, robot and target state |
| `execution_error`, `exception`, `state_capture_error` | Original errors, handling and traceback when exposed; logging capture errors are identified separately |
| `decision_media`, `episode_end` | Image references and missing reasons; the final episode summary |

`action_execution.execution_status="returned"` means `env.step` returned. A
controller termination signal is recorded as `controller_terminated`, not as
proof that a physical action succeeded. Existing chat-history `"Success"`
placeholders are unchanged and explicitly identified as synthetic feedback.
`summary.task_success` comes only from the configured original environment
success metric, before policy extras can overwrite metrics. Error exits without
that metric have null success, not inferred failure.

**State and media.** Base position/orientation comes from the articulated robot
rather than its sensor camera; orientation is `[x,y,z,w]`. End-effector position,
grasped object ID, rearrangement target object ID/handle/current and goal
positions, scalar task progress and decoded executed Habitat command are
recorded when available. Terminal state is captured inside the worker before
automatic reset. No unsupported contact/force state or action-specific physical
success predicate is fabricated. Generic navigation environments have no
rearrangement targets or gripper; these fields are null with reasons.

Actual multimodal request image bytes are saved under `media/`, with a hash and
relative path in the logged request copy. The SDK receives the original input.
Remote image URLs are fetched with a bounded timeout/size; an unavailable image
keeps its reference with a missing reason. Repeated content is deduplicated by
hash. The current EMOS LLM path supplies text, not camera images. RGB snapshots
at decision time are therefore labeled `seen_by_llm=false`, separately from
actual image inputs; headless/no-RGB runs record why no image exists. Output
images are not labeled as model inputs. No image is expanded into a JSON pixel
array. The policy's existing single-environment limitation is retained; each
environment still receives its own simulator trace.

**End summaries.** `summary.json` includes original task success, returned step
count, per-agent/phase decision counts, elapsed time, termination reason, last
actions and agent states, final metrics/state, media/file paths and raw error.
`exception_count` counts distinct error signatures within a transition and
agent/action/decision; repeated propagation catch records remain in events and
`exception_event_count`. `logging_error_count` is separate. Existing time-limit,
skill-timeout, wait/stop and task-inactive information is recorded without
classifying a failure as a system error.

**Verification.** Run the targeted unit/integration tests in the Habitat Python
environment, without API calls:

```sh
python -m unittest discover -s test -p 'test_*trajectory*.py' -v
python -m unittest discover -s test -p test_trace_schema.py -v
python -m unittest discover -s test -p test_run_config.py -v
```

Run the real short simulator validation without an API key:

```sh
python scripts/trajectory_smoke.py --output runs/trajectory-validation
```

The script uses the installed `van-gogh-room.glb` and navmesh. It records a
4-step geodesic-follower success, a 2-step environment-time-limit failure and
an original exception from an invalid diagnostic action. Each case creates a
fresh environment because the repository's existing PointNav second-reset
step logger assumes rearrangement metrics. This script is a diagnostic harness;
it does not alter benchmark policies. These headless cases have no RGB or
model calls. Multimodal media and LLM logging use real SDK types with a fake
transport in tests; a complete paid multi-robot LLM episode was not run.

For example, the real successful trace under `runs/trajectory-validation/`
contains this projection (other fields omitted):

```jsonl
{"sequence":4,"event_type":"action_start","agent_id":"main_agent","env_step":1,"decision_step":1,"action_id":"48ab9db2-0789-4f1e-b941-af9062bcc447","status":"started"}
{"sequence":8,"event_type":"action_submit","agent_id":"main_agent","env_step":1,"decision_step":1,"action_id":"48ab9db2-0789-4f1e-b941-af9062bcc447","execution_status":"submitted"}
{"sequence":10,"event_type":"action_execution","agent_id":"main_agent","env_step":1,"decision_step":1,"action_id":"48ab9db2-0789-4f1e-b941-af9062bcc447","execution_status":"returned"}
{"sequence":26,"event_type":"action_end","agent_id":"main_agent","env_step":4,"decision_step":1,"action_id":"48ab9db2-0789-4f1e-b941-af9062bcc447","status":"controller_terminated"}
```

Filter by `agent_id` and `action_id`, then sort by `sequence` to follow the whole
action, including the additional submissions/returns at env steps 2–4. Read
`summary.task_success` separately; the controller's status alone is not truth.

**Performance.** Recording adds synchronous serialization, state queries and
file flushes; images add encoding/copy/write cost and remote input URLs add
fetch latency. A measured real headless PointNav microbenchmark (100 fixed
turn-left steps, 3 rounds, warmed up, excluding initialization) had median
0.0816 ms/step disabled and 2.8631 ms/step enabled: +2.7815 ms/step, 35.08x this
very small baseline. Final poses and metrics were identical. There were no RGB
or API calls, so these numbers are not a full MAS/LLM overhead estimate.
Raw timings are in `runs/trajectory-smoke-verified/performance.json`.

**Implementation files.** New modules: `habitat_mas/utils/trajectory.py` and
`habitat_mas/utils/run_config.py`. Observer hooks: `utils/models.py`,
`agents/crab_agent.py`, `habitat_baselines/rl/hrl/hl/llm_policy.py`,
`rl/hrl/hierarchical_policy.py`, `rl/hrl/skills/skill.py`,
`rl/multi_agent/multi_llm_policy.py`, `rl/multi_agent/habitat_mas_evaluator.py`,
and `habitat/core/environments.py`. Configuration/documentation: baselines
`config/default_structured_configs.py`, this README and root `.gitignore`.
Validation: `scripts/trajectory_smoke.py`, `test/test_trajectory_recorder.py`,
`test/test_trace_schema.py`, `test/test_run_config.py`,
`test/test_llm_trajectory.py`, `test/test_execution_trajectory.py`,
`test/test_trajectory_state.py` and `test/test_mas_trajectory_integration.py`.

### System-error detection and episode-level evaluation

The separate `habitat_mas.detection` package replays Perception, Mobility,
Rearrange and Manipulation trajectories through team-progress heuristic rules
and a structured LLM detector. A consecutive-action-name baseline is also
provided. Detectors observe only current/past data and do not alter policies.

Run `python scripts/evaluate_system_errors.py --help` from the EMOS root.
The [Chinese workflow and annotation guide](../docs/system_error_detection.md)
describes sampling units, all four batch selections, API request auditing,
cache reuse and the independent human annotation CSV. Precision and recall
are computed per episode against runtime-anomaly labels, including successful
episodes; previous failure-attribution classes are not substituted for these
labels. Unknown labels or incomplete inference keep formal metrics unavailable.

### Dataset Structure

**Multi robot perception**
`data/datasets/hssd/0/hssd_height_per.json.gz`
- Mainly deploy robots spot and drone for object perception.
- Objects are easy for drone to find, but hard for spot robot.

**Multi robot manipulation**
`data/datasets/hssd/0/hssd_height_man.json.gz` or `data/datasets/hssd/0/hssd_dist_man.json.gz`

- Mainly deploy robots fetch and stretch for object rearrangement.
- Objects are easy for stretch robot to get, but hard for fetch robot.

**Multi robot mobility**
`data/datasets/mp3d/mobility_episodes_1.json.gz`
- Mainly deploy robots spot and fetch for cross-floor navigation.
- Cross-floor tasks are easy for spot to do, but impossible for fetch robot.

**Multi robot rearrange**
`data/datasets/mp3d/mp3d_episodes_1.json.gz`
- Mainly deploy spot, fetch and drone for cross-floor rearrangement.
- Cross-floor rearrangement is suitable for spot, same-floor rearrangement is suitable for fetch, cross-floor perception is suitable for drone.

### Dataset Generation

Besides the dataset we have provided, custom data can be also generated through the scripts we provided.

Here is a demo data generation command for dataset in `hssd` scene.

```sh
python habitat-lab/habitat/datasets/rearrange/run_hssd_episode_generator.py --run --config data/config/hssd/hssd_dataset.yaml --num-episodes 340 --out data/datasets/hssd_height.json.gz --type height
```

`--config`: path of your dataset generation configuration data.

`--num-episodes`: episodes number you want to generate. (Note: in `scene_balanced` type `scene_sampler`, the number should be Integer multiple of `34`, you can customize your dataset generation configuration, which should be the file in the path of `--config`)

`--out`: desired path of your newly generated dataset.

`--type`: the purpose of your dataset, currently there are three types: `height`, `distance`, `normal`.
