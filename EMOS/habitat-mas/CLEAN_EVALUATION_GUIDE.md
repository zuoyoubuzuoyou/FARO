# EMOS 无错误注入运行文档

本指南使用 EMOS 原始入口 `python -u -m habitat_baselines.run`，保留原有
`--config-name` 和 Hydra `key=value` 参数。

## 命令行直接指定 episode（推荐）

现在可以直接运行，不用手动准备单 episode 数据文件：

```bash
conda activate habitat
cd /home/dyc/Projects/FARO/EMOS
python scripts/run_clean.py --episode 20 --run-label clean_episode20_replay_v3
```

这个入口默认使用 `multi_rearrange/llm_spot_drone_per.yaml`，关闭 dropout、hang
和轨迹回放，开启正常轨迹记录（包括高层决策处的 `replay_context`）。默认使用
单环境评估选中的全部 episode，每个 episode 一次，并关闭旧检查点配置恢复。
显式提供的 EMOS 参数会保留；若与单环境评估要求冲突则报错。模型、仿真种子和
其他参数沿用配置，可以继续在命令后追加原有 Hydra `key=value` 参数。

`--episode` 指数据中的 ID，不是数据集下标。脚本从原始数据集筛选匹配条目，
自动把子集和 `plan.json` 保存到 `fault_output/<run_label>/`，再调用原始评估入口。
原始数据集及 episode ID 不修改。ID 在多个场景重复时，用 `--scene` 指定场景。

支持 `--episode 20,28`、`--random-episodes 10 --seed 42`，以及 `--config-name`、
`--dataset`、`--output-dir`、`--dry-run`。`--seed` 仅控制 episode 抽样；
仿真种子仍使用 `habitat.seed` 等原有参数。与原始入口不同，简写入口拒绝覆盖
已有同名输出；重复执行或在 dry-run 后正式运行时请换一个 `--run-label`。

上面的命令生成：

```text
fault_output/clean_episode20_replay_v3_trajectory.jsonl
fault_output/clean_episode20_replay_v3_trace-episode_20.txt
```

后续故障命令将 `--replay-trajectory` 指向这个新的 JSONL 即可。
`--help` 查看完整选项。以下原始入口的用法也继续支持。

clean 也默认启用错误检测器：正常等待不会触发；如果检测到成员掉线、连续
10 步无响应或无进展的连续重复通信，会保存 `<run_label>_crash.json` 并以
退出码 3 停止整个评估。错误注入开关和检测开关彼此独立。需要保留原来的
无监控行为时，追加 `habitat_baselines.error_detection.enabled=False`。
规则、阈值与报告说明见 [错误检测文档](FAULT_INJECTION_CLI.md#error-detection-system检测并模拟系统崩溃)。

## 公共记录与输出参数

clean、dropout 和 hang 统一使用以下参数：

```text
habitat_baselines.trajectory.enabled=True
habitat_baselines.output_dir=fault_output
habitat_baselines.run_label=clean_random10_seed42
```

`trajectory.enabled` 控制逐步 JSONL 和 TXT 轨迹；`output_dir` 和 `run_label`
同时用于轨迹、故障事件和崩溃报告，不改变视频或 LLM 历史的输出位置。
原始入口默认不记录轨迹，输出目录默认 `fault_output`，空标签按实际运行类型
使用 `clean`、`dropout` 或 `hang`。两个简写入口默认记录轨迹并生成时间戳标签。

旧的 `fault_injection.structure.agent_dropout/agent_hang` 下的
`record_trajectory`、`output_dir`、`run_label` 字段已移除，请迁移到上述公共参数。
故障 `enabled` 默认均为 `False`，默认配置下的 clean 命令无需逐个关闭故障。

```bash
python -u -m habitat_baselines.run \
  --config-name=multi_rearrange/llm_spot_drone_per.yaml \
  habitat_baselines.evaluate=True \
  habitat_baselines.num_environments=1 \
  habitat_baselines.test_episode_count=1 \
  habitat.environment.iterator_options.shuffle=False \
  habitat_baselines.trajectory.enabled=True \
  habitat_baselines.run_label=clean
```

## 1. 随机选择 40 个 episode，记录正常轨迹

在 EMOS 项目根目录、已配置好 Habitat 与模型服务的环境中执行：

```bash
env -u ALL_PROXY -u all_proxy python -u -m habitat_baselines.run \
  --config-name=multi_rearrange/llm_spot_drone_per.yaml \
  habitat_baselines.evaluate=True \
  habitat_baselines.num_environments=1 \
  habitat_baselines.test_episode_count=40 \
  habitat.environment.iterator_options.num_episode_sample=40 \
  habitat.environment.iterator_options.shuffle=True \
  habitat.environment.iterator_options.group_by_scene=False \
  habitat.seed=42 \
  habitat_baselines.trajectory.enabled=True \
  habitat_baselines.output_dir=fault_output \
  habitat_baselines.run_label=clean_random40_seed42
```

这里使用默认关闭的故障配置，没有错误注入。公共的
`habitat_baselines.trajectory.enabled=True` 开启轨迹记录，输出中的 `mode` 为
`clean`。输出文件以实际运行类型 `clean` 开头。

相对于原来的单 episode 命令，随机抽取 40 个的关键设置如下：

| 参数 | 值 | 作用 |
| --- | --- | --- |
| `habitat_baselines.test_episode_count` | `40` | 本次评估 40 个 episode |
| `habitat.environment.iterator_options.num_episode_sample` | `40` | 从加载的数据集中无放回随机抽取 40 条 episode |
| `habitat.environment.iterator_options.shuffle` | `True` | 打乱抽取后的执行顺序 |
| `habitat.environment.iterator_options.group_by_scene` | `False` | 不再按场景重新分组，保留随机顺序 |
| `habitat.seed` | `42` | 固定 episode 抽样与相关随机过程的种子 |

抽样发生在 episode iterator 初始化时；`num_episode_sample=40` 明确限定候选集，
`test_episode_count=40` 则限定评估数量。单纯把 `shuffle` 改为 `True` 只会打乱
顺序，不会把待评估数量从 1 改成 40。

数据集经过场景过滤后必须至少有 40 条 episode。当前默认数据文件
`data/datasets/replica_cad/two_agent_perception_eval.json.gz` 有 154 条 episode，
满足这一要求。默认每个 episode 评估一次；不要另行设置多次重复评估。

## 2. 更换抽样数量或种子

随机评估 N 个 episode 时，同时修改：

```text
habitat_baselines.test_episode_count=N
habitat.environment.iterator_options.num_episode_sample=N
```

更换 `habitat.seed` 可得到另一组随机样本，同时修改 `run_label` 避免覆盖轨迹。
例如下一组仍评估 40 个，可使用 `habitat.seed=43` 和
`run_label=clean_random40_seed43`。

在相同数据集、配置和软件环境下，固定种子可复现 episode 抽样。
模型输出和最终执行轨迹仍可能存在差异，不能仅凭种子保证 LLM 行为完全一致。

## 3. 输出文件

主命令默认会产生以下输出：

| 输出 | 路径或内容 |
| --- | --- |
| 完整正常轨迹 JSONL | `fault_output/clean_random40_seed42_trajectory.jsonl` |
| 可读逐步轨迹 | `fault_output/clean_random40_seed42_trace-episode_<id>.txt` |
| 仿真视频 | `video_multi/spot_drone/` |
| LLM 讨论、行动历史与 token 用量 | `chat_history_output/<日期>/llm_spot_drone_per/FULL/<episode_id>/` |
| TensorBoard 指标 | `tb/` |
| 终端输出 | 当前 episode、任务分配、动作，以及最终奖励、成功和步数等指标 |

不会新建本次错误注入事件文件。已有旧事件文件不会被删除，不能据其存在判断
本次是否注入了错误。

JSONL 中每条 `simulator_step` 记录包含 `scene_id`、`episode_id`、
从 0 开始的 `simulator_step`、动作、状态、奖励、指标及 `done`。
正常运行的记录带有 `mode=clean`，完成一个 episode 时最后一步为 `done=True`。

同一个 `output_dir` 和 `run_label` 的轨迹 JSONL 会在启动时重新写入；重复运行请
换 `run_label` 或输出目录。视频、LLM 记录仍采用各自的 EMOS 输出路径。

## 4. 确认是否完成 40 个 episode

进程正常结束后，可用以下只读检查统计完整轨迹：

```bash
python - <<'PY'
import json
from pathlib import Path

path = Path('fault_output/clean_random40_seed42_trajectory.jsonl')
completed = {}
with path.open(encoding='utf-8') as source:
    for line in source:
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get('type') != 'simulator_step':
            continue
        if row.get('mode') != 'clean':
            raise ValueError('发现非 clean 轨迹')
        if row.get('done'):
            key = (row['scene_id'], str(row['episode_id']))
            completed[key] = row['simulator_step'] + 1

print('完成的 scene/episode 数量:', len(completed))
print('不同 episode ID 数量:', len({key[1] for key in completed}))
for (scene, episode), steps in completed.items():
    print(f'episode={episode}, steps={steps}, scene={scene}')
if len(completed) != 40:
    raise SystemExit('尚未记录完整的 40 个 episode，请检查运行是否中断或配置是否改变')
PY
```

任务失败、碰撞结束、达到步数上限都可以是完整 episode；完整不等于任务成功。
中断运行时，仅有 `done=True` 的 episode 可作为后续随机注入的完整正常轨迹。

## 5. episode ID 重复时如何理解数量

Habitat 原生抽样按 episode 条目进行。同一个 `episode_id` 可能出现在不同场景中，
应使用 **scene_id + episode_id** 确定一个 episode。当前默认数据集有 154 个不同
的 scene/episode 组合，但只有 130 个不同的 episode ID。

因此，随机评估 40 个 episode 不保证获得 40 个不同 ID。JSONL 保留 scene_id，
可正确区分这些轨迹；当前可读文本和 LLM 记录的文件名只使用 episode ID，同名
ID 的后一次运行可能覆盖前一次文件，应以完整 JSONL 为准。

后续错误注入 CLI 的一次批次要求 ID 不重复。如果要从这份正常轨迹中随机挑选
10 个 episode 注入，需有至少 10 个带完整轨迹的不同 ID；如果需要 40 个不同 ID，
则应先准备 ID 唯一的数据集，或记录更多正常 episode。

## 6. 常用可选参数

以下参数可以直接追加到原始 EMOS 命令中：

| 需求 | 参数 |
| --- | --- |
| 关闭视频以减少录制和磁盘开销 | `habitat_baselines.eval.video_option=[]` |
| 修改视频目录 | `habitat_baselines.video_dir=video_multi/clean_random40_seed42` |
| 修改 TensorBoard 目录 | `habitat_baselines.tensorboard_dir=tb/clean_random40_seed42` |
| 不保存逐步 JSONL | `habitat_baselines.trajectory.enabled=False` |
| 更换源数据集 | `habitat.dataset.data_path=<数据集路径>` |

本指南命令已按当前代码中的参数和 iterator 抽样行为核对；未启动 40 个 episode
的实际模型调用和仿真评估。错误注入用法见[错误注入命令行文档](FAULT_INJECTION_CLI.md)。
