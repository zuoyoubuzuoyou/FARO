# EMOS 无错误注入运行文档

本指南使用 EMOS 原始入口 `python -u -m habitat_baselines.run`，保留原有
`--config-name` 和 Hydra `key=value` 参数。

python -u -m habitat_baselines.run \
  --config-name=multi_rearrange/llm_spot_drone_per.yaml \
  habitat_baselines.evaluate=True \
  habitat_baselines.num_environments=1 \
  habitat_baselines.test_episode_count=1 \
  habitat.environment.iterator_options.shuffle=False \
  habitat_baselines.fault_injection.structure.agent_dropout.record_trajectory=True \
  habitat_baselines.fault_injection.structure.agent_dropout.run_label=clean

## 1. 随机选择 40 个 episode，记录正常轨迹

在 EMOS 项目根目录、已配置好 Habitat 与模型服务的环境中执行：

```bash
env -u ALL_PROXY -u all_proxy python -u -m habitat_baselines.run \
  --config-name=multi_rearrange/llm_spot_drone_per.yaml \
  habitat_baselines.evaluate=True \
  habitat_baselines.num_environments=1 \
  habitat_baselines.test_episode_count=20 \
  habitat.environment.iterator_options.num_episode_sample=20 \
  habitat.environment.iterator_options.shuffle=True \
  habitat.environment.iterator_options.group_by_scene=False \
  habitat.seed=42 \
  habitat_baselines.fault_injection.structure.agent_dropout.enabled=False \
  habitat_baselines.fault_injection.structure.agent_hang.enabled=False \
  habitat_baselines.fault_injection.structure.agent_hang.record_trajectory=False \
  habitat_baselines.fault_injection.structure.agent_dropout.record_trajectory=True \
  habitat_baselines.fault_injection.structure.agent_dropout.output_dir=fault_output \
  habitat_baselines.fault_injection.structure.agent_dropout.run_label=clean_random20_seed42
```

这里没有错误注入。两个 `enabled=False` 显式关闭现有错误；
`agent_dropout.record_trajectory=True` 仅复用轨迹记录器，输出中的 `mode` 为
`clean`。输出文件以实际运行类型 `clean` 开头，不包含未启用的故障名称。

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
| 不保存逐步 JSONL | `habitat_baselines.fault_injection.structure.agent_dropout.record_trajectory=False` |
| 更换源数据集 | `habitat.dataset.data_path=<数据集路径>` |

本指南命令已按当前代码中的参数和 iterator 抽样行为核对；未启动 40 个 episode
的实际模型调用和仿真评估。错误注入用法见[错误注入命令行文档](FAULT_INJECTION_CLI.md)。
