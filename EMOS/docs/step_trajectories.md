# 按环境 step 保存轨迹

当前录制器使用 V3：每次实际返回的 `env.step` 对应 `steps.jsonl` 的一行，步号为 `1..N`。初始状态单独保存；模型调用、通信和动作事件归入所属环境步，保持原有事件顺序。

## 一个 episode 的文件

```text
<trajectory_dir>/run_20261026_015831/
  config.json
  episodes/episode_0002/
    episode.json
    steps.jsonl
    summary.json
    pending.json                # 有尚未提交的记录时存在
    payloads/<sha256>.json       # 有需要复用的大字段时存在
    media/<sha256>.<ext>         # 有图像时存在
```

运行目录使用 `run_年月日_时分秒`，时间为北京时间；同一秒的多次运行会加 `_02`、`_03` 等序号。episode 目录包含真实 ID，数字至少补齐四位，例如 ID 2 为 `episode_0002`；同一运行内重复出现时使用 `episode_0002_attempt_02`，避免覆盖。非数字 ID 会清理路径特殊字符。

`config.json` 保存运行配置和来源信息；不再重复生成 `run.json`，其中 `episodes[].directory` 指向实际目录。`episode.json` 保存 episode 身份、任务上下文和初始状态。内部 `run_id`、`evaluation_id` 继续使用唯一 UUID；目录名称与这些身份标识分开。多个环境也分别保存自己的 episode 目录。

`steps.jsonl` 的每行包含 `schema_version: 3`、`step`、`identity`、实际提交的 `action`、`reward`、`done`、`metrics`、返回的 `state` 和有序 `events`。有新决策时还包含 `replay_context`。字段缺失时保留原来的缺失含义，不用空对象或 `false` 补齐。步内事件保留动作、模型调用和决策标识，方便核对关联。

动作向量和环境转换后的 Habitat 命令分别保留。模型建议、解析结果、实际进入的技能及执行结果也分别记录。原生成功值、终止原因和返回状态保持原有语义，工具返回的占位 `Success` 不等同于 episode 成功。

`payloads/` 用内容哈希复用消息、完整请求/响应、工具定义和决策上下文；步内重复状态、动作和指标引用本行数据。看到 `{"$trace": [...]}` 表示存储引用，请通过下方读取器展开。复制轨迹时应复制整个 episode 目录，不能只复制 `steps.jsonl`。

`pending.json` 保存环境步提交前已产生的证据。API 出错、环境调用失败或进程中断时，已有请求和错误不会被丢弃，也不会新增一个虚构的环境步。正常提交后删除该 pending 文件。读取器忽略未以换行结束的最后一行，并处理“步已提交、pending 尚未清除”的中断窗口；这不提供恢复写入或仿真检查点功能。

`summary.json` 保存完成/错误/中断状态、原生成功值、步数、奖励和统计，较大的最终状态可以引用最后一步。原生成功值未知时保留 `null`。

## 读取与兼容

在已配置项目 Python 环境、能导入 `habitat_mas` 时：

```python
from habitat_mas.utils.trajectory_storage import iter_trace_events, read_trace_summary

episode_dir = "analysis_results/.../episodes/episode_0078"
for event in iter_trace_events(episode_dir):
    # 按原始顺序得到展开的事件，默认包含尚未提交的 pending 证据。
    print(event.get("env_step"), event.get("event_type"))

summary = read_trace_summary(episode_dir)
```

只读取已提交记录可传 `include_pending=False`。读取器验证步号、episode 身份和引用哈希，损坏或缺失的引用会报错。

`iter_trace_events` 也支持旧 V1/V2 JSONL 路径；旧记录按原样返回。检测器的 `iter_frames` 和稳定性脚本已使用统一读取器。历史结果不会自动改写。

## 运行

在已激活 Habitat Python 环境的终端、项目根目录，可以直接启动 evaluator：

```bash
python -u -m habitat_baselines.run \
  --config-name=multi_rearrange/llm_spot_drone_per.yaml \
  habitat_baselines.llm_model=gpt-5.6-sol \
  habitat_baselines.llm_temperature=0.0 \
  habitat_baselines.llm_seed=100 \
  habitat_baselines.evaluate=True \
  habitat_baselines.num_environments=1 \
  habitat_baselines.test_episode_count=1 \
  habitat.environment.iterator_options.shuffle=False \
  habitat_baselines.eval.save_trajectories=True \
  habitat_baselines.eval.trajectory_dir=analysis_results/perception_steps_v3
```

`habitat_baselines.eval.save_trajectories=True` 开启保存；改成 `False` 关闭。当前配置默认值为 `True`，可以通过命令行显式覆盖。在线恢复开启时会强制录制恢复所需轨迹，即使这个开关为 `False`。`trajectory_dir` 指定输出根目录。

`habitat_baselines.llm_model=模型名` 控制讨论、规划和执行阶段使用的任务模型；不传时默认 `gpt-5.6-sol`。模型名从配置直接传给讨论、规划和执行模块，不读取 `EMOS_LLM_MODEL`，终端中遗留的该变量也不会影响模型选择。底层代码显式传入 `OpenAIModel(model=...)` 的调用保留其指定模型。命令行中的空模型名称会在启动任务前报错。

`habitat_baselines.llm_temperature=0.0` 和 `habitat_baselines.llm_seed=100` 将这两个参数显式发送到讨论、规划和执行阶段的每次模型请求，并记录到轨迹的请求参数中。`llm_seed` 是模型侧 seed，与 `habitat.seed`、`habitat.simulator.seed` 分开设置；不传某个解码参数时默认 `null`，该参数不随请求发送。回放上下文也保存配置值。稳定性脚本通过相同配置入口传入其已有实验设置。

运行默认 HSSD Perception 数据集的全部 113 个 episode、每个一次，可以使用：

```bash
python -u -m habitat_baselines.run \
  --config-name=multi_rearrange/llm_height_per.yaml \
  habitat_baselines.llm_model=gpt-5.6-sol \
  habitat_baselines.llm_temperature=0.0 \
  habitat_baselines.llm_seed=100 \
  habitat_baselines.evaluate=True \
  habitat_baselines.num_environments=1 \
  habitat_baselines.test_episode_count=-1 \
  habitat_baselines.eval.evals_per_ep=1 \
  habitat.dataset.data_path=data/datasets/hssd/0/hssd_height_per.json.gz \
  habitat.environment.iterator_options.shuffle=False \
  habitat_baselines.eval.save_trajectories=True \
  habitat_baselines.eval.trajectory_dir=analysis_results/perception_hssd_all_steps_v3
```

这条命令按数据集顺序运行一个 episode，`test_episode_count=1` 是运行数量，不是指定 episode ID。当前分支没有注册旧示例中的 `fault_injection.object_localization.*` 参数；其 `record_trajectory` 不能替代上面的轨迹开关。

MAS evaluator 的配置入口为：

```text
habitat_baselines.eval.save_trajectories=True
habitat_baselines.eval.trajectory_dir=analysis_results/my_run/trajectories
```

稳定性实验可以在项目根目录运行：

```bash
/home/dyc/.conda/envs/habitat/bin/python scripts/run_episode_stability.py \
  --output analysis_results/stability_steps_v3
```

这个脚本当前默认选择 10 个 Manipulation episode，各运行 3 次，按顺序启动独立进程，使用 GPU 1。模型、temperature 和 seed 使用脚本内的 `DEFAULT_SETTINGS`。使用新的输出目录；该命令会启动真实仿真和模型请求。

## 回放数据的边界

真实新决策后记录 agent 的分组对话历史、system message、任务与规划状态、恢复指令、模型配置及有序通信队列。`CrabAgent.export_replay_context()` / `restore_replay_context()` 支持在兼容的现有对象上导出与恢复这些内存状态，恢复前验证结构与配置。

本次提供录制、读取和决策内存重建。完整的历史决策驱动仿真回放执行器尚未迁移，不能据此保证物理轨迹逐步一致。旧文件没有记录过的上下文也不能通过格式转换补出。

## 离线验证

使用历史 episode 78 转换到独立的 `analysis_results/step_format_validation_ep78_20261025_compact/`，未发起新仿真或 API 请求：339 行环境步，3556 条展开事件与原始记录完全一致，339 个检测输入及最终 summary 一致（文件索引按新目录调整）。包含 config、metadata、summary、图片和全部 payload 的总大小从 6,734,411 字节降到 5,408,725 字节，减少 19.69%。验证结果保存在该目录的 `validation.json`；原始历史文件保持原样。
