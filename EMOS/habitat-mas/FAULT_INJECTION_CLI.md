# 错误注入命令行

无错误注入、随机评估 40 个 episode 并记录正常轨迹，见[正常评估使用文档](CLEAN_EVALUATION_GUIDE.md)。

**保留 EMOS 原有参数，只把错误注入改成简写参数。**
在项目根目录使用 `python scripts/inject_fault.py`，原来的 `--config-name` 和
Hydra `key=value` 参数可以继续直接传入，不需要额外的 `--` 分隔符。

## 指定 episode 和注入步数

```bash
env -u ALL_PROXY -u all_proxy python scripts/inject_fault.py \
  --config-name=multi_rearrange/llm_spot_drone_per.yaml \
  habitat_baselines.evaluate=True \
  habitat_baselines.num_environments=1 \
  habitat_baselines.test_episode_count=1 \
  habitat.environment.iterator_options.shuffle=False \
  --fault hang --agent agent_1 --episode 28 --step 40
```

原有 EMOS 参数仍由你设置；最后一行表示在 episode 28 的第 40 个仿真步之前
挂起 agent_1。步数从 0 开始，挂起持续到当前 episode 结束。

当前支持 `hang` / `agent_hang` 和 `dropout` / `agent_dropout`，也接受带引号的
`"agent hang"`、`"agent dropout"`。`--agent` 默认 `agent_0`。
多个指定 episode 可写成 `--episode 28,49 --step 20`，并相应设置
`habitat_baselines.test_episode_count=2`。

## 随机选择 episode 和注入步数

```bash
python scripts/inject_fault.py \
  --config-name=multi_rearrange/llm_spot_drone_per.yaml \
  habitat_baselines.evaluate=True \
  habitat_baselines.num_environments=1 \
  habitat_baselines.test_episode_count=10 \
  habitat.environment.iterator_options.shuffle=False \
  --fault dropout --agent agent_1 --random-episodes 10 --seed 42 \
  --trajectory fault_output/clean_random10_stepwise_state_trajectory.jsonl
```

在当前数据集与正常轨迹的交集中，按 **scene_id + episode_id** 匹配，无放回抽取
N 个不同 episode ID；同一 ID 若有多个场景变体，再随机选择其中一个。
每个选中 episode 有 L 个实际执行步时，从 `0` 到 `L-1` 均匀抽取一步，并受本次
环境步数上限约束。只接受 `mode=clean`、从第 0 步连续记录到 `done=True` 的完整
JSONL 轨迹。未完成的 episode 不参与抽样；重复运行、缺步、错误轨迹或候选不足
会报错，不会退回到按环境最大步数随机选择。

`--seed` **仅控制故障抽样**，不修改 `habitat.seed`、`habitat.simulator.seed`。
EMOS 本身的种子继续使用原有参数。相同数据集、轨迹和抽样种子会生成相同计划。
重新运行可能比正常轨迹提前结束；若未到达预选步数，评估器会报告注入未触发。

也可用 `--episode 28 --trajectory <文件>` 给指定 episode 随机选一步，或用
`--random-episodes 10 --step 40` 随机选择 episode、统一在第 40 步注入。

## 原有参数的处理

脚本不再自动设置评估模式、环境数量、评估次数、shuffle、检查点选项、evaluator、
iterator 选项、仿真种子或 Hydra 工作目录。这些参数按原始命令和 YAML 配置组合。
`shuffle=True` 也可保留：错误按 episode ID 调度，不依赖评估顺序。

当前注入实现要求评估模式、单环境、每个 episode 评估一次，且不能启用
`load_resume_state_config` 让旧检查点配置覆盖本次参数。评估数量
`test_episode_count` 必须等于选中 episode 数或为 `-1`（评估选中的全部 episode）。
若 iterator 设置了 `num_episode_sample`，它也须等于选中数量或为 `-1`。
不兼容的设置会明确报错，**不会自动覆盖你提供的值**。

只有错误注入配置子树由简写参数生成，请勿同时传入与之冲突的
`habitat_baselines.fault_injection.*` 参数。

原来的 `habitat.dataset.data_path` 用于读取源数据集。为保证选中的 episode
真正被执行，脚本生成仅含这些 episode 的临时评估子集，并在子进程中使用这个
子集路径；源数据文件不修改，原始参数与源路径保存在 `plan.json` 中。
数据集须为包含内联 `episodes` 的 JSON/JSON.gz，当前不加载按场景拆分的 content 文件。

## 重复 episode ID

数据中不同场景可能共用 ID，例如 episode 15。固定选择有歧义时，脚本会列出
候选场景；在原命令后加 `--scene` 指定场景完整路径或去掉最后扩展名的文件名，
例如 `--scene data/replica_cad/configs/scenes/v3_sc3_staging_09.scene_instance.json`。
随机模式保证本次选中 ID 唯一，并保留原 ID，避免破坏机器人配置的索引。

## 预览与输出

在原命令后加 `--dry-run` 只生成计划和数据子集，不调用模型或启动仿真。
输出默认在 `fault_output/<错误类型_时间戳>/`，包含：

- `plan.json`：原始 EMOS 参数、错误类型、种子、场景、episode、步数和完整运行命令；
- `episodes.json.gz`：选中的评估数据子集；
- 对应错误类型的事件 JSONL、逐步轨迹 JSONL 和可读文本轨迹。

`--output-dir` 修改输出根目录，`--run-label` 指定子目录；已有目录不会覆盖。
`--no-trajectory` 仅关闭轨迹记录，事件仍保留。
`--config` 是 `--config-name` 的别名；`--dataset`、`--max-episode-steps` 是原数据集
路径和环境步数上限参数的可选简写，两种写法同时出现且值不一致时会报错。

缺少 Habitat 依赖时，离线预览需要显式提供数据路径和步数上限，可继续用 EMOS 写法：

```bash
python scripts/inject_fault.py \
  --config-name=multi_rearrange/llm_spot_drone_per.yaml \
  habitat_baselines.evaluate=True \
  habitat_baselines.num_environments=1 \
  habitat_baselines.test_episode_count=1 \
  habitat.environment.iterator_options.shuffle=False \
  habitat.dataset.data_path=data/datasets/replica_cad/two_agent_perception_eval.json.gz \
  habitat.environment.max_episode_steps=750 \
  --fault hang --episode 28 --step 40 --dry-run
```

离线预览只校验显式参数，不验证完整 Hydra 配置和机器人是否存在。
额外 EMOS 参数仍会传给最终运行命令。涉及 `content_scenes` 的过滤需要完整配置
环境，或在离线模式使用 `--scene`。

## 扩展与测试

新错误实现对应配置与运行时处理后，在 `habitat_mas/fault_injection/launcher.py`
的 `FAULTS` 注册配置路径、校验类、别名及额外配置。公共参数为 `enabled`、
`agent`、`episode_id`、`simulator_step_schedule`、`record_trajectory`、`output_dir`、
`run_label`。数据选择、抽样和计划保存无需为每个错误重复实现。

```bash
PYTHONPATH=habitat-mas python -m unittest \
  habitat_mas.test.test_fault_launcher \
  habitat_mas.test.test_agent_hang_fault \
  habitat_mas.test.test_agent_hang_policy \
  habitat_mas.test.test_agent_dropout_fault -v
```

这些测试包含原参数保留、冲突校验、抽样、数据子集、命令及策略分派验证。
完整 Hydra 配置组合和原生 Habitat 仿真需要安装相应依赖后验证。
