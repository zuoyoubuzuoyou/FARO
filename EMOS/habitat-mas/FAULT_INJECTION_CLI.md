# 错误注入命令行

无错误注入、随机评估 40 个 episode 并记录正常轨迹，见[正常评估使用文档](CLEAN_EVALUATION_GUIDE.md)。

**保留 EMOS 原有参数，只把错误注入改成简写参数。**
在项目根目录使用 `python scripts/inject_fault.py`，原来的 `--config-name` 和
Hydra `key=value` 参数可以继续直接传入，不需要额外的 `--` 分隔符。

## 公共记录与输出参数

clean、dropout、hang 使用同一组配置，原始入口与简写脚本均可传入：

```text
habitat_baselines.trajectory.enabled=True
habitat_baselines.output_dir=fault_output
habitat_baselines.run_label=my_run
```

轨迹开关只控制逐步 JSONL/TXT，关闭后故障事件和检测到的崩溃报告仍会输出。
公共目录和标签用于这三类输出，视频和 LLM 历史继续使用原有专用路径。
原始入口的轨迹开关默认 `False`，目录默认 `fault_output`，标签默认空字符串，
按实际运行类型命名为 `clean`、`dropout` 或 `hang`。

简写脚本默认开启记录并生成时间戳标签；`--no-trajectory`、`--output-dir`、
`--run-label` 分别是公共开关设为 `False`、输出目录、标签的别名。也可直接传入
上述 Hydra 参数；两种写法同时出现且值不同会报错。输出参数会规范化为计划中的
路径与值，保证打印的路径与实际输出一致。

故障配置中原有的 `record_trajectory`、`output_dir`、`run_label` 字段已移除，
旧命令需迁移到公共参数。故障 `enabled`、agent 和触发步等设置保持原有位置。

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

## 沿 clean 轨迹回放，再注入错误

`--replay-trajectory` 与 `--step` 配合使用。下面的命令读取已有 episode 20，
复现并校验第 **0～39 步**，在第 **40 步执行动作前**让 agent_1 掉线：

```bash
python scripts/inject_fault.py \
  --config-name=multi_rearrange/llm_spot_drone_per.yaml \
  habitat_baselines.evaluate=True \
  habitat_baselines.num_environments=1 \
  habitat_baselines.test_episode_count=1 \
  habitat.environment.iterator_options.shuffle=False \
  --fault dropout --agent agent_1 --episode 20 --step 40 \
  --replay-trajectory fault_output/clean_random10_seed42_trajectory.jsonl \
  --run-label dropout_replay_episode20_step40
```

这与 `--trajectory` 不同：后者**仅从日志抽样注入步号**，不会回放。
两者不能同时使用。回放要求 `1 <= step < clean 轨迹长度`；第 0 步注入没有
共享前缀，直接使用普通 `--step 0`。支持 `dropout` 和 `hang`，也支持用
`--random-episodes N --step T --replay-trajectory FILE` 选择有完整记录的 episode。
选择的每个 episode 都必须有足够长的前缀，否则报错。

回放前缀时不调用 leader 分工、LLM 子任务规划或 LLM 动作生成，而是复用记录的
高层决策。底层技能仍逐步运行，正常更新当前技能、计数器、RNN 和前一步动作；
不是把机器人坐标逐帧强行覆盖，也不是在注入点重新初始化策略。实际生成的联合
环境动作必须与 clean 匹配，执行后还会检查机器人位置、朝向、持物状态、目标物体
位置、奖励、done 以及记录中的任务指标。任何偏差都会立即停止，不能继续作为有效对照。
数值比较使用绝对容差 `--replay-atol`，默认 `0.00001`；设置为 `0` 要求记录值精确相等。

第 T 步起停止读取 clean 的动作与决策，进入故障运行。默认开启下面的错误检测器：
dropout 当步终止；hang 达到连续无响应阈值后终止。回放本身不增加自动接管机制。

新录制的轨迹在高层决策中保存 `replay_context`，包含任务分配、模型对话历史和
通信队列，回放时恢复这些上下文。**旧日志没有这些信息时，仅能用已记录的决策
重建历史**，事件标为 `context_fidelity=decision_history_only`；不能宣称原始
LLM 记忆和通信完全相同。新记录完整时为 `context_fidelity=recorded`。
若实验要求原始对话/通信也严格相同，应重新录制 clean 后配对运行。
轨迹不是完整仿真或随机数快照，仍需保持场景、配置、代码版本、checkpoint 和种子一致；
校验覆盖日志记录的字段，不证明未记录的关节、速度、传感器像素或随机状态完全相同。

JSONL 额外包含以下事件（TXT 继续记录实际执行轨迹）：

- `replay_start`：源文件、分叉步、容差和上下文完整性。
- `replay_prefix_verified`：前缀全部匹配，`verified_steps=T`，下一步注入。
- `replay_failed`：执行中发现偏差，记录首个错误字段并中止。

`--dry-run` 可先检查源日志、目标 episode 和步数并生成计划，不运行模型或仿真。
离线检查通过不表示实际物理轨迹已经匹配；运行时的 `replay_prefix_verified`
才表示前缀通过校验。回放不支持 `--no-trajectory`，以保留校验结果和后续轨迹。

## Error detection system：检测并模拟系统崩溃

`HabitatMASEvaluator` 默认启用独立于 LLM 决策的检测器，每个仿真步检查团队状态。
检测触发时抛出 `DetectedSystemCrash`，由评估器的 `except` 捕捉、保存崩溃结果、
关闭所有 agent 所在的环境，并以 **退出码 3** 结束本次评估。不会继续下一步或
后续 episode，也不会等待正在执行 `Wait[500]` 的队友再次调用 LLM。
这是明确的模拟崩溃结果，不是 Python 意外报错或等待到环境步数上限。

| 检测规则 | 默认阈值 | 终止位置 |
| --- | --- | --- |
| 有 agent 不在线（dropout） | 立即 | 当前步所有 agent 的策略、LLM 和环境动作执行前 |
| 有 agent 连续无响应（hang） | 10 个已执行的仿真步 | 第 10 个无响应步结束后 |
| 同一发送者连续发送相同内容给同一接收者，期间没有物理/任务进展 | 10 条实际发送的消息 | 当前步环境动作执行前 |

所有步号从 0 开始。注入点为 40 时，dropout 只执行第 0～39 步，
`detected_step=40`、`completed_steps=40`；hang 执行第 40～49 步共 10 个
无响应步后终止，`detected_step=49`、`completed_steps=50`。
hang 阈值改为 20 时，在第 59 步结束后终止，共执行 60 步。

使用你已记录的 clean 轨迹运行 hang 检测：

```bash
env -u ALL_PROXY -u all_proxy python scripts/inject_fault.py \
  --fault hang --agent agent_1 --episode 20 --step 40 \
  --replay-trajectory fault_output/clean_20260914_004649_044798_trajectory.jsonl \
  habitat_baselines.evaluate=True \
  habitat_baselines.num_environments=1 \
  habitat_baselines.test_episode_count=1 \
  habitat.environment.iterator_options.shuffle=False \
  habitat_baselines.error_detection.hang_steps=10
```

将 `--fault hang` 改为 `--fault dropout` 即运行掉线检测。默认生成新的时间戳标签；
也可以用 `--run-label` 指定未使用的名字。以下 Hydra 参数适用于注入入口、clean
入口和原始 `python -m habitat_baselines.run` 入口：

```text
habitat_baselines.error_detection.enabled=True
habitat_baselines.error_detection.hang_steps=20
habitat_baselines.error_detection.repeated_message_limit=10
```

设 `habitat_baselines.error_detection.enabled=False` 可以对照原来的“注入后继续
运行直到任务结束/超时”的行为。当前检测器只支持单环境评估。

检测过程和结果会记录 `error_detection.all_agents_online`、
`error_detection.all_agents_responsive`、`error_detection.no_message_loop`、
`error_detection.max_unresponsive_steps`、`error_detection.system_crash` 等指标。
崩溃时任务成功指标 `pddl_success=0`，`system_crash=1`。额外生成
`fault_output/<run_label>_crash.json`，包含原因、agent 状态、检测步、实际完成步数、
连续异常计数、检测阶段、最终状态和指标。JSONL 添加 `type=system_crash` 的终止
事件，TXT 添加 `System_Crash`。即使 `--no-trajectory`，崩溃报告也会保存。
动作执行前终止时不伪造当前步的 `simulator_step` 记录，应读取 `system_crash`
事件判断终止；动作执行后终止时最后一个实际步标记 `done=True`。

实现范围：hang 的无响应状态来自模拟运行时的挂起标记，检测器累计该状态，
不会仅凭位置不变、普通 `Wait` 或反复打印同一个导航动作判定 hang。
通信循环目前做空白归一化后的精确匹配，不调用 LLM 判断语义近似，也不覆盖
措辞变化的消息循环。它是随仿真步执行的监控代码，不是独立进程的墙钟超时器，
不能打断 Python/原生仿真/API 调用本身彻底卡死的情况。

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
轨迹与正常运行使用相同记录器，并直接写入 `fault_output/`，例如在上面的
Dropout 命令中加 `--run-label dropout_episode28_step40`，运行后得到：

```text
fault_output/dropout_episode28_step40_trajectory.jsonl
fault_output/dropout_episode28_step40_trace-episode_28.txt
fault_output/dropout_episode28_step40_events.jsonl
fault_output/dropout_episode28_step40_crash.json  # 仅检测到崩溃时生成
```

JSONL 每个仿真步包含动作、agent 和对象状态、奖励、结束标记等；TXT 每步包含
`Simulator_Step`、agent 位置/动作、对象位置及故障状态，与 clean 轨迹字段对齐。
故障运行保留 `mode=faulty` 和实际故障标记，不改写成正常数据。Hang 使用 `hang_`
前缀。默认标签为类型加时间戳，不再产生 `dropout_agent_dropout_` 这样的重复前缀。

辅助文件保存在 `fault_output/<run_label>/`：

- `plan.json`：参数、场景、episode、注入步、运行命令和 `outputs` 文件路径；
- `episodes.json.gz`：选中的评估数据子集。

`--output-dir` 修改轨迹输出目录，`--run-label` 指定文件标签及辅助文件子目录。
已有同名输出文件或运行目录不会覆盖；重复实验需更换标签。终端会显示 JSONL 和
每个 episode TXT 的完整路径。`--dry-run` 只显示这些预期路径，不生成轨迹内容。
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
`agent`、`episode_id`、`simulator_step_schedule`。轨迹开关及输出参数来自
`habitat_baselines.trajectory.enabled`、`habitat_baselines.output_dir` 和
`habitat_baselines.run_label`，新增故障无需单独定义。数据选择、抽样和计划保存
无需为每个错误重复实现。

```bash
PYTHONPATH=habitat-mas python -m unittest \
  habitat_mas.test.test_fault_launcher \
  habitat_mas.test.test_agent_hang_fault \
  habitat_mas.test.test_agent_hang_policy \
  habitat_mas.test.test_trajectory_replay \
  habitat_mas.test.test_agent_dropout_fault -v
```

这些测试包含原参数保留、冲突校验、抽样、数据子集、命令及策略分派验证。
完整 Hydra 配置组合和原生 Habitat 仿真需要安装相应依赖后验证。
