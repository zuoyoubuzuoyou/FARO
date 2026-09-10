# Agent Hang 与统一故障注入改动说明

## 1. 改动目标

本次改动在已有 Agent Dropout 故障注入能力上增加 Agent Hang，并提供统一的
错误注入命令行入口。改动同时修复 episode 边界状态、LLM 输出解析、PDDL 目标
校验和故障输出命名等问题，使正常运行、Dropout 和 Hang 实验可以使用一致的
配置、轨迹和审计方式。

## 2. Agent Hang

新增 `AgentHangFaultConfig`、`AgentHangFaultSpec` 和 `AgentHangState`。Hang 只按
从 0 开始的 simulator step 触发，触发后持续到当前 episode 结束。

目标 agent 挂起后具有以下行为：

- 不再调用执行阶段的高层 LLM 或低层策略；
- 不处理新收到的 agent 间消息；
- 冻结循环状态和上一动作；
- 每个 simulator step 向 Habitat 发送全零 no-op；
- 物理实体仍保留在仿真中，其他 agent 和仿真继续运行；
- 不参与“所有 agent 都在 wait”时的提前结束判断。

Hang 表示逻辑上仍存活但无响应，即 `logical_agent_active=True`、
`logical_agent_responsive=False`。这与 Dropout 的
`logical_agent_active=False` 区分开来。Hang 不会阻塞 Python 线程。

支持两种 episode 匹配方式：

- `episode_id` 配合单个 `simulator_step`；
- `simulator_step_schedule="episode@step;..."` 配置逐 episode 注入计划。

非空 `episode_id` 与 `simulator_step_schedule` 互斥。Hang 不支持
`high_level_decision` 触发，也不能与 Agent Dropout 同时启用。配置非法、目标
agent 不存在、没有触发任何故障或计划中的注入步未到达时，评估会明确报错。

## 3. 策略和评估器集成

`MultiLLMPolicy` 将原 Dropout 的零动作生成逻辑扩展为通用故障动作，并在每个
agent 执行前推进 Hang 状态。事件仅在触发步产生一次，活动状态持续记录到
episode 结束。

episode 初始化现在使用 mask 判断，而不是通过“上一动作是否全零”推断。这样
Hang 或 Dropout 持续产生零动作时不会错误地重新执行群体讨论。进入新 episode
时会清理：

- 高层决策计数；
- 已激活的 Dropout 事件；
- Hang 状态；
- CrabAgent 中未读的跨 agent 消息。

评估器统一选择当前启用的故障规格、事件记录器和轨迹命名，并支持同一数据集
episode ID 被重复评估时刷新文本上下文。环境暂停时同步过滤 simulator step
计数和高层策略的活动环境状态。

## 4. 统一故障注入命令行

新增入口：

```bash
python scripts/inject_fault.py [原 EMOS 参数] [错误注入简写参数]
```

当前支持：

- `--fault hang` / `--fault agent_hang`；
- `--fault dropout` / `--fault agent_dropout`；
- `--episode ID[,ID...]` 指定 episode；
- `--random-episodes N` 无放回抽取不同 episode ID；
- `--step N` 使用固定注入步；
- `--trajectory CLEAN.jsonl` 从完整正常轨迹中抽取实际执行步；
- `--scene` 消除跨场景重复 episode ID 的歧义；
- `--seed` 固定故障计划抽样；
- `--dry-run` 只生成计划，不启动评估；
- `--no-trajectory` 只保留故障事件。

命令行保留原有 `--config-name` 和 Hydra `key=value` 参数，不自动替换用户设置。
它会检查评估模式、环境数量、episode 次数、恢复配置和故障配置冲突，并拒绝
静默覆盖不兼容的参数。

每次运行会创建独立目录，包含：

- `plan.json`：参数、数据集、episode、场景、注入步和最终命令；
- `episodes.json.gz`：只含选中 episode 的数据子集；
- 故障事件 JSONL；
- 可选的逐步轨迹 JSONL 和可读文本轨迹。

已存在的运行目录不会被覆盖。离线 `--dry-run` 可在 Habitat 依赖不可用时通过
显式数据集路径和最大 episode 步数生成计划，但不会验证完整 Hydra 配置和 agent
列表。

## 5. 轨迹和输出命名

正常运行和不同故障使用实际运行类型作为文件名前缀：

```text
clean_<label>_trajectory.jsonl
dropout_<label>_events.jsonl
dropout_<label>_trajectory.jsonl
hang_<label>_events.jsonl
hang_<label>_trajectory.jsonl
<type>_<label>_trace-episode_<id>.txt
```

如果 `run_label` 已经包含相同类型前缀，不会重复添加。文件名组件会进行安全化
处理。

Hang 轨迹新增 `hang_state`、`agent_hang_noop` 和 `hang_noop` 状态。文本轨迹会
标记触发 simulator step、当时的决策步以及挂起起点。Dropout 原有轨迹格式继续
兼容。

## 6. LLM 与动作可靠性

OpenAI Chat Completions 的瞬态错误重试范围由 HTTP 429/503 扩展到 429/503/524。
429 遵循冷却时间，503/524 使用指数退避。

规划阶段如果模型返回未闭合的 Markdown code fence，会要求模型重新发送完整
响应，最多重试两次；仍不完整时返回原始文本，避免代码提取时发生索引越界。

CrabAgent 对 `nav_to_obj`、`nav_to_goal`、`pick` 和 `place` 的目标参数增加 PDDL
实体名校验。场景 asset 名不能直接作为动作目标。非法目标会要求 LLM 重新生成，
连续失败后回退到 `wait`，避免把无效动作传入 Habitat。

Scene Graph 对象描述中的条件分支也已修正，防止已解析出的 goal alias 被对象
full name 覆盖。

## 7. 测试覆盖

新增和更新的测试覆盖：

- Hang 配置校验、触发、永久状态和事件输出；
- Hang 策略跳过、状态冻结、step 0、重复 episode 和 episode 重置；
- Hang 与 Dropout 的互斥和 Dropout 回归；
- CLI fault 别名、episode 选择、scene 消歧、固定步和轨迹抽样；
- 数据子集、计划文件、Hydra 参数透传和冲突检测；
- dry-run 与子进程退出码传播；
- PDDL target 接受、拒绝、重试和 wait 回退；
- episode 间消息清理；
- 输出命名；
- HTTP 524 重试和未闭合 code fence 恢复。

相关运行文档：

- `AGENT_HANG_FAULT_MVP.md`；
- `AGENT_DROPOUT_FAULT_MVP.md`；
- `FAULT_INJECTION_CLI.md`；
- `CLEAN_EVALUATION_GUIDE.md`。

## 8. 当前限制

- 运行时故障策略仍按 EMOS 当前能力限定为单环境评估；
- Hang 仅支持 simulator step，不支持 decision step；
- Hang 和 Dropout 不能在同一次运行中同时启用；
- Hang/Dropout 不实现检测、任务重分配或恢复；
- 物理实体保留在仿真中，全零动作不保证其世界坐标绝对不变；
- 完整 Habitat 仿真仍需在安装了 Habitat、模型服务和数据集的环境中验证。
