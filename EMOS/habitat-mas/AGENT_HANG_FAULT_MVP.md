# Agent Hang 错误注入

推荐使用[统一错误注入命令行](FAULT_INJECTION_CLI.md)，通过 `--fault`、`--episode` / `--random-episodes` 和 `--step` / `--trajectory` 指定注入。

在执行阶段静默挂起指定 agent，持续到当前 episode 结束。挂起后不再调用该
agent 的高层 LLM 或低层策略，不处理新消息，冻结策略状态和上一动作，向
Habitat 发送全零 no-op。机器人实体仍在仿真中，其他 agent 和物理仿真继续运行。
全零动作不保证机器人世界坐标绝对不变，例如物理碰撞仍可能改变位置。

Hang 的逻辑状态为 `logical_agent_active=True`、
`logical_agent_responsive=False`；dropout 为 `logical_agent_active=False`。
Hang 保留成员身份和消息队列，发给它的请求正常入队但不会被读取或回复，也不向
队友发送退出通知。Dropout 则从可用成员列表移除，清空未读消息，新请求立即反馈
`agent_unavailable`；队友在下一次执行阶段 LLM 决策中看到成员退出信息。这个区别
由共享消息通道执行，不仅是日志标签。两者的状态都在本步所有 agent 执行之前生效。

两者都抑制目标 agent 的执行，但 hang 不参与“全部 agent 都在 wait”时的
提前结束判定：只要有 agent 挂起，就跳过该终止捷径，episode 仍可由环境自身
的成功、终止条件或步数上限结束。这里没有实际阻塞 Python 线程。
目前不实现 Hang 的超时检测或 episode 内恢复；消息入队不代表对方已确认。

初始群体讨论仍会完成；`simulator_step=0` 表示任务分配完成后，目标 agent
从未开始执行。下一 episode 清除挂起状态，并按配置重新判断是否注入。
默认关闭注入。当前实现与 EMOS 执行策略一致，仅支持一个环境；hang 和
dropout 不允许同时启用。

## 仿真步触发

在已配置好 Habitat 和模型服务的 EMOS 环境中运行：

```bash
env -u ALL_PROXY -u all_proxy python -u -m habitat_baselines.run \
  --config-name=multi_rearrange/llm_spot_drone_per.yaml \
  habitat_baselines.evaluate=True \
  habitat_baselines.num_environments=1 \
  habitat_baselines.test_episode_count=1 \
  habitat.environment.iterator_options.shuffle=False \
  habitat_baselines.fault_injection.structure.agent_hang.enabled=True \
  habitat_baselines.fault_injection.structure.agent_hang.agent=agent_1 \
  habitat_baselines.fault_injection.structure.agent_hang.simulator_step=40 \
  habitat_baselines.trajectory.enabled=True \
  habitat_baselines.output_dir=fault_output \
  habitat_baselines.run_label=hang_faulty
```

步数从 0 开始：目标 agent 正常执行 0–39 步，第 40 步动作之前开始挂起，
可以打断正在执行的 Navigate/Pick/Place 技能。默认 `episode_id=""` 匹配所有
被评估的 episode。指定 `agent_hang.episode_id=15` 仅匹配数据集 ID 15，
该配置不负责选择评估数据集，评估必须实际包含此 episode。

也支持逐 episode 调度，覆盖默认触发步数，只对列出的 episode 注入：

```bash
'habitat_baselines.fault_injection.structure.agent_hang.simulator_step_schedule="233@18;27@40;96@0"'
```

`simulator_step_schedule` 与非空 `episode_id` 互斥。Hang 仅支持仿真步触发，
无需设置 `trigger_domain`，不支持 `decision_step`。日志中的决策计数仅用于审计。

## 输出与验证

示例命令产生：

- `fault_output/hang_faulty_events.jsonl`：每个匹配 episode 一条
  `StructureFault/AgentHang` 注入事件，包含实际仿真步、决策步及永久持续语义。
- `fault_output/hang_faulty_trajectory.jsonl`：开启轨迹后记录每一步
  的动作、`hang_state`、机器人及对象状态。
- `fault_output/hang_faulty_trace-episode_<id>.txt`：可读轨迹，挂起期间显示
  `AgentHangNoOp[]`、`hang_noop` 和挂起起点。

未触发任何配置故障、调度中的 episode 未触发、目标 agent 不存在或配置非法
都会报错。若 episode 在触发步之前结束，不会伪造注入事件。

对照运行将故障的 `enabled=False`，保留公共的
`habitat_baselines.trajectory.enabled=True` 并改用不同的
`habitat_baselines.run_label`。同名输出文件会被覆盖。

无需 Habitat/PyTorch 的测试命令：

```bash
PYTHONPATH=habitat-mas python -m unittest \
  habitat_mas.test.test_agent_hang_fault \
  habitat_mas.test.test_agent_hang_policy \
  habitat_mas.test.test_agent_dropout_fault -v
```

策略测试执行真实分派方法，但使用轻量替身替代张量和单 agent 策略，截止于
联合动作张量拼接之前；它们不能替代完整 Habitat 仿真验证。
