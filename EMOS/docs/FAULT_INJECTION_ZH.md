# Habitat 四类潜在执行故障注入

四种故障共用 `scripts/inject_fault.py`，作用于正常运行的 agent。
agent_dropout 和 agent_hang 的原有命令仍可使用。

| --fault | 类别 | 注入效果 | 必填参数 |
|---|---|---|---|
| action_noop | AF-1 | 策略继续运行，指定 agent 的导航命令被屏蔽 | 无 |
| wrong_subgoal | PL-1 | 替换子任务中的目标实体，重新提供规划输入 | --object、--replacement-object |
| message_delay | CF-1 | 发给指定 agent 的消息延后投递 | --delay-steps |
| object_localization | PF-1 | 指定 agent 使用带偏移的物体位置进行语义判断和 oracle 导航 | --object、--position-offset |

## 运行示例

已激活 `habitat` 环境后，在 EMOS 目录运行以下命令。命令前的 `env -u`
仅清除本次运行及其子进程的代理环境变量，不修改当前终端的代理设置。

Spot 的导航在第 20–49 步失效，第 50 步恢复正常下发：

```bash
env -u ALL_PROXY -u all_proxy \
    -u HTTP_PROXY -u http_proxy \
    -u HTTPS_PROXY -u https_proxy \
  python scripts/inject_fault.py --fault action_noop --agent agent_0 \
  --episode 295 --scene v3_sc3_staging_07.scene_instance --step 20 \
  --duration-steps 30 --run-label noop_case \
  habitat_baselines.evaluate=True \
  habitat_baselines.num_environments=1 \
  habitat_baselines.test_episode_count=1 \
  habitat_baselines.eval.evals_per_ep=1 \
  habitat_baselines.load_resume_state_config=False \
  habitat.dataset.should_terminate_on_wait=False \
  'habitat_baselines.eval.video_option=[]'
```

将 Spot 子任务中的目标 0 改成目标 1，修改从第 20 步持续至任务结束：

```bash
env -u ALL_PROXY -u all_proxy \
    -u HTTP_PROXY -u http_proxy \
    -u HTTPS_PROXY -u https_proxy \
  python scripts/inject_fault.py --fault wrong_subgoal --agent agent_0 \
  --episode 295 --scene v3_sc3_staging_07.scene_instance --step 20 \
  --object 'any_targets|0' --replacement-object 'any_targets|1' \
  --run-label subgoal_case \
  habitat_baselines.evaluate=True \
  habitat_baselines.num_environments=1 \
  habitat_baselines.test_episode_count=1 \
  habitat_baselines.eval.evals_per_ep=1 \
  habitat_baselines.load_resume_state_config=False \
  habitat.dataset.should_terminate_on_wait=False \
  'habitat_baselines.eval.video_option=[]'
```

第 20–49 步发给无人机的消息，每条延迟 10 个仿真步投递。窗口结束时仍排队的消息，继续按原定时间投递：

```bash
env -u ALL_PROXY -u all_proxy \
    -u HTTP_PROXY -u http_proxy \
    -u HTTPS_PROXY -u https_proxy \
  python scripts/inject_fault.py --fault message_delay --agent agent_1 \
  --episode 295 --scene v3_sc3_staging_07.scene_instance --step 20 \
  --duration-steps 30 --delay-steps 10 --run-label delay_case \
  habitat_baselines.evaluate=True \
  habitat_baselines.num_environments=1 \
  habitat_baselines.test_episode_count=1 \
  habitat_baselines.eval.evals_per_ep=1 \
  habitat_baselines.load_resume_state_config=False \
  habitat.dataset.should_terminate_on_wait=False \
  'habitat_baselines.eval.video_option=[]'
```

无人机对目标 1 的位置估计增加世界坐标偏移 `[1, 0, -1]` 米，持续 30 步：

```bash
env -u ALL_PROXY -u all_proxy \
    -u HTTP_PROXY -u http_proxy \
    -u HTTPS_PROXY -u https_proxy \
  python scripts/inject_fault.py --fault object_localization --agent agent_1 \
  --episode 295 --scene v3_sc3_staging_07.scene_instance --step 20 \
  --object 'any_targets|1' --position-offset 1 0 -1 --duration-steps 30 \
  --run-label localization_case \
  habitat_baselines.evaluate=True \
  habitat_baselines.num_environments=1 \
  habitat_baselines.test_episode_count=1 \
  habitat_baselines.eval.evals_per_ep=1 \
  habitat_baselines.load_resume_state_config=False \
  habitat.dataset.should_terminate_on_wait=False \
  'habitat_baselines.eval.video_option=[]'
```

真实模型运行使用项目既有模型配置与凭据。原生测试模型的验证命令见下文。
每次使用不同的 `--run-label`，以保护已有产物。`--scene` 使用完整场景路径或
文件 stem，例如 `v3_sc3_staging_07.scene_instance`，可消除重复 episode ID 的歧义。

## 参数语义

- `--step` 从 0 编号，表示该步执行前进入故障窗口。
- `--duration-steps N` 为半开窗口 `[step, step+N)`；省略或设为 `-1` 表示持续至 episode 结束。wrong_subgoal 只支持持续至结束。
- `--agent` 对消息延迟表示接收者，其他故障表示被注入的执行者。
- `--object` 与 `--replacement-object` 使用完整 PDDL 实体名，必须加引号保护 `|`。实体必须存在；错误子目标的原实体必须在该 agent 的实际子任务中。
- `--position-offset DX DY DZ` 单位为米，Y 为竖直方向。偏移不得为零或包含 NaN/无穷大。
- 单次运行只启用一种故障；其他结构故障和潜在故障自动关闭。
- 保留 `--random-episodes`、`--trajectory CLEAN.jsonl`、`--seed` 与 `--dry-run`。随机采样的故障步可能在更短的重跑结束之后，运行会明确报告未触发。

`--dry-run` 写出所选数据集和 plan.json，验证配置并打印正式命令。
对完整 clean 轨迹，添加 `--replay-trajectory CLEAN.jsonl --step N` 可回放并
校验故障前前缀，到 N 步切换为带故障的实时策略。前缀发生偏差会终止运行。

## 产物与如何判断生效

默认输出目录为 fault_output。计划在 `<run-label>/plan.json` 中，记录配置、
所选场景/episode、故障步、完整生效参数和正式命令；JSONL、TXT 和 summary
使用 `<fault>_<run-label>` 前缀。

- `_events.jsonl`：fault_injected 为窗口触发；fault_applied 才说明作用到实际输入/执行；有限窗口有 fault_expired。
- `_trajectory.jsonl`：每步真实 joint_env_action、模型决策、物理状态、真实任务指标及 fault_events。
- `_trace-episode_N.txt`：可读轨迹，包含故障事件及原始/修改后的动作或任务。
- `_summary.json`：逐 episode 的 triggered、effective、applications 和 ineffective_reason。

AF-1 的日志同时记录策略原始指令和被屏蔽后的指令。正常 wait 不计为一次
有效屏蔽。PF-1 记录导航实际消费的真实/估计位置，偏移不改变物体真实位置
或成功条件；窗口切换会清理目标 agent 的导航缓存。CF-1 的 message_queued、
message_delivered、message_dropped 使用相同 message_id，记录发送与投递步；
发送成功只表示传输接受，不表示接收者已处理或回应。

PF-1 到期后记录 `localization_target_restored`，证明导航重新读取真实位置；
后续模型决策也收到恢复后的坐标描述。清理缓存包括已完成导航的锁存状态。
发生策略/API/环境异常时，保留原异常，并保存已缓冲的故障证据、待投递消息
的丢弃记录和带 `termination_reason`、`error_type` 的汇总。

只有触发、没有有效导航命令或消息经过窗口时，effective 为 false；不能把
这类运行当成有效故障样本。故障生效也不意味着任务必然失败，例如另一台
机器人可能恰好完成了目标。最终成功仍读取真实 pddl_success。

这是错误注入层。现有掉线检测与任务重分配仍保留；新的无进展检测、
潜在错误诊断和恢复策略需要另行研究。

## 原生 Habitat 验证

验证脚本使用本地场景和固定测试模型，无需 API key，不调用外部模型。
它检查两种机器人导航的屏蔽/恢复和位置偏移、实际错误规划输入、消息
延迟前后的接收者模型输入、四类故障的 clean 前缀校验，以及重复 episode
重置和异常退出时的证据保存：

```bash
env -u ALL_PROXY -u all_proxy \
    -u HTTP_PROXY -u http_proxy \
    -u HTTPS_PROXY -u https_proxy \
  python scripts/validate_latent_faults.py \
  --output-dir fault_output/latent_native_validation --replay
```

可用 `--case noop_0`、`noop_1`、`planning`、`delay`、`localization_0`、
`localization_1` 或 `clean` 单独检查。输出目录应使用新目录，避免覆盖证据。
只有全部断言通过才写出 verification.json；每个 case 的 native.log 保存
实际 Habitat 运行输出。这些验证证明注入链路，不能证明真实模型的恢复效果。

`native_command.json` 保存实际测试命令，`model_observations.jsonl` 保存固定
模型收到的输入。测试模块会在消息送达和定位边界主动请求高层决策，以检查
模型输入的变化；正式运行继续使用项目正常的技能完成/决策机制。重复
episode 案例直接使用 evaluator 的重复评估配置，单次注入 CLI 仍要求
`evals_per_ep=1`。

2026-10-07 验证结果：185 项相关 unittest 全部通过；14 个原生案例全部通过，
证据目录为 `fault_output/latent_native_validation_20261007_final/`，汇总见
`verification.json`。原生案例包含 clean、两台机器人各自的 AF-1/PF-1、PL-1、
CF-1、四种前缀回放、消息/定位的同 ID episode 重置、实际策略异常退出。

CF-1 的回放后缀没有新的发送动作，因此该回放案例正确报告 `effective=false`；
独立 CF-1 原生案例验证了真实排队、延迟投递以及接收者模型输入的变化。
所有 Important 审查问题已通过复现测试修正；编译及 `git diff --check` 通过。
