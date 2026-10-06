# Clean 成功子集上的配对 recovery 实验

入口是 `python scripts/recovery_batch.py {prepare,run,analyze}`，从 EMOS 根目录执行。
它调用现有 clean/故障启动器，不改变 recovery 算法。

## 固定协议

- 从原始 clean 日志的终局 `pddl_success=1` 筛选，使用 `(scene_id, episode_id)` 作身份。
- 固定筛选集，之后不根据重跑结果重新挑选。一步成功的任务保留在 manifest，标记为无非空回放前缀，不进入本轮故障试验。
- 在原 clean 长度 50% 处注入（向下取整，限定 `1 <= T < length`），分别掉线 `agent_0` 和 `agent_1`。
- dropout 与 dropout_recovery 回放同一份**原始成功轨迹**，需要完整 `recorded` 上下文和运行时 `replay_prefix_verified`。新 clean 对照独立重跑，不替换分叉来源。
- 三组均 `should_terminate_on_wait=False`、750 步预算、`habitat.seed=66`、仿真 seed=100；LLM 模型参数沿用当前项目代码。
- clean/dropout 关闭 watchdog 和 recovery；dropout_recovery 同时开启两者。比较的是检测加恢复整套机制。
- 第一轮每任务每条件一次：70 个 clean 对照、140 个 dropout、140 个 dropout_recovery，共 350 个进程。
- 主检验分别比较每个故障 agent 下的 recovery 与 dropout，两个检验用 Holm 校正。成功筛选的历史 clean 为100%是筛选结果，不是独立泛化指标。

## 准备实验

```bash
conda activate habitat
python scripts/recovery_batch.py prepare \
  --trace clean_output/clean_all_seed66_trajectory.jsonl \
  --source-config outputs/2026-09-24/10-07-14/.hydra/config.yaml \
  --output fault_output/recovery_clean_success_seed66_v3
```

已准备过就跳过这一步。prepare 拒绝覆盖目录。manifest 包含来源 SHA256、原始配置文件摘要、当前代码/配置摘要、冻结的数据集和每任务轨迹摘要。
不复制配置里的模型凭据，也不记录 API key。实际运行使用当前环境的模型服务；务必保持模型和服务设置一致。代码或冻结输入改变后需新建实验目录，不能继续混入旧结果。

`clean_success_subset.json.gz` 包含全部74个原始成功任务；`manifest.json` 明确列出4个短任务的排除原因。输入文件和输出包含模型对话上下文，应按现有实验日志管理。

## 已有 clean 结果：按阶段运行

已完成全量 clean 时，可直接在冻结的成功子集上运行 dropout：

```bash
python scripts/recovery_batch.py run \
  --manifest fault_output/recovery_clean_success_seed66_v3/manifest.json \
  --groups dropout
```

共70个任务，分别注入agent_0/agent_1，共140个进程。追加 `--limit-tasks 5` 可先跑10个故障条件。
之后用同一manifest将 `--groups dropout` 改为 `--groups dropout_recovery`，运行匹配的恢复条件。
历史clean作为筛选和回放来源，其成功率100%来自筛选；分析中的新clean对照会保持pending。
若需统一配置下的三组对照，再单独执行 `--groups clean`；dropout与recovery的配对比较不依赖新clean先完成。
不传 `--groups` 时仍运行全部三组。各阶段均会跳过已有的有效结果。

## 先检查命令，再运行5个任务的 pilot

```bash
python scripts/recovery_batch.py run \
  --manifest fault_output/recovery_clean_success_seed66_v3/manifest.json \
  --dry-run

python scripts/recovery_batch.py run \
  --manifest fault_output/recovery_clean_success_seed66_v3/manifest.json \
  --limit-tasks 5
```

dry-run 只生成 `commands.json`，不占用正式输出位置，不调用模型。
`--python /home/dyc/.conda/envs/habitat/bin/python` 可显式指定仿真解释器；默认使用运行本脚本的解释器。
真实运行需要模型环境已配置，例如进程可读取 `OPENAI_API_KEY`。不要将密钥写进命令行或 manifest。

pilot 是固定排序的前5个任务，共25个进程，用于检查流程，不能据其成功率调整正式筛选集。
每个进程默认最多1800秒，`--timeout` 可修改墙钟限制。正常仿真失败仍是有效结果；API错误、回放偏差、缺少终局、未按计划注入等会标记 error 并停止批次。
回放输入的离线检查只验证格式/上下文，物理状态一致性必须由真实运行确认。

## 续跑全量

```bash
python scripts/recovery_batch.py run \
  --manifest fault_output/recovery_clean_success_seed66_v3/manifest.json
```

完成的有效结果会被重新检查并跳过；完成时记录的输出SHA256也必须一致。失败尝试不会覆盖；解决基础设施问题后，用同一命令追加 `--retry-errors` 新建 attempt。重试不用于重跑有效的任务失败，避免挑选有利结果。
默认顺序是每任务 clean、agent_0 两组、agent_1 两组。每次只运行一个仿真进程，并加批次文件锁。SIGTERM/Ctrl-C会终止当前子进程组；异常终止后的存活PID会阻止重试，应先检查该旧进程。
终局必须来自正确场景/episode。退出码3只有真实 `system_crash` 终局且注入/回放有效时才算任务失败，不能只凭返回码判分。

## 分析

```bash
python scripts/recovery_batch.py analyze \
  --manifest fault_output/recovery_clean_success_seed66_v3/manifest.json
```

每次批量运行结束也会自动生成分析。输出位于 `analysis/`：

- `episodes.csv`：每个计划条件一行，包含 pending/error；有效运行有成功、等待、恢复事件和终止原因。
- `pairs.csv`：同任务/agent/repeat 的 clean、dropout、recovery 配对结果。
- `summary.json`：有效分母、成功率、救回与反向损失、成功率差、95% CI、精确 McNemar 和 Holm p 值。
- `REPORT.md`：简要可读报告；缺配对时明确为阶段性结果。

救回率 = dropout失败且recovery成功 / dropout失败。反向损失 = dropout成功且recovery失败；harm_rate的分母是dropout成功数。
`recovery_completed` 只说明重规划完成，最终成功严格读取终局 `pddl_success`。
未完成/基础设施错误不算任务失败，也不从计划分母中消失；统计同时报告预期对数和缺失对数。

等待统计仅在 `step >= 原始注入步` 的共同观察窗口内，成功终局步不计等待：

- `double_wait`：两个在线机器人实际动作均为wait，当前任务未成功。
- `offline_wait`：一个机器人掉线，另一机器人实际动作wait，当前任务未成功。
- 连续30步记为持续等待，同时保存总步数、最长连续长度、首次出现和首次持续段起点。

停滞导航不自动算wait；零动作或旧高层动作不能把掉线机器人算成在线等待。

CI 使用按任务配对的 percentile bootstrap（10000次，固定分析随机种子）；极小样本或全相同结果可能退化，不应过度解读。
`--repeats` 可在准备阶段增加重复，但当前推断明确只使用预先指定的repeat 0，其余重复保留在CSV供描述性分析，不把相关重复当独立样本。
两种故障agent也分层检验，不把140个条件当140个独立任务。

## 输出与边界

`runs/<task_condition>/attempt_001/` 保存 console.log、command.json、result.json、Hydra输出、TensorBoard和逐步轨迹；status.json 用于续跑。
现有策略的 `chat_history_output/` 仍使用其旧共享命名，可能被后续条件覆盖。配对分析和审计以各attempt隔离的trajectory JSONL中的 `replay_context` 和recovery事件为准。

本工具没有修改感知成功条件，也不把历史stage指标当作当前PDDL真值。当前 recovery 是dropout即时接管，不是双等待触发恢复；实验可评价它是否减少后续等待和失败。


## 不调用模型API的原生流程检查

```bash
PYTHONPATH=habitat-mas:habitat-lab:habitat-baselines python -m habitat_mas.test.native_recovery_batch_smoke \
  --source-manifest fault_output/recovery_clean_success_seed66_v3/manifest.json \
  --episode 295 --output fault_output/recovery_batch_native_check_new
```

需要可用GPU、Habitat环境和场景资源。该入口使用项目现有FixedModel，真实执行clean和两个agent的dropout/dropout_recovery共5个条件。
默认仅60步，用于验证流程，不能评价真实模型恢复能力。manifest、JSON汇总和Markdown报告均带validation-only标识，正常运行入口拒绝将这种实验续跑为真实模型实验。

2026-09-26原生验证：五个条件全部生成有效终局；四个故障条件的前缀均通过实际校验并完成注入；两个recovery条件均记录恢复完成事件。
agent_1 recovery最终仍失败，确认统计没有将recovery_completed误当成成功。报告位于`fault_output/recovery_batch_native_check/analysis/REPORT.md`。
