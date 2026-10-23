# 四任务系统运行异常检测

检测对象是**一条轨迹中是否真实出现过系统级运行异常**。不是最终任务是否失败，也不是失败的主要原因。两种检测器均用于四个任务的完整episode：

| 任务 | 批次 | 完整episode | 原生成功／失败 |
| --- | --- | ---: | --- |
| Rearrange | 20261011_215201 | 49 | 11／38 |
| Mobility | 20261012_152653 | 99 | 47／52 |
| Perception（HSSD） | 20261013_162242 | 103 | 69／34 |
| Manipulation | 20261015_092413＋retry3 | 139 | 66／73 |
| 总计 | | 390 | 193／197 |

数据配置为[configs/detection/emos.json](../configs/detection/emos.json)。从已有manual_review中的逐集原始JSONL读取，不修改旧分析、原始轨迹、模型或控制策略。不完整的Perception200、Manipulation192及已被完整版本替代的主批318不重复计入。

## 两类检测方法

`team_progress`是主要启发式方法：联合任务当前条件、机器人职责、位姿／朝向、持物、物体变化、导航距离、实时检测和动作持续时间判断。持续导航时仍在移动、已完成职责后等队友、队友仍在正常取放等情形不会仅凭动作名称报警。职责的文字解析和位置条件是保守近似，规则仍会漏报或误报，需要独立评估。历史轨迹没有完整机械臂关节状态，复位异常不被解释为机械损坏。

`repeat_action`额外保留用户例子的简单基线：同一机器人的动作名连续超过10个观测点就报警。它故意不具备团队理解，可用于观察“只数动作名称”的误报代价；主表启发式方法使用team_progress。

`llm`使用独立的LLM客户端，固定结构化Prompt包含任务、场景、当前分工、最近窗口状态与期间新增的协作消息和反馈。只读取当前及过去的信息；不输入人工标签、最终结果、旧检测器判断或注入故障标记。输出system_error、normal或uncertain；报警引用真实观测步号。工具Success占位不会被当作物理成功。

证据步号可以是采样帧步号，也可以是当前窗口中明确保存的决策／消息步号。例如，第350帧输入中包含第321步决策时，321属于已提供的观测；校验器仍拒绝输入中不存在的步号、未来事件和非整数步号。通过格式校验不代表判断内容正确，仍需独立人工标注评估。

有效的uncertain表示未报警，若此集查询执行完整，则按阴性预测参与评估，可能构成漏报。API异常、坏JSON、预算耗尽等属于预测未完成，不伪装成正常运行。

## 时间单位与参数

默认配置为`sample_every=50, window=10, check_every=5`：每50个仿真帧采样，最近10个采样点组成窗口，每新增5个采样点检查一次。旧schema1从0步开始，因此首查窗口为0、50、…、450，随后每250步检查。期间所有新决策和消息会累积到下一个采样点，不会因采样丢掉求助内容。

这些是预先固定的工程起始参数，尚未根据人工标签调优。把`--sample-every`设为1，就得到用户举例的“每5个仿真帧检查过去10帧”；这会显著增加LLM调用次数。启发式还保留累计无进度时间，避免10点短窗口永远达不到100步停滞阈值。不要把高层模型决策轮次与仿真帧混用。

短于窗口的轨迹不在终局强行补查，查询数记为0；这种配置仍可能漏掉短轨迹的真实异常。每种方法首次报警后停止该方法后续检测，因episode级“是否报警”已经确定。

## 运行

使用已有Habitat环境，无需重跑仿真：

```bash
PYTHONDONTWRITEBYTECODE=1 /home/dyc/.conda/envs/habitat/bin/python \
  scripts/evaluate_system_errors.py \
  --methods repeat_action team_progress \
  --output analysis_results/detection/emos
```

LLM使用现有的OpenAI兼容配置。先明确授权把任务和轨迹窗口发送到所配置的服务，再运行；密钥来自环境变量或仓库.env，不写入输出。首次可用`--limit 1`对每个任务各取一集验证接口：

```bash
PYTHONDONTWRITEBYTECODE=1 /home/dyc/.conda/envs/habitat/bin/python \
  scripts/evaluate_system_errors.py \
  --methods repeat_action team_progress llm \
  --model gpt-6-sol --workers 4 \
  --output analysis_results/detection/emos
```

`--max-llm-calls N`限制每个episode的未缓存请求数。设为0可离线验证整个流程，LLM预算不足的记录会明确标为未完成，不能据此报告真实LLM实验完成。可按`--dataset mobility`等筛选任务。实际输入、原始响应、模型版本、token和错误保存在llm_requests；相同模型、Prompt和输入可复用cache，避免重复付费。

使用SOCKS代理的环境还需安装`httpx[socks]`。本次已将缺失的`socksio==1.0.0`隔离安装到`/tmp/emos_detection_deps`；在当前机器运行时可在命令前加`PYTHONPATH=/tmp/emos_detection_deps`，原有代理设置保持不变。若接口返回`401 Invalid token`，先核对本地`.env`中的密钥是否有效且对应所配置的接口，验证通过后再批量运行。

## 独立人工标注

首次运行生成`manual_labels.csv`，390条运行异常标签初始全部为空，包括成功轨迹。成功不等于没发生异常，其他失败主因也不等于没出现系统停滞。不要把旧AI分类或最终失败归因复制成运行异常真值。

先独立查看各集原始记录／中英MD，暂时不参考检测器报警，再填写：

| 字段 | 如何填写 |
| --- | --- |
| runtime_system_error | 1＝存在真实系统运行异常；0＝复核后未发现；无法确认则留空或unknown |
| onset_step／offset_step | 可选，异常开始／结束的原仿真步号 |
| reason | 支持判断的实际状态、职责及证据步号 |
| reviewer | 人工复核者，填写0或1时必填 |

判断应结合团队目标：正常长导航、到位后等待、错误规划被正常执行，不自动算系统错误；如果执行、终止、反馈或通信机制持续使必要工作无法推进，可标异常。短暂异常后自行恢复、最终成功的episode仍可标1。缺少证据时保持未知，不强迫二选一。

填写完成后，用相同命令并加`--labels`重新计算。现有人工表不会被检测结果覆盖，扩展评估范围时只追加新episode的空白行。

```bash
PYTHONDONTWRITEBYTECODE=1 /home/dyc/.conda/envs/habitat/bin/python \
  scripts/evaluate_system_errors.py \
  --methods repeat_action team_progress llm \
  --labels analysis_results/detection/emos/manual_labels.csv \
  --output analysis_results/detection/emos
```

## 指标与输出

一条轨迹多次报警仍只算一条检出。人工为1且报警计TP，人工为0但报警计FP，人工为1且未报警计FN，人工为0且未报警计TN。

`Precision=TP/(TP+FP)`，`Recall=TP/(TP+FN)`。程序以0到1保存比例，展示时可转百分比。用户的假设例子25条真异常、20条报警、其中15条正确，对应TP=15、FP=5、FN=10，因此P=0.75、R=0.60；这只是公式测试，不是实验结果。

主表包含成功与失败全部episode，另存仅失败集副表，两者分母不混用。未知人工标签不计入混淆矩阵；全体人工标注未完成，或某方法查询覆盖不完整时，正式P/R为空并注明原因。部分已核查集合的计数仍可审计。未标注时TP/FP/FN/TN为0只表示尚未评分，并不表示没有误报或漏报。零分母不编造0%或100%。

- `main_table_excel.csv`：四任务×检测方法，TP、FP、FN、TN和P/R状态。
- `failed_episodes_only_table_excel.csv`：仅失败episode的独立范围。
- `predictions.json`、`predictions_excel.csv`：逐集是否报警、首报步号、实际查询与完整性。
- `alarms.jsonl`、`episodes/`：具体报警及证据。
- `manual_labels.csv`：独立人工运行异常标注表。
- `metrics.json`、`run_config.json`：指标、参数、样本范围。
- `llm_requests/`、`cache/`：真实LLM审计及可恢复缓存。

CSV使用UTF-8 BOM，便于Excel打开中文。当前native成功指标和精确位姿属于模拟器特权观测，本版不是只靠机器人自带视觉的检测实验。当前成功谓词可用，但历史锁存stage、最终结果和原有故障检测输出均不能进入检测器输入。

2026-10-20真实`gpt-6-sol`回放已完成，结果见[本轮说明](../analysis_results/detection/emos_runtime_gpt6sol_20261020/README.md)。额外导出了四行`table1_excel.csv`和390行中文逐集预测表。390条逐集预测完整，其中262条触发LLM、128条短于窗口未查询；正式P/R仍待独立人工标签。主目录的累计请求审计与逐集最近一次结果的请求计数分别记录，避免补跑后混用数字。

单元／集成测试不访问真实模型：

```bash
PYTHONDONTWRITEBYTECODE=1 /home/dyc/.conda/envs/habitat/bin/python \
  -m unittest discover -s test -p 'test_detection*.py' -v
```
