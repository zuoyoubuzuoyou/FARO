# LLM代码恢复与原地Resume（兼容扩展模式）

本文描述此前的`mode=extension`兼容实现。当前默认的原始源码修改、新增Resume与多轮源码修复见[原始源码恢复说明](source_code_recovery.md)。

此功能把在线运行异常检测接到EMOS真实执行循环：返回当前仿真状态 → 检测异常 → 整理固定上下文 → LLM生成Python恢复分支 → 校验 → 下一步执行恢复动作。默认关闭；开启后仍在同一仿真和episode中继续，保留机器人位姿、持物、模型实例、规划与对话历史。

## 实现边界

LLM生成完整的 `recover(context, memory)` 函数，用Python条件、计数和状态记忆控制恢复。每版代码作为运行中的恢复扩展安装，后续采样继续执行；再次检测到异常时，下一版生成会看到旧代码及实际反馈。宿主为原控制循环补充Resume、重新规划和技能调度入口。当前版本迭代的是这段恢复代码，不直接覆盖磁盘上的模拟器、控制器或模型调用源码，也不更新模型权重。

生成代码只使用三种命令：

| type | 必要字段 | 实际效果 |
| --- | --- | --- |
| `resume` | `agent`；可选`reason`、`instruction` | 中断陈旧技能，保留历史，在下次模型调用中说明中断原因，从真实当前状态选择新技能 |
| `replan` | `agent`、非空`instruction` | 在后续模型调用中加入明确的纠正要求，重新选择剩余工作所需的技能 |
| `action` | `agent`、`name`、`arguments` | 下一次高层决策直接选择已配置的技能，通过原有`on_enter/act`执行 |

工具和参数来自每个机器人实际可执行的Action schema与技能配置的交集；`known_entities`提供精确实体名称。整批命令在任何修改前校验；无效机器人、工具、参数类型以及实际技能PDDL域中不存在的物体/目标被拒绝。`robot`由宿主注入，生成代码不自行填写。不会通过清空`prev_actions`或重新初始化agent来Resume；有效的全零动作也不会被当成新episode。

## 固定Prompt位置

系统提示词在 `habitat-mas/habitat_mas/recovery/generator.py`；用户输入由 `context.py`统一构造。

| 固定字段 | 内容 |
| --- | --- |
| `schema_version` | `emos-code-recovery-v1` |
| `modification_instruction` | 明确修改要求，可在配置中覆盖 |
| `error_report` | 检测状态、原因、证据步号和置信度 |
| `task` | 当前任务、场景、职责、目标位置和能力信息；缺失项保持未知 |
| `planner_plan` | 每个agent实际生成的原规划 |
| `recent_history` | 每个agent最近10个完整对话轮，保留模型原消息结构 |
| `runtime` | 当前帧、最近采样帧、当前技能、可用工具及参数模式 |
| `code` | 固定源码白名单：仿真环境、执行循环、控制器、技能、agent、模型调用，以及当前生效的恢复代码 |
| `recovery` | 当前/下一版编号、已生成版本、执行结果、进展反馈、持久化记忆 |
| `available_api` | 函数签名、返回结构、三类命令和支持的Python子集 |

源码含路径与SHA256，超过每文件64000字符会明确标记截断。不会读取`.env`等密钥文件。输入和审计会脱敏；人工标签、故障注入标记、最终结果和未来状态不作为检测证据。记录的当前成功谓词及精确位姿仍属于模拟器特权观测。

例如模型可以生成以下恢复逻辑；这仅展示接口，运行时不会硬编码此策略：

```python
def recover(context, memory):
    commands = []
    frame = context["runtime"]["current_frame"]
    for agent, observation in frame["agents"].items():
        if observation is None:
            continue
        action = observation.get("action") or {}
        age = action.get("age") or 0
        already_resumed = memory.get(agent, False)
        if age > 100 and not already_resumed:
            commands.append({
                "type": "resume", "agent": agent,
                "reason": "当前动作长期无进展，检查剩余职责后选择新的动作。"
            })
            memory[agent] = True
    return {"commands": commands, "memory": memory}
```

代码先经过AST检查，再在独立Python子进程中执行。子进程不继承环境变量、模型客户端或仿真对象，限制CPU、内存、运行时间与结果大小，只开放JSON数据操作和固定内置函数。这是受限Python接口，不是任意Python程序的通用安全沙箱。

## 开启运行

沿用EMOS原来的环境与模型环境变量配置（`OPENAI_API_KEY`和可选`OPENAI_BASE_URL`）。从仓库目录运行，例如Perception：

```bash
PYTHONDONTWRITEBYTECODE=1 /home/dyc/.conda/envs/habitat/bin/python -u -m habitat_baselines.run \
  --config-name=multi_rearrange/llm_spot_drone_per.yaml \
  habitat_baselines.evaluate=True \
  habitat_baselines.num_environments=1 \
  habitat_baselines.eval.recovery.enabled=True \
  habitat_baselines.eval.recovery.mode=extension \
  habitat_baselines.eval.recovery.task=Perception \
  habitat_baselines.eval.recovery.model=gpt-6-sol \
  habitat_baselines.eval.trajectory_dir=analysis_results/recovery/runs
```

当前多机器人策略只支持单环境，恢复也明确限制为单环境。开启恢复会强制采集结构化轨迹与真实状态，即使`save_trajectories=False`；采集失败会明确报错。

默认仍用LLM检测器：每50个仿真步采样、10点窗口、每5点检查；recorder-v2从第1步开始，因此首查通常在451步。恢复代码最多生成3版，模型请求超时60秒，代码执行超时1秒；命令应用及生成之间有250步冷却，250步后记录进展反馈。用以下覆盖可提前检查：

```text
habitat_baselines.eval.recovery.sample_every=10
habitat_baselines.eval.recovery.cooldown_steps=50
habitat_baselines.eval.recovery.verify_after_steps=50
```

`recovery.detector=team_progress`可切换到原团队进展规则，恢复代码仍由LLM生成。更换任务时修改原任务配置与`recovery.task`（`Mobility`、`Perception`、`Rearrange`、`Manipulation`），核心恢复框架不按benchmark手写恢复分支。`task=auto`仅做保守的任务名称/文本识别；明确指定任务更可控。

## 审计、迭代与回退

结果位于 `trajectory_dir/<run_id>/recovery/<evaluation_id>/`：

- `events.jsonl`：检测、生成、应用、反馈、回退及终局跳过。
- `detection_requests.jsonl`、`detection_cache/`：LLM检测原请求、响应与缓存。
- `versions/001/context.json`：该次生成的完整固定结构上下文。
- `versions/001/generation.json`：模型原请求、响应、usage和代码；无效回复/接口错误保存在`generation_error.json`。
- `versions/001/recovery.py`：模型实际生成的完整恢复代码。
- `versions/001/candidate_result.json`、`version.json`：代码返回值、版本状态及后续反馈。

候选代码执行失败或命令无效时保留旧版；已安装代码在后续运行中失败时回退到上一版及其记忆，没有上一版则回到原循环。生成次数计入失败尝试。回退指恢复策略版本，不会倒退物理仿真或撤销已执行的动作。

反馈区分`applied`（已交给控制器）、`progress_observed`（目标距离下降或当前成功谓词新满足）、`state_changed`（仅位姿发生变化）和`no_observed_progress`。动作已下发、代码可执行或机器人移动，都不等于恢复成功。每个新episode使用新检测器、代码版本、记忆和预算。

当前恢复处理仍在运行的仿真。VectorEnv在`done=True`时会自动reset，终局返回后只记录反馈，不把旧episode恢复命令应用到新episode；进程崩溃、已经终止的仿真、底层技能抛出的异常也不能靠此返回后回调恢复。短于检测窗口的异常需要调小采样间隔/窗口。

## 验证

以下命令使用本地模型传输替身，不调用真实API或启动场景：

```bash
PYTHONDONTWRITEBYTECODE=1 /home/dyc/.conda/envs/habitat/bin/python \
  -m unittest discover -s test -p 'test_*recovery*.py' -v
```

其中`test_recovery_end_to_end.py`串起真实在线事件适配、LLM检测器、代码生成器、受限子进程和MultiLLMPolicy/CrabAgent恢复入口，检查下一动作改变、持物/历史保留及后续目标距离下降；其他测试覆盖实际层级技能切换、无效批次原子拒绝、代码超时、版本回退和episode隔离。测试通过证明机制接通，真实模型在Habitat场景中的恢复成功率还需要独立实验。

2026-10-23本地联合回归通过201项：恢复41项、原检测97项、轨迹63项。真实Perception配置的Hydra解析与恢复开关覆盖也已验证；未启动真实场景或调用真实模型。
