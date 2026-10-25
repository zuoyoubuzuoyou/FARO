# 修改原始源码的LLM恢复

本模式实现“系统异常 → 固定上下文 → LLM修改原执行代码 → 安装修改 → 原地继续 → 根据反馈继续改代码”。`recovery.mode=source`是现在的默认模式；总开关`enabled`仍默认关闭。此前的`recover(context, memory)`扩展保留为显式的`mode=extension`兼容模式。

## 实际修改什么

LLM可以给原`MultiLLMPolicy`新增此前不存在的Resume方法，也可以修改原`act`、`CrabAgent`、`OpenAIModel`、层级控制器、具体导航/抓放/等待技能，以及仿真worker中的允许方法。修改会写入原`.py`文件，并安装到**原有类对象**上；已经创建的策略、agent、模型和仿真对象继续使用它们，不通过重新创建对象加载修改。

补丁生成协议是：

```json
{
  "patches": [
    {
      "path": "habitat-baselines/habitat_baselines/rl/multi_agent/multi_llm_policy.py",
      "target": "MultiLLMPolicy.Resume",
      "base_sha256": "该文件当前完整SHA256",
      "code": "def Resume(self, context):\n    for child in self._active_policies:\n        child._cur_call_high_level[0] = True\n    return {\"resumed\": True}\n"
    }
  ],
  "entrypoint": {
    "path": "habitat-baselines/habitat_baselines/rl/multi_agent/multi_llm_policy.py",
    "target": "MultiLLMPolicy.Resume"
  },
  "explanation": "新增原类的Resume处理；下一步从当前状态重新选择技能"
}
```

这只是协议示例，恢复分支由模型结合实际代码、任务和错误生成。`entrypoint`必须是本轮新增或修改的父策略方法，签名为`(self, context)`；宿主安装后会真正调用它。返回值只能是`None`或有限JSON数据，最大256KB。它不需要返回预定义动作列表，也不依赖旧版`apply_recovery_commands`入口。

下一轮可以继续修改该Resume方法，同时修复原模型响应解析方法、控制分支或具体技能。`base_sha256`和上下文源码每轮重新读取，因此下一轮看到的是已经修改过的当前代码。前轮失败补丁、错误信息、执行结果和可见进展也会输入下一轮。

## 固定Prompt

完整系统提示词在`habitat-mas/habitat_mas/recovery/source_generator.py`。

| 固定位置 | 输入 |
| --- | --- |
| `schema_version` | `emos-source-recovery-v1` |
| `modification_instruction` | 用户明确的代码修改/恢复要求 |
| `error_report` | 检测报告，或原执行异常的类型、消息、最近10个栈帧 |
| `task` | 任务、场景、职责、目标、能力信息 |
| `planner_plan` | 各agent实际已经生成的规划 |
| `recent_history` | 各agent最近10个完整对话轮 |
| `runtime` | 当前和近期真实状态、工具/实体、当前对象的实际类绑定、允许的源码目标 |
| `code.source_files` | 原模拟、控制、模型调用文件的完整内容、路径、SHA256、模块、类、parent/worker角色 |
| `recovery` | 已有源码版本、失败/应用结果、进展反馈及当前生效版本 |
| `available_api` | 源码补丁格式、生效方式、调用约束 |

人工标签、未来结果和故障注入标记不进入检测证据。任务文字、历史和源码作为数据处理，系统规则不由它们覆盖。模型调用使用现有OpenAI兼容环境变量；源码快照和审计会脱敏，不读取`.env`作为代码输入。

## 同一仿真怎样继续

在线路径有两个入口：

1. 正常返回一个仿真步后，检测器判断异常；此时安装源码并调用生成的Resume，下一次决策使用新方法。
2. 模型解析、策略或技能在决策期间抛出异常，尚未调用下一次`env.step`：恢复失败决策前的逻辑检查点，生成/安装源码，调用新增Resume，再重试这次决策。

第二种情况在第一步决策失败时也适用，使用真实初始状态，不伪造已完成的仿真步。多机器人中前一个机器人已经产生的部分决策、历史、消息、技能计数和循环张量会在重试前恢复；原模型和agent对象身份保持，已发生的API token用量不会被倒扣。`env.step`在重试边界外，不会因一次决策重试而重复物理推进。

worker补丁通过已有`VectorEnv.call_at`通道安装在原worker中的原类上。新增的安装、检查、回退RPC捕获异常并返回结构化错误，补丁错误不会通过该RPC直接终止worker。parent和worker只接受在目标进程已加载的类，避免晚导入模块留下无法回退的方法。

## 校验与回退

先校验整批文件哈希、目标、完整方法签名和Python语法，再安装方法和提交原文件。现有类、参数默认对象、装饰器、`super()`及类私有名称绑定会保留。可新增方法；替换既有方法必须保留签名。原始文件与实例身份不匹配、重复目标、路径越界、导入/顶层执行、评分或恢复框架修改会被拒绝。

方法变化在下一次调用时生效，已经运行的Python栈帧不会被追溯修改。原评估循环已加入固定的决策重试边界，LLM可修改其后调用的原方法；不允许把正在执行的整个`_evaluate_agent`作为补丁目标。

候选安装或Resume执行失败时，回退源码、类方法与即时逻辑变更。后续执行报错会先撤销本控制器上一版修改，再结合失败版本生成下一版。首个决策异常中的坏补丁也会在剩余预算内继续修订，因为此时没有后续仿真帧可以等待。

回退前检查全部文件、parent及worker。若发现并发用户编辑或代码冲突，保留用户修改并停止进一步生成/安装；不会把部分回退当成普通拒绝后继续执行。回退源码不倒退已经完成的物理推进。

默认运行结束时恢复本轮改过的原文件和方法，所有版本留在审计目录。要保留生成的源码供后续运行使用，显式设置：

```text
habitat_baselines.eval.recovery.persist_source_changes=True
```

## 运行

使用原EMOS环境与`OPENAI_API_KEY`、可选`OPENAI_BASE_URL`。`EMOS_LLM_MODEL`配置原任务讨论、规划和执行模型；`recovery.model`单独配置恢复代码生成模型。下面将两者都设为`gpt-6-sol`：

```bash
EMOS_LLM_MODEL=gpt-6-sol PYTHONDONTWRITEBYTECODE=1 /home/dyc/.conda/envs/habitat/bin/python -u -m habitat_baselines.run \
  --config-name=multi_rearrange/llm_spot_drone_per.yaml \
  habitat_baselines.evaluate=True \
  habitat_baselines.num_environments=1 \
  habitat_baselines.eval.recovery.enabled=True \
  habitat_baselines.eval.recovery.mode=source \
  habitat_baselines.eval.recovery.task=Perception \
  habitat_baselines.eval.recovery.model=gpt-6-sol \
  habitat_baselines.eval.trajectory_dir=analysis_results/recovery/runs
```

更换benchmark时修改原任务配置和`recovery.task`：`Perception`、`Mobility`、`Rearrange`或`Manipulation`。核心框架不为每个benchmark硬编码恢复方案。当前EMOS多机器人策略及恢复均限制为单环境。

默认每50步采样、10点窗口、每5点检测；`recorder-v2`首查通常是451步。源码修改最多尝试3版（拒绝也计数），决策最多重试3次，普通报警的生成/应用间隔为250步，250步后检查可见进展。执行异常无需等待冷却，但仍受版本预算限制。相关参数为`sample_every`、`window`、`check_every`、`max_iterations`、`max_policy_retries`、`cooldown_steps`、`verify_after_steps`。

`recovery.detector=team_progress`可改用团队进展规则，源码仍由LLM生成。源码方法在原进程执行，旧的`execution_timeout`只限制兼容扩展模式的独立Python进程，不能抢占源码模式中的死循环或长时间原生调用。静态检查与回退不构成任意Python安全沙箱。

进程崩溃、无法返回的物理步、或`done=True`后已被VectorEnv自动重置的episode，不在本机制的原地Resume范围内。仿真步抛错也不会自动重试，因为物理状态可能已经部分改变。新代码仍需真实任务实验验证，安装成功不等于恢复成功。

## 结果位置与验证

`trajectory_dir/<run_id>/recovery/<evaluation_id>/`保存：

- `events.jsonl`：检测、源码生成、应用、进展、异常、回退和终局处理。
- `versions/NNN/context.json`、`generation.json`：真实生成上下文、原请求/响应、代码及用量。
- `versions/NNN/version.json`：补丁、entrypoint、transaction_id、版本状态与执行反馈。
- `source/<transaction_id>/before/`、`after/`：本轮原始文件与修改后完整文件。
- `source/<transaction_id>/changes.diff`、`manifest.json`：可审查差异、哈希和parent/worker清单。

本地机制验证不调用真实API或加载场景：

```bash
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/emos-matplotlib \
  /home/dyc/.conda/envs/habitat/bin/python \
  -m unittest discover -s test -p 'test_*recovery*.py' -v
```

`test_recovery_source_integration.py`使用真实`OpenAIModel`与`MultiLLMPolicy`，只替换网络传输：先让原解析器失败，再实际修改原模型文件和方法、新增并调用原策略Resume，检查同一模型/历史保留、失败轮撤销、两次响应token均保留、结束后原文件逐字恢复。`test_recovery_source_worker.py`在独立Python进程中通过真实`RLTaskEnv`补丁RPC安装、执行并回退worker原控制方法，检查对象身份、episode和持物不变。这些验证证明实现链路，尚未测量真实模型和Habitat场景中的恢复成功率。

2026-10-23联合回归通过275项（恢复、原检测、轨迹、trace schema和run config），真实Perception任务的Hydra源码模式配置解析通过。测试没有调用真实API或加载Habitat场景。
