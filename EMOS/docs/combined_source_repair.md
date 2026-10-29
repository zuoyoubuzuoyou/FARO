# 一次 LLM 调用完成检测与源码修复

启用 `recovery.mode=source`、`recovery.detector=llm` 后，默认在线路径为：

```text
任务＋近期观测＋当前原始源码
  → 一次 LLM：判断异常，并在可安全修复时输出代码
  → 宿主校验 → 安装原类／原文件 → 调用 Resume
  → 原 episode 继续执行 → 根据真实进展评估效果
```

总开关 `recovery.enabled` 仍默认关闭。独立离线检测器仍只返回布尔判断；扩展模式与显式注入的旧接口保留兼容路径。

## 代码输入与输出

固定 general instruction 与 `{problem}`、`{trajectory}`、`{code}` 等占位符的完整模板见[通用 Prompt 模板](llm_prompt_templates.md)。占位符对应现有 `task`、`runtime`、`code` 等 JSON 字段；每次只填充实际值，保持运行接口一致。trajectory 沿用最近采样窗口和事件保留规则，暂未新增历史事件检索或严格 token 截断。

在线流程统一使用 `recovery/source_generator.py` 中的 `GENERAL_INSTRUCTION` 与 `SYSTEM_PROMPT`，由 `detect_and_repair()` 一次调用判断系统运行异常，并在需要且可安全修复时直接返回代码。通用指令只有一份，不再拼接独立的检测角色和恢复角色。指令不预设EMOS、特定类／恢复函数名或第一轮修复步骤；具体可改范围、入口签名和恢复方式由输入中的 `code.edit_policy`、`available_api` 与源码元数据提供。它根据任务、状态和代码判断执行／调度／参数处理／异常处理问题，没有为某个 episode 写定恢复动作。

每次输入包括任务与能力约束、原规划、最近对话、当前及近期状态、真实异常信息、方法级源码、已有修复及实际反馈。本地仍在每次检查重新读取完整原文件；发送前提取所有现有可改方法、运行及异常栈相关方法、必要调度和反馈入口、一跳辅助方法，以及相关导入／继承／字段初始化上下文。选择规则集中在 `source_view.DEFAULT_SOURCE_VIEW_RULES`，可由适配器提供 `code.source_view_rules`，不按某个 episode 写定根因。

`code.source_files[path].content` 是带原行号的方法视图，完整方法只出现一次；`methods` 提供索引和片段哈希，`symbol_index` 列出未发送正文的符号。`sha256` 始终是完整原文件哈希，供 `base_sha256` 原样复制；`content_sha256` 和 `snippet_sha256` 仅标识视图片段。模型必须返回完整替换方法，宿主仍读取完整文件实施校验和安装。

`input_layout.runtime_compaction` 明确记录省略字段的来源：重复当前帧指向 `recent_frames` 的索引和步号，agent任务／分工／规划／历史引用顶层唯一副本，完全相同的物体、分工和固定相机配置移到 `runtime.shared_frame_fields`。每个采样步、事件、动态状态、False、null和缺失均保留；不推断采样间隔内的隐藏状态。`expand_runtime_context` 可在本地逐项还原原runtime。恢复入口收到的仍是原展开上下文，展示去重不改变其运行接口。

`code.edit_policy` 提供实际可修改范围，可读源码不等于可改源码。普通窗口的 `error_report` 为 null，执行异常则带真实异常和栈位置，不把旧检测结论当成当前答案。源码输入不包含 `.env`，请求及审计经过脱敏。

正常响应只有：

```json
{"system_error": false}
```

异常且有安全修复时，同一响应返回代码：

```json
{
  "system_error": true,
  "patches": [{
    "path": "允许的原始相对路径.py",
    "target": "Class.method",
    "base_sha256": "输入中该文件当前哈希",
    "code": "def method(self, context):\n    ..."
  }],
  "entrypoint": {"path": "父进程策略相对路径.py", "target": "PolicyClass.Resume"}
}
```

每个补丁是完整方法定义；宿主重建修改后的完整文件与 diff。模型不必额外返回理由、证据或置信度。发现异常但无法安全修复时，可返回 `{"system_error": true, "patches": [], "entrypoint": null}`，记录为 `unrepairable` 并保持代码不变。

## 自动应用与防作弊

真实在线生成器的提案使用 `emos-execution-source-v2`，在准备及安装两处执行同一校验，宿主固定该配置，省略或降级 manifest 不能绕过。

当前自动配置除参数解析和 Resume 外，也开放执行方法中的声明区域；以 `code.edit_policy` 的实际内容为准：

- `WaitSkillPolicy._is_skill_done`：保留实际计时和原等待时长的完成判断。
- `SkillPolicy.should_terminate`：高层中断信号的合并，保留超时、后置条件、诊断和动作逻辑。
- `OracleNavPolicy._is_skill_done`：对真实完成传感器增加控制时序门控，不能合成完成信号。
- `OracleNavAction.step`、`OracleNavCoordinateAction.step`、`OracleNavDiffBaseAction.step`：导航完成谓词、已到达分支中遗漏的完成标记，以及 DiffBase 旧完成缓存的抑制分支。原几何来源、配置阈值和所有运动控制语句保持；每条完成分支必须保留实际距离上界。
- `Resume`：除重新触发高层决策外，可以遍历现有技能，清零临时技能计数、清除上一轮运行标记；不能改目标参数或伪造完成。

模型仍返回完整方法代码，但校验器要求声明区域之外的 AST 与原方法一致。导航 worker 补丁经已有 RPC 安装到原类，已有 action 实例下次调用生效；安装本身不执行动作、不重建或 reset 环境。环境、任务、模拟器、传感器、物理和评分代码继续只读，其他方法不能由模型自行申请权限。

受限检查拒绝任务／评分／能力／真实观测和物理状态修改、额外步进或 reset、传送、外部 IO、动态执行、危险别名／计算调用及无界代码。导航判断不能恒真、伪造距离或放宽配置阈值，运动生成及调用位置保持。解析修复须来源于实际参数，不能硬编码目标答案、覆盖模型所选参数或移除既有解析审计。修改原方法保留签名和装饰器，Resume 在原实例上运行，保留模型、对话、持物与已完成工作。

这是一个严格限定的自动修复子集，**不是任意 Python 的安全沙箱，也不证明补丁的语义正确性**。超出可验证范围的提案拒绝或记录不可修复，不能以降低边界来强行修复。

## 时机、预算及 Resume

普通检查沿用每 50 步采样、10 点窗口、每新增 5 点检查；只有冷却结束且有修复预算时才联合请求。正常窗口不创建修复版本、不消耗修复迭代、不启动冷却。默认只尝试修复一轮（`max_iterations=1`），失败或不可修复也计入这一轮，不再生成第二版。应用后的进展评估仍在 250 步后进行，原任务继续执行。冷却间隔仍为 250 步；显式增加预算时才支持多轮修订。

策略／模型解析在下一次 `env.step` 前抛出异常时，可以立即检查，包括真实第 0 步初始状态。宿主恢复失败决策的逻辑检查点，安装有效补丁并调用 Resume，再重试同一次决策。已有模型、历史、消息及技能状态保持，真实 API 用量不倒扣，物理推进仍由原循环执行一次。异常上的正常或不可修复响应立即退出修复，让原异常传播，不无限重试。

安装或 Resume 失败时事务回退；回退冲突会停止后续修复，保护并发用户编辑。已运行栈帧不追溯改变，新方法在下次调用时生效。运行结束默认恢复原文件和方法，保留所有审计；显式设置 `persist_source_changes=True` 可保留源码版本。安装成功与观察到恢复进展分别记录。

无法返回的原生调用、进程崩溃、已终止并被 reset 的 episode 不在该回调的恢复范围内；仿真步抛错不自动重试，因为物理状态可能已部分改变。

## 开启运行

使用当前 `.env`／环境变量中的 `OPENAI_API_KEY` 和 `OPENAI_BASE_URL`：

```bash
PYTHONDONTWRITEBYTECODE=1 /home/dyc/.conda/envs/habitat/bin/python -u -m habitat_baselines.run \
  --config-name=multi_rearrange/llm_spot_drone_per.yaml \
  habitat_baselines.llm_model=gpt-6-sol \
  habitat_baselines.evaluate=True \
  habitat_baselines.num_environments=1 \
  habitat_baselines.eval.recovery.enabled=True \
  habitat_baselines.eval.recovery.mode=source \
  habitat_baselines.eval.recovery.detector=llm \
  habitat_baselines.eval.recovery.max_iterations=1 \
  habitat_baselines.eval.recovery.task=Perception \
  habitat_baselines.eval.recovery.model=gpt-6-sol \
  habitat_baselines.eval.trajectory_dir=analysis_results/recovery/runs
```

当前在线适配仍是 EMOS 单环境。通用提示词不意味着其他 benchmark 的日志与源码接口已经自动接入。`detector=team_progress` 使用启发式报警后生成源码的兼容路径；真实生成器的自动提案仍走受限校验。开发者显式注入、只提供 `generate()` 的旧生成器保留原事务接口，不是默认在线模型可选择的绕行方式。

## 审计与验证

`trajectory_dir/<run_id>/recovery/<evaluation_id>/` 保存联合请求 `source_requests.jsonl`、事件 `events.jsonl`、版本的 `context.json`／`generation.json`／失败响应、以及源码修改前后完整文件、`changes.diff` 和 manifest。

每次联合请求前先保存 `input_snapshots/*.full.json.gz`，其中是本地完整原上下文，正常判定和请求失败也保留；引用记录在审计或版本的 `input_snapshot`。完整快照不会嵌入模型输入。实际发给模型的内容以 `request.messages` 为准，`context.json` 和恢复入口仍使用展开结构。离线对比见 `analysis_results/recovery/method_input_20261028/README.md`。

终端同步显示检测结果、补丁应用或拒绝、后续进展反馈，并在修复预算耗尽时提示一次。错误摘要经过脱敏和长度限制，完整诊断以 `events.jsonl` 和各版本的 `version.json` 为准。预算耗尽后原任务继续执行，当前联合路径不再发起检测修复请求。`source_application` 只表示补丁安装和恢复入口调用成功，实际恢复效果仍需后续观测判断。

本地验证使用假的网络传输，检查同响应安装、正常不改代码、有限预算、真实初始状态异常、同实例 Resume、无额外物理步、源码回退和越界补丁拒绝：

```bash
PYTHONPATH=test:habitat-mas PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/emos-matplotlib \
  /home/dyc/.conda/envs/habitat/bin/python -m unittest discover -s test -p 'test_*recovery*.py' -v
```

这些检查证明实现链路及覆盖的边界；真实 LLM 在 Habitat 中的恢复成功率仍需实际运行验证。
