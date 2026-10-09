# Habitat 四类潜在执行故障注入设计

日期：2026-10-07。状态：用户已确认“按设计继续实现”；实现与验证完成。

## 目标与范围

用户已选择先实现此前推荐的四类：AF-1 Action No-op、PL-1 Goal
Formulation Error（错误子目标）、CF-1 Message Delivery Error（消息延迟）、
PF-1 Object Localization Error。依据 EMOS/研究内容.md，agent 保持在线，
故障作用于正常执行链路，而非复用挂起逻辑跳过策略调用。

交付包括真实 Habitat 执行接入、统一 CLI、可控参数、事件与轨迹证据、
可运行示例以及原生仿真验证。这次任务是错误注入；新的自动检测和恢复算法
不纳入范围。现有 dropout/hang、clean 记录和前缀回放需要继续兼容。

当前运行配置为 multi_rearrange/llm_spot_drone_per.yaml，Spot 与无人机负责
物体检测。成功仍由未篡改的真实 PDDL 条件判断；本实现不伪造 success。

## 现有链路与方案选择

现有 CLI 通过 FAULTS 注册表生成 Hydra 配置。evaluator 在单环境中维护
simulator_step，并可回放 clean 前缀；multi_llm_policy 分别调用 agent 的
HierarchicalPolicy，随后合并 take_actions 发送至 env.step。高层模型通常
只在需要新技能时调用，不能以每步必须产生新对话作为在线判据。

消息通过 AgentMessageBus.send_request 立即进入收件队列。OracleNavPolicy
输出 PDDL 实体索引，OracleNavAction 再从仿真真值取得物体位置并缓存导航
目标。只改 LLM 文本的位置描述无法确保 PF-1 影响实际运动。

采用按类别独立注入器与共同生命周期协议，在上述已有环节接入。
相比在提示词中模拟全部故障，这一方案能证明执行受到影响；相比新增独立
环境包装器，它保留已有 CLI、回放和轨迹入口，减少两套实验逻辑。

## 共同配置与生命周期

新增 action、planning、communication、perception 四个配置子树，每个
注入器拥有 enabled、fault_id、agent、episode_id、simulator_step、
simulator_step_schedule。单次运行只启用一种故障，与现有结构故障互斥。

AF-1/PF-1 支持 duration_steps：正整数表示半开窗口 [start, start+duration)，
-1 表示持续至 episode 结束。PL-1 在指定步执行一次规划输入替换并持续至
episode 结束。CF-1 在窗口内截获消息，每条消息由独立 delay_steps 确定
投递步；窗口结束后已截获消息仍按原定时间送达。

触发统一使用从零开始的 simulator_step，发生于该步的策略/环境执行之前。
重复评估同一 episode、新 episode 和环境自动 reset 必须清空生命周期、
消息队列及仿真侧定位偏差。PF-1 切换窗口时清理该 agent 的相关导航缓存。

CLI 保留 --fault、--agent、--episode/--random-episodes、--scene、--step、
--trajectory、--replay-trajectory、--dry-run；增加 --duration-steps、
--object、--replacement-object、--delay-steps、--position-offset DX DY DZ。
仅接受当前故障适用的参数；不可用实体、非法持续时间、非有限坐标、零偏移
等直接报错，不产生看似有效的故障运行。plan.json 保存实际生效参数。

## AF-1：导航动作无效

故障名 action_noop。窗口内正常执行目标 agent 的 policy.act，保留技能
状态、模型上下文、消息处理和控制循环。策略返回后复制真实下发的动作张量，
仅屏蔽导航通道；oracle 实体索引 0 是不执行导航。若配置包含坐标导航，
依据其实际动作空间和无效值单独处理，不能直接清零整条 joint action。

保留原始策略意图与屏蔽后的 take_actions，不覆盖高层技能编号、停止控制
或另一台机器人的动作。依据 HierarchicalPolicy 的 actions 与 take_actions
语义维护历史，不能将高层技能编号当作低层动作清零。正常 wait 不计为
effective application；只在屏蔽了有效导航命令时记录故障实际作用。

原生验证应证明指令有效、实际下发无效、故障窗口内导航位移/转向被阻止，
窗口外动作恢复，策略与另一台机器人仍可运行。不通过改写机器人位置来
强制静止；物理环境中的外力位移不属于执行器正常执行。

## PL-1：错误子目标

故障名 wrong_subgoal。指定 --object 为原子任务中的目标实体，
--replacement-object 为场景内存在且不同的替换实体。替换目标 agent 的
subtask_description 中完整匹配的实体 token，并更新其执行系统提示词及
规划输入。总体任务与真实成功条件保留。

若触发于首次初始化之前，修改将传入 init_agent 的子任务；若在中途触发，
保留模型历史，重新提供被篡改的任务并要求规划，再打断旧技能以进入下一次
高层决策。使用明确的“任务更新/重新规划”提示，避免错误声称旧动作完成。
这属于规划输入故障，不直接替换模型已经选择的导航动作，后者应归为 AF-2。

若原目标不在实际分配中，或替换后输入没有变化，应拒绝/明确记为未作用，
不能虚报有效注入。记录原子任务、修改后任务、规划输入和后续真实决策。
模型是否接受错误目标、是否导致最终失败，是实验结果，不预设为必然。

## CF-1：消息延迟

故障名 message_delay，--agent 表示受影响的接收者。窗口内发给它的消息
先进入待投递队列，交付步为 send_step+delay_steps。每步策略调用之前
统一投递到期消息，以便相同步中较晚执行的 agent 也不能提前收到消息。

待投递消息拥有 episode 内唯一 ID、发送者、接收者、原文、发送步、
计划投递步与实际投递步。同一投递步保持发送顺序。原始 sent_messages
仍表示真实发送，另有 delivery 记录表示实际送达，不能把投递重复算作
发送。正常成员不可达行为继续适用；接收者后续 dropout 时消息丢弃并
记录原因。episode 结束仍待投递的消息显式记为未送达，reset 不得泄漏。

验证至少发生一条真实 send_request，接收者在到期前读不到该消息，到期步
可以处理，内容和身份未变。没有消息经过窗口只算触发，没有有效作用。

## PF-1：物体定位偏差

故障名 object_localization。指定目标物体 --object 与世界坐标偏移
--position-offset DX DY DZ（米，Habitat 中 Y 为竖直方向）。目标 agent
读到的估计位置为真实位置加偏移；只作用于该 agent 与该实体。

Habitat task 暴露受控的 agent/entity 位置估计接口，经 VectorEnv 同步
当前窗口状态。OracleNavAction 查询导航实体位置时读取估计位置，而 PDDL
真值和 rigid object 的真实 transform 继续读取原始位置。偏差变更时清理
受影响缓存，保证已经开始的导航也会受影响，窗口结束后重新使用真值。

目标 agent 的 scene_description 同时使用相同估计位置，并确保下一次
高层决策可读到更新；其他 agent 的描述保持各自正常观测。更新应替换
对应物体的定位描述，不在原描述后叠加一条与真实位置冲突的“故障告警”。
当前 HSSD 描述以高度与到可导航点的水平距离呈现位置；应从同一估计位置
重新生成这些字段，并可提供估计 XYZ，不能只改一个未被模型读取的坐标。
结构化实体映射避免凭物体显示名称猜测 PDDL ID。

记录真实位置、估计位置、偏移、目标实体及导航消费证据。偏移可能导致
不可达或导航失败，这仍是有效故障结果；不能静默退回真实位置掩盖故障。
原生验证包括初始导航目标改变、另一 agent 不受影响、物体真实位置未被
修改，以及窗口结束后的导航缓存恢复。

## 事件、轨迹与回放

复用 JSONL 与中文可读轨迹入口，但扩展通用事件支持，避免只匹配
_dropout_event/_hang_event。每类包含 fault_injected、实际作用记录和
适用的 fault_expired；消息另外记录排队、送达、丢弃。

所有证据关联 scene_id、episode_id、agent、simulator_step 与 fault_id。
运行结束分别汇总是否触发、是否有效作用、作用次数与未作用原因。
回放严格验证注入前前缀，在注入步切到 live policy；任何故障不得影响
此前的上下文、消息、动作与真实指标。clean 默认关闭全部故障。

已有 recovery_batch 是 dropout 配对实验协议，不将新故障偷偷映射为
dropout。四类先通过统一单次 CLI 和其 episode 选择功能运行；自动批量
恢复收益分析不是本次错误注入的验收条件。

## 验证与交付

1. 生命周期与参数测试：触发边界、持续窗口、reset、互斥与非法参数。
2. 行为测试：真实策略调用与 AF-1 双动作证据；PL-1 完整实体替换、
   保留历史和规划输入；CF-1 延迟顺序和跨 episode 隔离；PF-1 单 agent
   估计与缓存失效。测试需验证外部行为，不能只镜像实现。
3. CLI/Hydra：四种 fault 的 dry-run、参数写入 plan/config、真实配置
   compose、无故障默认兼容、dropout/hang/replay 回归。
4. 原生 Habitat：使用已安装 /home/dyc/.conda/envs/habitat/bin/python
   与本地 ReplicaCAD 资产，固定测试模型避免 API 凭据依赖；Spot 与
   无人机均覆盖导航注入，四类均有真实执行证据。显式断言被测运行并非
   仅加载配置或发出事件。沙箱内 CUDA 不可见；已通过提升权限的只读
   nvidia-smi 确认两张 NVIDIA RTX PRO 6000 Blackwell 显卡可用，原生
   仿真应使用相应运行权限。若原生仿真无法运行，必须报告尚未完成该项，
   不能据单元测试宣称完成。
5. 使用文档提供每种故障参数、实际运行命令、日志阅读方式和语义边界。

保留用户当前暂存的研究内容.md 及父目录同名文件，所有实现局限在 EMOS。
当前分支 other-fault；不自动提交用户已暂存的文件。

## 修改前的验证基线

2026-10-07，在未修改实现代码的状态下，使用 Habitat Python 运行现有
dropout、hang、policy、trajectory、launcher、clean launcher、replay、
error detection、recovery、perception、recovery batch、evaluation output
及 PDDL target validation 共 13 个 unittest 模块：159 项测试全部通过，
运行退出码 0。测试覆盖现有行为，不代表新四类故障已实现或已验证。

只读原生环境检查：habitat_sim 0.3.1、torch 2.8.0+cu128 可导入；
ReplicaCAD 场景 v3_sc3_staging_19.scene_instance.json 存在；沙箱中
torch.cuda.is_available() 为 False，但提升权限后 nvidia-smi -L 成功列出
GPU 0 和 GPU 1。设计确认后，验证流程应使用真实仿真运行进一步检查 EGL
与资产依赖，不能只以 GPU 设备可见作为原生验收通过的依据。

## 完成证据

185 项回归通过；14 个原生案例通过，见 fault_output/latent_native_validation_20261007_final/verification.json。包括两台机器人的真实动作/定位注入、错误目标决策、消息接收模型输入、四种前缀回放、消息/定位重复 episode 重置与异常终止证据。独立复审未发现残余 Important/Critical 问题。使用方法见 docs/FAULT_INJECTION_ZH.md。
