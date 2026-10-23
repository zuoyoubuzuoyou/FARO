# 系统错误检测与评估

用户目标：实现启发式与LLM两类检测，用已有自然运行轨迹评估episode级Precision与Recall。沿用四批数据：Rearrange 20261011_215201（49），Mobility 20261012_152653（99），Perception HSSD 20261013_162242（103），Manipulation 20261015_092413＋retry3（139）。共390个完整episode，未完成记录不算失败。

## 检测对象与信息边界

检测器只接收当前及过去的任务、分工、动作、通信和实际状态，不读取最终结果、原有分类、故障注入状态或旧检测器输出。原生当前成功指标可作为模拟器特权观测，必须说明；历史锁存stage不是当前条件。先支持原始schema1逐帧JSONL及新schema2事件日志，两者转换为相同的流式frame接口。回放与在线调用使用同一检测器状态机，不修改原模型、动作、反馈或恢复策略。

同一导航命令持续10帧并不是停滞证据。实现三项可比较方法：repeat_action基线（动作名连续超过阈值）；team_progress启发式（未完成的团队职责与持续静止/复位/等待/错误完成反馈结合）；LLM检测（结构化当前上下文、最近窗口及历史进度摘要，按固定间隔调用）。正常守点、队友仍在推进、已完成职责者复位，不自动报警。

时间参数必须写明单位。window=10、check_every=5默认指10个观测点、每5个观测点检查；sample_every=1时等于仿真帧，sample_every=50时覆盖约500帧、每250帧检查。用户示例与推荐配置都提供，评估记录配置，不混用高层决策次数与环境步数。两种主要检测器使用相同采样与窗口，短episode不会通过终局强制补查窥视结束。每集首次有效报警后可停止LLM调用以降低开销。

## 数据与接口

Python3.9，无新的必需依赖；核心规则和指标只用标准库。独立包habitat_mas.detection，CLI位于scripts/evaluate_system_errors.py。原始文件与旧分析只读，新输出放analysis_results/detection/。

Frame为JSON字典：step；task；context（task_description、scene_description、assignments、robot_capabilities）；agents（position、rotation、held_object、action{name,arguments,id,age}、navigation、termination、detection、stop）；objects（标识到position/object_id）；metrics（白名单当前任务指标）；decisions（本步新增的agent/tool/arguments/feedback/subtask）；messages（本步真实通信）。未知字段为null，不补造false。适配器逐帧产生Frame；迭代终局/未来数据不提前进入上下文。

统一Alarm：step、detector、reason_code、explanation、agents、evidence、confidence、status。每条报警必须引用窗口内真实步号；LLM必须返回JSON，区分system_error/normal/uncertain，解析错误与请求失败另记，不能当正常或阴性。LLM请求的原文、响应、时间、usage、模型、缓存键全部保存；只从环境或.env读取凭证，输出脱敏。缓存包含输入、prompt、配置与模型hash，可恢复中断评估，避免重复收费。

## 评估真值

用户已明确选择：评估运行中是否真实异常，另做人工标注后计算正式P/R。正式评估唯一目标为runtime_system_error；390条新人工标注初始均为空，包括成功episode。既有失败归因仅保留为背景来源，不生成代理分数，不自动迁移为运行异常标签。输出待标注表，只有人工提供的有效0/1进入正式分母；人工未完成时明确显示awaiting_manual_labels，P/R为null，不把这一状态宣称已完成数值实验。

严格区分runtime_system_error与system_failure_attribution两种目标。前者需要独立运行异常标签，现有记录未完成，不能从成功或其他主归因反推阴性。其他三任务AI归因也不作为人工GT。标签表保留来源、目标和unknown，允许导入用户独立标注。没有足够已确认标签或LLM覆盖不完整时，指标为null/partial且解释原因，不伪造数值。

TP/FP/FN/TN按episode计算，一集报警多次也只计一次。P=TP/(TP+FP)，R=TP/(TP+FN)，零分母为null。失败率F/N与系统失败占比S/F独立于检测方法。主表、逐集结果、报警明细、请求统计、标签模板均输出JSON和UTF-8 BOM CSV；中文说明与机器字段兼容。可同时报告all_episodes与failed_episodes_only两个明确范围。

## 验证

用会破坏实际行为的反例测试：正常长导航、到位等队友、已完成者复位、未完成者静止、终止反馈错误、缺诊断、schema2自动重置、未知标签、无报警零分母、重复报警、缺失预测、LLM坏JSON/超时/缓存、未到查询步的短episode、禁止未来与旧检测输出泄漏。真实API先运行有限样本确认服务与JSON输出，再进行可恢复的完整评估；不把假传输测试结果当真实LLM分数。
