| L1                      | 建议保留的代表性 L2                                                                                                            | 合并逻辑                                                                                                                 |
| ----------------------- | ---------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------- |
| **Structure Fault**     | Agent dropout / crash；agent hang / straggler；role or identity mismatch                                                 | 把 dropout、crash、hang 保留，因为好实现；unexpected join 暂时不做                                                                   |
| **Perception Fault**    | object localization error；object recognition error；spatial relation error；agent/action recognition error               | 把 false positive、false negative、misclassification 合并为 recognition error；把 depth、occlusion 等先并入 localization/relation |
| **Memory Fault**        | memory corruption；wrong / missing retrieval；summary / context distortion                                               | 保留和 agent decision context 直接相关的错误，不要做太多存储细节                                                                         |
| **Planning Fault**      | wrong goal/subgoal；missing subgoal；invalid ordering / dependency violation；failure to replan / plan loop               | 把 PL-1 到 PL-10 合并成 4 类，更适合实验                                                                                         |
| **Communication Fault** | message loss/delay；wrong recipient；ambiguous / referential drift；contradictory or stale message                        | 传输层和语义层各保留 2 类                                                                                                       |
| **Coordination Fault**  | duplicate / missing assignment；resource/path conflict；deadlock/livelock；bad handoff / role drift                       | 只保留最能影响多智能体协作的错误                                                                                                     |
| **Capability Fault**    | capability mismatch；missing/wrong tool；tool timeout；reachability violation                                             | 很适合 Habitat / PARTNR 这种 heterogeneous agents                                                                         |
| **Action Fault**        | action no-op；wrong target/action；partial execution / grasp failure；navigation failure / collision；false success return | 把动作层面错误合并为 5 类，保留机器人最常见失败                                                                                            |
| **Verification Fault**  | incomplete verification；false positive / false negative verification；cross-agent verification failure                  | 和 action fault 区分： Action 是“做错了”，Verification 是“没发现做错了”。                                                             |





### 第一阶段我们可以先实现最简单的一些例子：

| 编号  | Fault                         | 例子                             |
| --- | ----------------------------- | ------------------------------ |
| F1  | Object localization error     | agent 以为杯子在桌上，实际在柜子里           |
| F2  | Object recognition error      | 把 cup 识别成 bowl                 |
| F3  | Wrong subgoal                 | 总任务要拿杯子，agent 收到“拿盘子”          |
| F4  | Missing subgoal               | 应该先 open cabinet，但 planner 省略了 |
| F5  | Message delay / stale message | agent 收到过期位置或任务进度              |
| F6  | Duplicate assignment          | 两个 agent 都去拿同一个物体              |
| F7  | Action no-op / false success  | pick up 失败，但返回 success         |
| F8  | Missing / false verification  | 任务没完成，但 verifier 判断完成          |





### 三个比较适合开始的MAS框架

- **EMOS / Habitat-MAS** -> 可以先用这个跑通pipeline
- **PARTNR Planner / PARTNR Baselines**
- **Watch-And-Help / VirtualHome-Social Baselines**


### Schema

{
  "fault_id": "PF_wrong_location_001",
  "fault_type": "PerceptionFault",
  "fault_subtype": "WrongObjectLocation",
  "injected_at_step": 12,
  "faulty_agent": "agent_1",
  "affected_object": "cup",
  "affected_subtask": "pick_up_cup",
  "severity": "medium",
  "ground_truth_state": "cup_on_table",
  "agent_observed_state": "cup_on_counter",
  "expected_recovery": "refresh_observation_or_ask_teammate"
}