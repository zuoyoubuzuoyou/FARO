"""Opt-in native Habitat runs with a fixed model and no external API calls.

Use the normal habitat_baselines.run arguments plus --smoke-fault=NAME.
The model is only an integration fixture, not a recovery performance baseline.
"""

import runpy
import sys
import os
import json
from pathlib import Path

from habitat_mas.test.native_dropout_recovery_smoke import FixedModel as NavigationModel


class FixedModel(NavigationModel):
    fault_kind = ""

    def chat(self, prompt, crab_planning=False):
        if (not self.discussion_stage and not crab_planning and self.fault_kind == "message_delay"
                and self.agent_name == "agent_0" and self.action_index == 0):
            self.chat_history.append([{"role": "user", "content": prompt}])
            self.action_index += 1
            result = "send_request", {"target_agent": "agent_1", "request": "Please detect any_targets|1 and report progress."}
        else:
            result = super().chat(prompt, crab_planning=crab_planning)
        if os.environ.get("EMOS_NATIVE_MODEL_EVIDENCE"):
            from habitat_mas.agents.message_bus import AgentMessageBus
            with Path(os.environ["EMOS_NATIVE_MODEL_EVIDENCE"]).open("a") as output:
                output.write(json.dumps({"agent": self.agent_name, "step": AgentMessageBus.simulator_step,
                    "planning": crab_planning, "discussion": self.discussion_stage,
                    "prompt": prompt, "assignment": self.targets, "result": result}) + "\n")
        if (os.environ.get("EMOS_NATIVE_ABORT") and not self.discussion_stage and not crab_planning
                and self.agent_name == "agent_1" and self.fault_kind == "message_delay"):
            # The earlier-dispatched sender already queued a real request.
            from habitat_mas.agents.message_bus import AgentMessageBus
            if AgentMessageBus.pending_messages:
                raise RuntimeError("native smoke deliberate abort")
        return result


def main():
    option = next((value for value in sys.argv[1:] if value.startswith("--smoke-fault=")), None)
    if option is not None:
        FixedModel.fault_kind = option.partition("=")[2]
        sys.argv.remove(option)
    from habitat_mas.agents import crab_agent
    from habitat_baselines.rl.multi_agent import multi_llm_policy
    crab_agent.OpenAIModel = FixedModel
    multi_llm_policy.OpenAIModel = FixedModel
    original_act = multi_llm_policy.MultiLLMPolicy.act

    def act_with_fault_observation(self, *args, **kwargs):
        runtime = kwargs.get("latent_fault_runtime")
        step = kwargs.get("simulator_steps", [0])[0]
        if runtime is not None and runtime.spec.name == "message_delay":
            start = runtime.spec.target_simulator_step(runtime.window.episode_id)
            if start is not None and start <= step <= start + runtime.spec.delay_steps:
                # Exercise actual recipient model observations on both sides of
                # delivery, without waiting for the full navigation skill.
                receiver = int(runtime.spec.agent.rsplit("_", 1)[1])
                self._active_policies[receiver]._cur_call_high_level.fill_(True)
        if runtime is not None and runtime.spec.name == "object_localization":
            start = runtime.spec.target_simulator_step(runtime.window.episode_id)
            if step in (start, start + runtime.spec.duration_steps):
                receiver = int(runtime.spec.agent.rsplit("_", 1)[1])
                self._active_policies[receiver]._cur_call_high_level.fill_(True)
        return original_act(self, *args, **kwargs)

    multi_llm_policy.MultiLLMPolicy.act = act_with_fault_observation
    runpy.run_module("habitat_baselines.run", run_name="__main__")


if __name__ == "__main__":
    main()
