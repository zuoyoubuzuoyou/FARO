"""Opt-in native Habitat smoke run with a deterministic model, requiring no API key.

Pass the usual habitat_baselines.run arguments. This exercises simulator/policy
integration, not real model quality or recovery success rate. Not a unittest test:
it requires scene assets and GPU access and is run explicitly as a module.
"""

import re
import runpy


class FixedModel:
    def __init__(self, system_prompt, action_space, discussion_stage=False,
                 agent_name="unknown", **kwargs):
        self.set_system_message(system_prompt)
        self.chat_history = []
        self.token_usage = 0
        self.discussion_stage = discussion_stage
        self.agent_name = agent_name

    def set_system_message(self, prompt):
        self.system_message = {"role": "system", "content": prompt}
        assignment = prompt.split('"""')[1]
        self.targets = list(dict.fromkeys(re.findall(r"(?<![\w|])any_targets\|\d+", assignment)))
        self.action_index = 0

    def chat(self, prompt, crab_planning=False):
        self.chat_history.append([{"role": "user", "content": prompt}])
        if self.discussion_stage:
            if self.agent_name == "leader":
                return "{agent_0||Detect any_targets|0}{agent_1||Detect any_targets|1}"
            return "{yes}"
        if crab_planning:
            return "Navigate to the remaining assigned targets in order."
        if not self.targets:
            return "wait", ["500"]
        target = self.targets[self.action_index % len(self.targets)]
        self.action_index += 1
        return "nav_to_obj", {"target_obj": target}


def main():
    from habitat_mas.agents import crab_agent
    from habitat_baselines.rl.multi_agent import multi_llm_policy

    crab_agent.OpenAIModel = FixedModel
    multi_llm_policy.OpenAIModel = FixedModel
    runpy.run_module("habitat_baselines.run", run_name="__main__")


if __name__ == "__main__":
    main()
