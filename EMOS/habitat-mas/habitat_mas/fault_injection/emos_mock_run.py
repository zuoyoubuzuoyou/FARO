"""Run the real EMOS/Habitat pipeline with a deterministic local mock LLM.

This is an integration smoke runner, not an experimental policy baseline. It
never creates an OpenAI client or sends scene/task content over the network.
"""

from __future__ import annotations


class MockOpenAIModel:
    """Small compatibility stub for discussion and CrabAgent action planning."""

    def __init__(
        self,
        *args,
        discussion_stage=False,
        agent_name="unknown",
        **kwargs,
    ):
        self.discussion_stage = discussion_stage
        self.agent_name = agent_name
        self.chat_history = []
        self.token_usage = 0

    def chat(self, content, crab_planning=False):
        if self.discussion_stage:
            if self.agent_name == "leader":
                return (
                    "{agent_0||Nothing to do}\n"
                    "{agent_1||Detect all target objects}"
                )
            return "{{yes}}"
        if crab_planning:
            return "Use wait while testing the fault-injection pipeline."
        return "wait", {}


def main() -> None:
    # Patch both symbols that construct an OpenAIModel before Hydra starts the
    # evaluator. All remaining environment and policy code is the real path.
    from habitat_baselines.rl.multi_agent import multi_llm_policy
    from habitat_mas.agents import crab_agent

    multi_llm_policy.OpenAIModel = MockOpenAIModel
    crab_agent.OpenAIModel = MockOpenAIModel

    from habitat_baselines.run import main as habitat_main

    habitat_main()


if __name__ == "__main__":
    main()
