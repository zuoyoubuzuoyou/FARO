import unittest

from habitat_mas.agents.crab_agent import (
    CrabAgent,
    _invalid_pddl_target_arguments,
)


class FakeLLMModel:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.observations = []

    def chat(self, observation):
        self.observations.append(observation)
        return next(self.responses)


class CrabAgentTargetValidationTest(unittest.TestCase):
    def setUp(self):
        CrabAgent.reset_message_pipe()

    def tearDown(self):
        CrabAgent.reset_message_pipe()

    def test_accepts_pddl_navigation_target(self):
        self.assertEqual(
            _invalid_pddl_target_arguments(
                "nav_to_obj", {"target_obj": "any_targets|1"}
            ),
            {},
        )

    def test_accepts_pddl_place_targets(self):
        self.assertEqual(
            _invalid_pddl_target_arguments(
                "place",
                {
                    "target_obj": "any_targets|0",
                    "target_location": "TARGET_any_targets|0",
                },
            ),
            {},
        )

    def test_rejects_scene_asset_name(self):
        self.assertEqual(
            _invalid_pddl_target_arguments(
                "nav_to_obj", {"target_obj": "048_hammer_:0000"}
            ),
            {"target_obj": "048_hammer_:0000"},
        )

    def test_rejects_missing_target_argument(self):
        self.assertEqual(
            _invalid_pddl_target_arguments("pick", {}),
            {"target_obj": None},
        )

    def test_ignores_actions_without_pddl_targets(self):
        self.assertEqual(_invalid_pddl_target_arguments("wait", {}), {})

    def test_retries_invalid_scene_asset_name(self):
        agent = CrabAgent("agent_1", [])
        agent.llm_model = FakeLLMModel(
            [
                ("nav_to_obj", {"target_obj": "048_hammer_:0000"}),
                ("nav_to_obj", {"target_obj": "any_targets|1"}),
            ]
        )

        result = agent.chat("observation")

        self.assertEqual(
            result,
            {
                "name": "nav_to_obj",
                "arguments": {
                    "target_obj": "any_targets|1",
                    "robot": "agent_1",
                },
            },
        )
        self.assertEqual(len(agent.llm_model.observations), 2)
        self.assertIn(
            "048_hammer_:0000", agent.llm_model.observations[1]
        )

    def test_falls_back_to_wait_after_repeated_invalid_targets(self):
        agent = CrabAgent("agent_1", [])
        invalid_response = (
            "nav_to_obj",
            {"target_obj": "048_hammer_:0000"},
        )
        agent.llm_model = FakeLLMModel([invalid_response] * 3)

        result = agent.chat("observation")

        self.assertEqual(result, {"name": "wait", "arguments": ["500"]})
        self.assertEqual(len(agent.llm_model.observations), 3)

    def test_message_is_delivered_within_an_episode(self):
        sender = CrabAgent("agent_0", [])
        recipient = CrabAgent("agent_1", [])
        sender.llm_model = FakeLLMModel(
            [
                (
                    "send_request",
                    {
                        "target_agent": "agent_1",
                        "request": "current episode message",
                    },
                )
            ]
        )
        recipient.llm_model = FakeLLMModel([("wait", {})])

        sender.chat("sender observation")
        recipient.chat("recipient observation")

        self.assertIn(
            "current episode message",
            recipient.llm_model.observations[0],
        )

    def test_reset_discards_unread_message_from_previous_episode(self):
        sender = CrabAgent("agent_0", [])
        recipient = CrabAgent("agent_1", [])
        sender.llm_model = FakeLLMModel(
            [
                (
                    "send_request",
                    {
                        "target_agent": "agent_1",
                        "request": "previous episode message",
                    },
                )
            ]
        )
        recipient.llm_model = FakeLLMModel([("wait", {})])

        sender.chat("previous episode observation")
        CrabAgent.reset_message_pipe()
        recipient.chat("new episode observation")

        self.assertEqual(
            recipient.llm_model.observations,
            ["new episode observation"],
        )


if __name__ == "__main__":
    unittest.main()
