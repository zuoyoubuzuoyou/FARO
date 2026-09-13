import unittest

from habitat_mas.agents.message_bus import AgentMessageBus
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

    def test_dropout_is_visible_and_failed_request_replans_immediately(self):
        AgentMessageBus.members = {"agent_0", "agent_1", "agent_2"}
        AgentMessageBus.drop_agent("agent_1")
        sender = CrabAgent("agent_0", [])
        sender.llm_model = FakeLLMModel([
            ("send_request", {"target_agent": "agent_1", "request": "help"}),
            ("send_request", {"target_agent": "agent_2", "request": "help"}),
        ])
        sender.chat("observation")
        self.assertIn("agent_1 left the team", sender.llm_model.observations[0])
        self.assertIn("Current team members: agent_0, agent_2", sender.llm_model.observations[0])
        self.assertIn("agent_unavailable", sender.llm_model.observations[1])
        self.assertNotIn("agent_1", CrabAgent.message_pipe)
        self.assertEqual(len(CrabAgent.message_pipe["agent_2"]), 1)

    def test_repeated_unreachable_requests_have_bounded_retries(self):
        AgentMessageBus.drop_agent("agent_1")
        sender = CrabAgent("agent_0", [])
        sender.llm_model = FakeLLMModel([
            ("send_request", {"target_agent": "agent_1", "request": "help"})
        ] * 3)
        self.assertEqual(sender.chat("obs"), {"name": "wait", "arguments": ["500"]})
        self.assertEqual(len(sender.llm_model.observations), 3)
        self.assertNotIn("agent_1", CrabAgent.message_pipe)

    def test_dropped_agent_cannot_run_or_send_and_reset_restores_it(self):
        agent = CrabAgent("agent_1", [])
        agent.llm_model = FakeLLMModel([("wait", {})])
        AgentMessageBus.send_request("agent_0", "agent_1", "pending")
        AgentMessageBus.drop_agent("agent_1")
        agent.chat("dropped")
        self.assertEqual(agent.llm_model.observations, [])
        self.assertNotIn("agent_1", CrabAgent.message_pipe)
        self.assertIn("agent_unavailable", AgentMessageBus.send_request("agent_1", "agent_0", "help"))
        CrabAgent.reset_message_pipe()
        agent.chat("restored")
        self.assertEqual(agent.llm_model.observations, ["restored"])


if __name__ == "__main__":
    unittest.main()
