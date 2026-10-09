import unittest

from habitat_mas.agents.message_bus import AgentMessageBus as Bus


class MessageDelayTest(unittest.TestCase):
    def setUp(self):
        Bus.reset()
        Bus.members = {"agent_0", "agent_1"}
        self.addCleanup(Bus.reset)

    def test_no_early_delivery_and_order(self):
        Bus.begin_step(5, {"agent": "agent_1", "delay_steps": 3, "fault_id": "delay"})
        for content in ("first", "second"):
            self.assertIsNone(Bus.send_request("agent_0", "agent_1", content))
        self.assertEqual(Bus.messages, {})
        self.assertEqual(len(Bus.sent_messages), 2)
        events = Bus.drain_delivery_events()
        self.assertEqual([e["type"] for e in events], ["message_queued"] * 2)
        self.assertNotEqual(events[0]["message_id"], events[1]["message_id"])
        for step in (6, 7):
            Bus.begin_step(step)
            self.assertEqual(Bus.messages, {})
        Bus.begin_step(8)
        self.assertIn("first", Bus.messages["agent_1"][0])
        self.assertIn("second", Bus.messages["agent_1"][1])
        delivered = Bus.drain_delivery_events()
        self.assertEqual([e["actual_delivery_step"] for e in delivered], [8, 8])
        self.assertEqual(len(Bus.sent_messages), 2)
        Bus.begin_step(9)
        self.assertEqual(len(Bus.messages["agent_1"]), 2)

    def test_unaffected_receiver_is_immediate(self):
        Bus.begin_step(5, {"agent": "agent_1", "delay_steps": 3, "fault_id": "delay"})
        Bus.send_request("agent_1", "agent_0", "immediate")
        self.assertIn("immediate", Bus.messages["agent_0"][0])
        self.assertEqual(Bus.drain_delivery_events(), [])

    def test_dropout_discards_pending_and_reset_does_not_leak(self):
        Bus.begin_step(5, {"agent": "agent_1", "delay_steps": 3, "fault_id": "delay"})
        Bus.send_request("agent_0", "agent_1", "lost")
        Bus.drain_delivery_events()
        Bus.drop_agent("agent_1")
        self.assertEqual(Bus.drain_delivery_events()[0]["reason"], "agent_unavailable")
        Bus.begin_step(8)
        self.assertNotIn("agent_1", Bus.messages)
        Bus.reset()
        Bus.begin_step(8)
        self.assertEqual(Bus.pending_messages, [])
        self.assertEqual(Bus.messages, {})

    def test_episode_end_reports_undelivered(self):
        Bus.begin_step(5, {"agent": "agent_1", "delay_steps": 3, "fault_id": "delay"})
        Bus.send_request("agent_0", "agent_1", "pending")
        Bus.drain_delivery_events()
        Bus.finish_episode(6)
        event = Bus.drain_delivery_events()[0]
        self.assertEqual(event["type"], "message_dropped")
        self.assertEqual(event["reason"], "episode_ended_before_delivery")
        self.assertEqual(Bus.pending_messages, [])


if __name__ == "__main__":
    unittest.main()
