"""Run with habitat's Python; requests use an in-memory OpenAI transport."""

from trajectory_test_utils import read_events as expanded_events

import contextlib
import copy
import io
import json
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from openai.types.chat import ChatCompletion

from habitat_baselines.rl.hrl.hl.llm_policy import LLMHighLevelPolicy
from habitat_mas.agents.crab_agent import CrabAgent, REQUEST_TEMPLATE
from habitat_mas.utils import models
from habitat_mas.utils.trajectory import EvaluationTrace


def completion(content="done", tools=None):
    return ChatCompletion(
        id="completion-1",
        object="chat.completion",
        created=0,
        model="test-model",
        choices=[{
            "index": 0,
            "finish_reason": "tool_calls" if tools else "stop",
            "message": {
                "role": "assistant", "content": content, "tool_calls": tools,
            },
        }],
        usage={"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7},
    )


def tool(name, arguments, call_id="tool-1"):
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }


class Transport:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.requests = []

    def create(self, **kwargs):
        self.requests.append(copy.deepcopy(kwargs))
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        return response


class LLMTraceTests(unittest.TestCase):
    def setUp(self):
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
        output_dir = self.stack.enter_context(tempfile.TemporaryDirectory())
        self.trace = self.stack.enter_context(EvaluationTrace(output_dir))
        self.trace.start_episode(
            0, SimpleNamespace(scene_id="scene", episode_id="episode-1"),
        )
        self.stack.enter_context(self.trace.policy_step())

    def make_model(self, responses, actions=(), discussion=False, code=False):
        transport = Transport(responses)
        client = SimpleNamespace(chat=SimpleNamespace(completions=transport))
        self.stack.enter_context(patch.object(models.openai, "OpenAI", return_value=client))
        model = models.OpenAIModel(
            "system", list(actions), model="test-model",
            discussion_stage=discussion, code_execution=code,
            save_on_each_chat=False, agent_name="agent_0",
        )
        return model, transport

    def action(self, name, function):
        return SimpleNamespace(
            name=name, run=function,
            to_openai_json_schema=lambda: {
                "name": name,
                "parameters": {"type": "object", "properties": {}},
            },
        )

    def by_event(self, event_type):
        events = expanded_events(self.trace.run_dir)
        return [event for event in events if event["event"] == event_type]

    def first_event(self, event_type):
        events = self.by_event(event_type)
        self.assertTrue(events, f"Missing {event_type} event")
        return events[0]

    def test_discussion_parsers_preserve_original_outputs_and_log_fallback(self):
        from habitat_baselines.rl.multi_agent.multi_llm_policy import (
            parse_agent_response, parse_leader_response,
        )
        self.assertEqual(parse_leader_response("{agent_0||pick cup}"), {"agent_0": "pick cup"})
        self.assertEqual(parse_agent_response("{{yes}}", agent_name="agent_0"), ("yes", None))
        self.assertEqual(parse_agent_response("invalid reply", agent_name="agent_0"), ("no", "invalid reply"))
        events = self.by_event("discussion_parse")
        self.assertEqual(len(events), 3)
        self.assertTrue(events[-1]["used_fallback"])
        self.assertEqual(events[-1]["handling"], "return_existing_no_response")

    def test_client_timeout_and_retry_config_are_observed_without_new_kwargs(self):
        model, transport = self.make_model([completion("plan")])
        model.client.timeout = SimpleNamespace(connect=5.0, read=60.0, write=60.0, pool=10.0)
        model.client.max_retries = 2
        model.chat("plan", crab_planning=True)
        config = self.first_event("model_config")
        self.assertEqual(config["api_timeout_seconds"], {"connect": 5.0, "read": 60.0, "write": 60.0, "pool": 10.0})
        self.assertEqual(config["api_max_retries"], 2)
        self.assertNotIn("timeout", transport.requests[0])

    def test_execution_records_first_selected_tool_without_claiming_execution(self):
        response = completion("choice", [tool("pick", {"target_obj": "cup"}),
                                         tool("wait", {}, "tool-2")])
        model, transport = self.make_model([response], [self.action("pick", lambda: None)])

        self.assertEqual(model.chat("observation"), ("pick", {"target_obj": "cup"}))

        requests = self.by_event("llm_request")
        self.assertEqual(len(requests), 1)
        self.assertEqual(requests[0]["messages"], [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "observation"},
        ])
        self.assertEqual(requests[0]["phase"], "execution")
        self.assertEqual(requests[0]["agent_name"], "agent_0")
        self.assertEqual(requests[0]["model"], "test-model")
        self.assertEqual(transport.requests[0]["tool_choice"], "required")
        self.assertEqual(transport.requests[0]["tools"], model.openai_tools)
        response_event = self.first_event("llm_response")
        self.assertEqual(response_event["message"]["content"], "choice")
        self.assertEqual(response_event["usage"]["total_tokens"], 7)
        self.assertEqual(response_event["episode_id"], "episode-1")
        self.assertEqual(response_event["step"], 1)
        self.assertEqual(model.token_usage, 7)
        selected = self.by_event("tool_selection")
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0]["action_name"], "pick")
        self.assertEqual(selected[0]["parameters"], {"target_obj": "cup"})
        self.assertNotIn("success", selected[0])
        self.assertEqual(self.by_event("tool_execution"), [])

    def test_planning_records_text_and_does_not_add_tools_to_request(self):
        model, transport = self.make_model([completion("pick then place")])

        self.assertEqual(model.chat("plan", crab_planning=True), "pick then place")

        self.assertEqual(self.first_event("llm_request")["phase"], "planning")
        self.assertEqual(set(transport.requests[0]), {"messages", "model"})
        self.assertEqual(self.first_event("llm_response")["phase"], "planning")

    def test_discussion_records_actual_internal_tool_result(self):
        model, transport = self.make_model([
            completion("calculation", [tool("add", {"a": 2, "b": 3})]),
            completion("ready"),
        ], [self.action("add", lambda a, b: a + b)], discussion=True)

        self.assertEqual(model.chat("assign task"), "ready")

        events = self.by_event("tool_execution")
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["result"], "5")
        self.assertEqual(events[0]["parameters"], {"a": 2, "b": 3})
        self.assertEqual(events[0]["agent_name"], "agent_0")
        self.assertEqual(events[0]["phase"], "discussion")
        self.assertEqual(self.first_event("llm_request")["phase"], "discussion")
        self.assertEqual(len(self.first_event("llm_request")["messages"]), 2)
        self.assertEqual(transport.requests[1]["messages"][-1]["content"], "5")
        self.assertNotIn("tool_choice", transport.requests[0])

    def test_discussion_without_tools_records_response(self):
        model, transport = self.make_model([completion("ready")], discussion=True)

        self.assertEqual(model.chat("assign task"), "ready")

        self.assertEqual(self.first_event("llm_response")["phase"], "discussion")
        self.assertEqual(set(transport.requests[0]), {"messages", "model"})

    def test_api_error_is_recorded_and_original_exception_propagates(self):
        error = RuntimeError("transport unavailable")
        model, _ = self.make_model([error])

        with self.assertRaises(RuntimeError) as raised:
            model.chat("observation")

        self.assertIs(raised.exception, error)
        self.assertEqual(len(self.by_event("llm_request")), 1)
        self.assertEqual(self.by_event("llm_response"), [])
        event = self.first_event("llm_error")
        self.assertEqual(event["error_type"], "RuntimeError")
        self.assertEqual(event["error_message"], "transport unavailable")
        self.assertEqual(event["phase"], "execution")
        self.assertEqual(event["call_id"], self.first_event("llm_request")["call_id"])
        self.assertEqual(event["status"], "error")
        self.assertGreaterEqual(event["duration_seconds"], 0)
        self.assertIsNone(event["usage"])
        self.assertEqual(event["usage_unavailable_reason"], "request_failed_before_response")

    def test_calls_record_raw_response_configuration_and_duration_without_sending_defaults(self):
        model, transport = self.make_model([completion("plan")])

        self.assertEqual(model.chat("room", crab_planning=True), "plan")

        request = self.first_event("llm_request")
        response = self.first_event("llm_response")
        config = self.first_event("model_config")
        self.assertEqual(config["request_kwargs"], {"model": "test-model"})
        self.assertEqual(config["decoding_parameters"]["temperature"], None)
        self.assertEqual(config["parameter_sources"]["temperature"], "provider_default_not_sent")
        self.assertEqual(request["request_kwargs"], transport.requests[0])
        self.assertEqual(response["raw_response"]["id"], "completion-1")
        self.assertEqual(response["raw_response"]["choices"][0]["message"]["content"], "plan")
        self.assertEqual(request["call_id"], response["call_id"])
        self.assertEqual(config["call_id"], request["call_id"])
        self.assertEqual(response["status"], "success")
        self.assertGreaterEqual(response["duration_seconds"], 0)
        self.assertEqual(set(transport.requests[0]), {"messages", "model"})

    def test_malformed_tool_json_records_failure_and_retains_original_exception(self):
        malformed = tool("pick", {})
        malformed["function"]["arguments"] = '{"target_obj":'
        model, _ = self.make_model([completion("choice", [malformed])])

        with self.assertRaises(json.JSONDecodeError) as raised:
            model.chat("observation")

        self.assertEqual(raised.exception.doc, '{"target_obj":')
        event = self.by_event("llm_parse")[-1]
        self.assertEqual(event["status"], "error")
        self.assertEqual(event["handling"], "rethrow")
        self.assertEqual(event["raw_arguments"], '{"target_obj":')
        self.assertEqual(event["call_id"], self.first_event("llm_response")["call_id"])
        self.assertEqual(self.by_event("tool_selection"), [])

    def test_missing_tool_calls_logs_parse_failure_and_keeps_original_type_error(self):
        model, _ = self.make_model([completion("no tools")])

        with self.assertRaises(TypeError):
            model.chat("observation")

        event = self.first_event("llm_parse")
        self.assertEqual(event["parse_stage"], "tool_calls")
        self.assertEqual(event["status"], "error")
        self.assertEqual(event["handling"], "rethrow")

    def test_empty_response_choices_records_raw_response_before_original_failure(self):
        response = completion()
        response.choices = []
        model, _ = self.make_model([response])

        with self.assertRaises(IndexError):
            model.chat("observation")

        self.assertEqual(model.token_usage, 7)
        self.assertEqual(self.first_event("llm_response")["raw_response"]["choices"], [])
        event = self.first_event("llm_parse")
        self.assertEqual(event["parse_stage"], "response_message")
        self.assertEqual(event["handling"], "rethrow")

    def test_discussion_api_rounds_share_one_decision_and_distinct_call_ids(self):
        model, _ = self.make_model([
            completion("calculate", [tool("add", {"a": 2, "b": 3})]),
            completion("ready"), completion("ready again"),
        ], [self.action("add", lambda a, b: a + b)], discussion=True)

        self.assertEqual(model.chat("task"), "ready")
        self.assertEqual(model.chat("another task"), "ready again")

        requests = self.by_event("llm_request")
        self.assertEqual([event["decision_step"] for event in requests], [1, 1, 2])
        self.assertEqual(len({event["call_id"] for event in requests}), 3)
        self.assertEqual([event["env_step"] for event in requests], [1, 1, 1])

    def test_execution_placeholders_are_labeled_without_changing_chat_history(self):
        model, _ = self.make_model([completion("choice", [tool("wait", {})])])

        self.assertEqual(model.chat("observation"), ("wait", {}))

        self.assertEqual(model.chat_history[-1][-1]["content"], "Success")
        event = self.first_event("tool_response_placeholder")
        self.assertTrue(event["is_placeholder"])
        self.assertEqual(event["content"], "Success")
        self.assertNotIn("success", self.first_event("tool_selection"))

    def test_communication_send_receive_and_action_transformation_keep_original_behavior(self):
        sender_model, _ = self.make_model([
            completion("request", [tool("send_request", {"target_agent": "agent_1", "request": "move cup"})]),
        ])
        receiver_model, receiver_transport = self.make_model([
            completion("choice", [tool("pick", {"target_obj": "cup"})]),
        ])
        receiver_model.agent_name = "agent_1"
        sender, receiver = CrabAgent("agent_0", []), CrabAgent("agent_1", [])
        sender.llm_model, receiver.llm_model = sender_model, receiver_model
        self.stack.enter_context(patch.object(CrabAgent, "message_pipe", {}))

        self.assertEqual(sender.chat("room"), {"name": "wait", "arguments": ["500"]})
        self.assertEqual(receiver.chat("room"), {
            "name": "pick", "arguments": {"target_obj": "cup", "robot": "agent_1"},
        })

        sent, received = self.first_event("communication_send"), self.first_event("communication_receive")
        for event in (sent, received):
            self.assertEqual(event["sender"], "agent_0")
            self.assertEqual(event["recipient"], "agent_1")
            self.assertEqual(event["message"], "move cup")
            self.assertTrue(event["timestamp"])
        self.assertEqual(CrabAgent.message_pipe["agent_1"], [])
        self.assertEqual(receiver_transport.requests[0]["messages"][-1]["content"],
                         'room ' + REQUEST_TEMPLATE.format(source_agent="agent_0", request="move cup"))
        decisions = self.by_event("agent_tool_decision")
        self.assertEqual(decisions[-1]["parameters"], {"target_obj": "cup"})
        mapped = self.by_event("agent_action")[-1]
        self.assertEqual(mapped["output"]["arguments"], {"target_obj": "cup", "robot": "agent_1"})
        self.assertEqual(decisions[-1]["action_id"], mapped["action_id"])

    def test_self_request_is_suppressed_with_the_existing_none_return(self):
        model, _ = self.make_model([
            completion("request", [tool("send_request", {"target_agent": "agent_0"})]),
        ])
        agent = CrabAgent("agent_0", [])
        agent.llm_model = model
        self.stack.enter_context(patch.object(CrabAgent, "message_pipe", {}))

        self.assertIsNone(agent.chat("room"))

        self.assertEqual(CrabAgent.message_pipe, {})
        sent = self.first_event("communication_send")
        self.assertEqual(sent["status"], "suppressed")
        self.assertEqual(sent["reason"], "self_request")
        self.assertEqual(self.first_event("agent_action")["output"], None)

    def test_internal_tool_error_is_recorded_and_original_exception_propagates(self):
        error = ValueError("cannot add")

        def fail(**parameters):
            raise error

        model, _ = self.make_model([
            completion("calculation", [tool("add", {"a": 2})]),
        ], [self.action("add", fail)], discussion=True)

        with self.assertRaises(ValueError) as raised:
            model.chat("assign task")

        self.assertIs(raised.exception, error)
        event = self.first_event("tool_error")
        self.assertEqual(event["action_name"], "add")
        self.assertEqual(event["parameters"], {"a": 2})
        self.assertEqual(event["error_message"], "cannot add")
        self.assertEqual(self.by_event("tool_execution"), [])

    def test_python_execution_records_actual_output(self):
        model, _ = self.make_model([
            completion("```python\nprint(2 + 3)\n```"), completion("ready"),
        ], discussion=True, code=True)

        self.assertEqual(model.chat("calculate"), "ready")

        event = self.first_event("code_execution")
        self.assertEqual(event["code"], "print(2 + 3)")
        self.assertEqual(event["result"], "5\n")
        self.assertEqual(event["phase"], "discussion")

    def test_python_execution_error_is_recorded_and_propagates(self):
        error = OSError("cannot launch interpreter")
        model, _ = self.make_model([
            completion("```python\nprint(5)\n```"),
        ], discussion=True, code=True)
        self.stack.enter_context(patch.object(model.interpreter, "run", side_effect=error))

        with self.assertRaises(OSError) as raised:
            model.chat("calculate")

        self.assertIs(raised.exception, error)
        event = self.first_event("code_error")
        self.assertEqual(event["code"], "print(5)")
        self.assertEqual(event["error_type"], "OSError")
        self.assertEqual(event["error_message"], "cannot launch interpreter")

    def test_skill_selection_records_mapped_arguments_and_wait_fallback(self):
        cases = [
            ({"name": "pick", "arguments": {"target_obj": "cup", "robot": "agent_0"}},
             "pick", ["cup", "agent_0"], None),
            ({"name": "unsupported", "arguments": {}}, "wait", ["500"], "unknown_action"),
            (None, "wait", ["500"], "llm_output_none"),
        ]
        for index, (output, expected_name, expected_args, fallback_reason) in enumerate(cases):
            with self.subTest(output=output):
                policy = LLMHighLevelPolicy.__new__(LLMHighLevelPolicy)
                torch.nn.Module.__init__(policy)
                policy._skill_name_to_idx = {"wait": 0, "pick": 1}
                policy.llm_agent = SimpleNamespace(
                    initialized=True, start_act=False, name="agent_0",
                    chat=lambda _: output, get_token_usage=lambda: 0,
                )
                skills, args, _, _ = policy.get_next_skill(
                    {}, None, None, torch.ones((1, 1)), [1.0], False, [],
                    envs_text_context=[{"scene_description": "room"}],
                    agent_arguments=[object()],
                )
                self.assertEqual(skills.item(), policy._skill_name_to_idx[expected_name])
                self.assertEqual(args, [expected_args])
                events = self.by_event("skill_selection")
                self.assertEqual(len(events), index + 1)
                event = events[-1]
                self.assertEqual(event["llm_output"], output)
                self.assertEqual(event["skill_name"], expected_name)
                self.assertEqual(event["skill_args"], expected_args)
                self.assertEqual(event["agent_name"], "agent_0")
                self.assertEqual(event["fallback_reason"], fallback_reason)

    def test_skill_argument_error_is_logged_and_original_exception_propagates(self):
        policy = LLMHighLevelPolicy.__new__(LLMHighLevelPolicy)
        torch.nn.Module.__init__(policy)
        policy._skill_name_to_idx = {"wait": 0, "pick": 1}
        policy.llm_agent = SimpleNamespace(
            initialized=True, start_act=False, name="agent_0",
            chat=lambda _: {"name": "pick", "arguments": {}},
            get_token_usage=lambda: 0,
        )

        with self.assertRaises(KeyError) as raised:
            policy.get_next_skill(
                {}, None, None, torch.ones((1, 1)), [1.0], False, [],
                envs_text_context=[{"scene_description": "room"}],
                agent_arguments=[object()],
            )

        self.assertEqual(raised.exception.args, ("target_obj",))
        event = self.first_event("skill_parse")
        self.assertEqual(event["status"], "error")
        self.assertEqual(event["handling"], "rethrow")
        self.assertEqual(self.by_event("skill_selection"), [])


if __name__ == "__main__":
    unittest.main()
