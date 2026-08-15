from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from habitat_mas.utils import models
from habitat_mas.agents.actions.base_actions import nav_to_obj, send_request, wait
from habitat_mas.agents.crab_agent import CrabAgent
from habitat_mas.utils.models import OpenAIModel, QwenActionProtocolError
from habitat_mas.utils.python_interpreter import ExecutionResult


class FakeAction:
    def __init__(self, name):
        self.name = name

    def to_openai_json_schema(self):
        return {
            "name": self.name,
            "description": f"{self.name} action",
            "parameters": {"type": "object", "properties": {}},
        }


def completion(content=None, tool_calls=None):
    message = SimpleNamespace(content=content, tool_calls=tool_calls)
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message)],
        usage=SimpleNamespace(total_tokens=1),
    )


def tool_call(name, arguments="{}", call_id="call-1"):
    return SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def make_model(monkeypatch, responses, *, planning=False, code_execution=False, actions=None):
    create = Mock(side_effect=responses)
    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    monkeypatch.setattr(models.openai, "OpenAI", lambda: client)
    model = OpenAIModel(
        system_prompt="test",
        action_space=actions or [],
        discussion_stage=planning,
        code_execution=code_execution,
        save_on_each_chat=False,
    )
    return model, create


def configure_interpreter(model, results):
    model.interpreter = Mock()
    model.interpreter._CODE_TYPE_MAPPING = {"python": "python"}
    model.interpreter.run_python_detailed.side_effect = results


def test_qwen_numerical_success_requests_code_free_final_decision(monkeypatch):
    monkeypatch.setenv("EMOS_LLM_BACKEND", "qwen")
    model, create = make_model(
        monkeypatch,
        [
            completion("```python\nprint('distance: 1.25')\n```"),
            completion("{{yes}}"),
        ],
        planning=True,
        code_execution=True,
    )
    configure_interpreter(
        model, [ExecutionResult(0, "distance: 1.25\n", "", False)]
    )

    assert model.chat("check geometry") == "{{yes}}"
    assert create.call_count == 2
    feedback = create.call_args_list[1].kwargs["messages"][-1]["content"]
    assert "distance: 1.25" in feedback
    assert "without another code block" in feedback


def test_qwen_repairs_failed_python_with_at_most_two_retries(monkeypatch):
    monkeypatch.setenv("EMOS_LLM_BACKEND", "qwen")
    model, create = make_model(
        monkeypatch,
        [
            completion("```python\nif True print('bad')\n```"),
            completion("```python\nprint('reachable: True')\n```"),
            completion("{{yes}}"),
        ],
        planning=True,
        code_execution=True,
    )
    configure_interpreter(
        model,
        [
            ExecutionResult(1, "", "SyntaxError: line 1: invalid syntax", False),
            ExecutionResult(0, "reachable: True\n", "", False),
        ],
    )

    assert model.chat("check geometry") == "{{yes}}"
    assert model.interpreter.run_python_detailed.call_count == 2
    assert create.call_count == 3
    repair_prompt = create.call_args_list[1].kwargs["messages"][-1]["content"]
    assert "SyntaxError" in repair_prompt
    assert "Correct only" in repair_prompt


def test_qwen_timeout_feedback_forbids_loops(monkeypatch):
    monkeypatch.setenv("EMOS_LLM_BACKEND", "qwen")
    model, create = make_model(
        monkeypatch,
        [
            completion("```python\nwhile True: pass\n```"),
            completion("{{no||cannot verify safely}}"),
        ],
        planning=True,
        code_execution=True,
    )
    configure_interpreter(model, [ExecutionResult(-9, "", "", True)])

    assert model.chat("check geometry") == "{{no||cannot verify safely}}"
    feedback = create.call_args_list[1].kwargs["messages"][-1]["content"]
    assert "exceeded 10 seconds" in feedback
    assert "Do not use loops" in feedback


def test_qwen_three_failed_code_attempts_return_safe_negative(monkeypatch):
    monkeypatch.setenv("EMOS_LLM_BACKEND", "qwen")
    model, create = make_model(
        monkeypatch,
        [completion("```python\nraise NameError('x')\n```")] * 3,
        planning=True,
        code_execution=True,
    )
    configure_interpreter(
        model,
        [ExecutionResult(1, "", "NameError: x", False)] * 3,
    )

    response = model.chat("check geometry")

    assert response.startswith("{{no||Numerical verification failed")
    assert create.call_count == 3
    assert model.interpreter.run_python_detailed.call_count == 3


@pytest.mark.parametrize(
    "first_tool_calls",
    [None, [tool_call("wait", "not-json")], [tool_call("forbidden", "{}")]],
)
def test_qwen_invalid_action_response_retries_to_valid_tool(
    monkeypatch, first_tool_calls
):
    monkeypatch.setenv("EMOS_LLM_BACKEND", "qwen")
    model, create = make_model(
        monkeypatch,
        [
            completion(content="invalid action", tool_calls=first_tool_calls),
            completion(tool_calls=[tool_call("wait", "{}")]),
        ],
        actions=[FakeAction("wait")],
    )

    assert model.chat("act") == ("wait", {})
    assert create.call_count == 2
    correction = create.call_args_list[1].kwargs["messages"][-1]["content"]
    assert "action response is invalid" in correction


def test_qwen_action_retry_exhaustion_raises_diagnostic(monkeypatch):
    monkeypatch.setenv("EMOS_LLM_BACKEND", "qwen")
    model, create = make_model(
        monkeypatch,
        [completion(content="no function call", tool_calls=None)] * 3,
        actions=[FakeAction("wait")],
    )

    with pytest.raises(QwenActionProtocolError, match="no tool call"):
        model.chat("act")

    assert create.call_count == 3


def test_qwen_multiple_tool_calls_are_retried(monkeypatch):
    monkeypatch.setenv("EMOS_LLM_BACKEND", "qwen")
    model, create = make_model(
        monkeypatch,
        [
            completion(
                tool_calls=[
                    tool_call("wait", "{}", "first"),
                    tool_call("pick", "{}", "second"),
                ]
            ),
            completion(tool_calls=[tool_call("wait", "{}")]),
        ],
        actions=[FakeAction("wait"), FakeAction("pick")],
    )

    assert model.chat("act") == ("wait", {})
    assert create.call_count == 2


def test_qwen_semantic_validator_retries_premature_wait(monkeypatch):
    monkeypatch.setenv("EMOS_LLM_BACKEND", "qwen")
    model, create = make_model(
        monkeypatch,
        [
            completion(tool_calls=[tool_call("wait", "{}")]),
            completion(
                tool_calls=[
                    tool_call(
                        "nav_to_obj",
                        "{\"target_obj\": \"any_targets|0\"}",
                    )
                ]
            ),
        ],
        actions=[FakeAction("wait"), FakeAction("nav_to_obj")],
    )

    def reject_wait(action_name, _parameters):
        return "unfinished assigned goal" if action_name == "wait" else None

    assert model.chat("act", action_validator=reject_wait) == (
        "nav_to_obj",
        {"target_obj": "any_targets|0"},
    )
    assert create.call_count == 2


def test_gpt_normal_action_response_is_unchanged(monkeypatch):
    monkeypatch.setenv("EMOS_LLM_BACKEND", "gpt")
    model, _ = make_model(
        monkeypatch,
        [completion(tool_calls=[tool_call("pick", '{"target_obj": "cup"}')])],
        actions=[FakeAction("pick")],
    )

    assert model.chat("act") == ("pick", {"target_obj": "cup"})


def test_gpt_numerical_execution_still_uses_legacy_interpreter(monkeypatch):
    monkeypatch.setenv("EMOS_LLM_BACKEND", "gpt")
    model, create = make_model(
        monkeypatch,
        [
            completion("```python\nprint('value: 3')\n```"),
            completion("{yes}"),
        ],
        planning=True,
        code_execution=True,
    )
    model.interpreter = Mock()
    model.interpreter._CODE_TYPE_MAPPING = {"python": "python"}
    model.interpreter.run.return_value = "value: 3\n"

    assert model.chat("check value") == "{yes}"
    model.interpreter.run.assert_called_once_with("print('value: 3')", "python")
    model.interpreter.run_python_detailed.assert_not_called()
    assert create.call_count == 2


def test_qwen_never_executes_a_fourth_code_block(monkeypatch):
    monkeypatch.setenv("EMOS_LLM_BACKEND", "qwen")
    model, create = make_model(
        monkeypatch,
        [completion("```python\nprint('value: 1')\n```")] * 4,
        planning=True,
        code_execution=True,
    )
    configure_interpreter(
        model,
        [ExecutionResult(0, "value: 1\n", "", False)] * 4,
    )

    response = model.chat("check value")

    assert response.startswith("{{no||Numerical verification failed")
    assert create.call_count == 4
    assert model.interpreter.run_python_detailed.call_count == 3


def make_crab_agent(monkeypatch, backend, subtask, chat_history=None):
    monkeypatch.setattr(models.OpenAIModel, "save_chat_history", lambda self, path: None)
    monkeypatch.setenv("EMOS_LLM_BACKEND", backend)
    create = Mock(return_value=completion("1. nav_to_obj(any_targets|0)"))
    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    monkeypatch.setattr(models.openai, "OpenAI", lambda: client)
    agent = CrabAgent("agent_0", [send_request, nav_to_obj, wait])
    agent.init_agent(
        "SpotRobot",
        "Detect any_targets|0",
        subtask,
        chat_history=chat_history,
        scene_description="scene description",
        peer_ids=("agent_1",),
    )
    return agent


def test_qwen_action_context_omits_numerical_transcript(monkeypatch):
    raw_history = [[{"role": "assistant", "content": "```python\nprint(1)\n```"}]]

    agent = make_crab_agent(
        monkeypatch,
        "qwen",
        "Detect any_targets|0",
        chat_history=raw_history,
    )

    assert "```python" not in repr(agent.llm_model.chat_history)
    system = agent.llm_model.system_message["content"]
    assert "Detect any_targets|0" in system
    assert "scene description" in system
    assert "agent_1" in system


def test_gpt_action_context_keeps_existing_group_history(monkeypatch):
    raw_history = [[{"role": "assistant", "content": "```python\nprint(1)\n```"}]]

    agent = make_crab_agent(
        monkeypatch,
        "gpt",
        "Detect any_targets|0",
        chat_history=raw_history,
    )

    assert "```python" in repr(agent.llm_model.chat_history)


def test_qwen_crab_validator_rejects_unknown_peer_and_premature_wait(monkeypatch):
    agent = make_crab_agent(monkeypatch, "qwen", "Detect any_targets|0")

    assert "real peer" in agent._validate_qwen_action(
        "send_request",
        {"request": "help", "target_agent": "agent_2"},
    )
    assert "unfinished" in agent._validate_qwen_action("wait", {})
    assert agent._validate_qwen_action(
        "nav_to_obj",
        {"target_obj": "any_targets|0"},
    ) is None
    assert "assigned subtask" in agent._validate_qwen_action(
        "nav_to_obj",
        {"target_obj": "TARGET_any_targets|0"},
    )


def test_qwen_nothing_to_do_allows_model_wait(monkeypatch):
    agent = make_crab_agent(monkeypatch, "qwen", "Nothing to do")

    assert agent._validate_qwen_action("wait", {}) is None


def test_qwen_issued_navigation_does_not_mark_detection_complete(monkeypatch):
    agent = make_crab_agent(monkeypatch, "qwen", "Detect any_targets|0")
    agent.llm_model.chat = Mock(
        return_value=("nav_to_obj", {"target_obj": "any_targets|0"})
    )

    action = agent.chat("step", completed_targets=())

    assert action == {
        "name": "nav_to_obj",
        "arguments": {"target_obj": "any_targets|0", "robot": "agent_0"},
    }
    assert agent.completed_targets == set()
