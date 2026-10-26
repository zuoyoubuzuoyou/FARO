from typing import List, Optional
from copy import deepcopy
import json

from .crab_core import Action
from ..utils.models import OpenAIModel
from ..utils.trajectory import current_action_id, decision_scope, record_event


REQUEST_TEMPLATE = '"{source_agent}" agent sent you requests: "{request}".'

ROBOT_EXECUTION_SYSTEM_PROMPT_TEMPLATE = (
    'You are a "{robot_type}" agent called "{robot_key}".'
    # " Your task is to work with other agents to complete the task described below:\n\n"
    # '"""\n{task_description}\n"""\n\n'
    "You MUST take finish subtask assigned to you:"
    '"""\n{subtask_description}\n"""\n\n'
    # "You have the following capabilities:\n\n"
    # '"""\n{capabilities}\n"""\n\n'
    "You MUST take one and only one action using function call in each step."
    " If you think the task definitely cannot be done by yourself, you can use `send_request` function to ask other agents for help."
)


class CrabAgent:
    message_pipe: dict[str, list[str]] = {}

    def __init__(
        self,
        name: str,
        actions: List[Action],
        code_execution: bool = False,
        **kwargs,
    ):
        self.name = name
        self.actions = actions
        self.code_execution = code_execution
        self.enable_logging =  kwargs.get("enable_logging", False)
        self.logging_file = kwargs.get("logging_file", "")

        self.llm_model:OpenAIModel = None
        self.initialized = False
        self.start_act = False
        self.planner_plan = None
        self._recovery_actions = []
        self._recovery_feedback = []

        self.action_prompt = _generate_action_prompt(self.actions, include_arguments=True)

    def get_token_usage(self):
        return self.llm_model.token_usage

    def init_agent(
        self,
        robot_type: str,
        task_description: str,
        subtask_description: str,
        chat_history: Optional[List] = None,
        enable_logging: bool = False,
        logging_file: str = "",
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        seed: Optional[int] = None,
    ):
        """This function is a hack to initialize agent after the object is created"""
        self.robot_type = robot_type
        self.task_description = task_description
        self.subtask_description = subtask_description
        # create system message
        system_message = ROBOT_EXECUTION_SYSTEM_PROMPT_TEMPLATE.format(
            robot_type=self.robot_type,
            robot_key=self.name,
            # task_description=task_description,
            subtask_description=subtask_description,
        )
        # Initialize the OpenAI LLM model
        self.llm_model = OpenAIModel(
            system_message, 
            action_space=self.actions,
            model=model,
            temperature=temperature,
            seed=seed,
            code_execution=self.code_execution,
            enable_logging=enable_logging,
            logging_file=logging_file,
            agent_name=self.name
        )
        
        # Inject chat history from group discussion phase
        if chat_history is not None:
            self.llm_model.chat_history = chat_history

        # Set logging parameters
        self.llm_model.enable_logging = enable_logging
        self.llm_model.logging_file = logging_file

        self.initialized = True
        self.start_act = False
        self._recovery_actions = []
        self._recovery_feedback = []

        # Guide agent to decouple subtasks into actions
        if len(subtask_description) > 0:
            subtask_to_actions_prompt = (
                f"Your assigned subtask is:"
                f'"""\n{subtask_description}\n"""\n\n'
                f"You are provided with the following actions:\n{self.action_prompt}\n"
                "Now you should convert it into a FULL subtask action sequence to complete YOUR assigned subtask."
            )
        else:
            subtask_to_actions_prompt = (
                " Your task is to work with other agents to complete the task described below:\n\n"
                f'"""\n{task_description}\n"""\n\n'
                f"You are provided with the following actions:\n{self.action_prompt}\n"
                "Now you should plan a FULL subtask action sequence to complete the WHOLE task."
            )
        print("===============CrabAgent Prompt==============")
        print(subtask_to_actions_prompt)
        response = self.llm_model.chat(subtask_to_actions_prompt, crab_planning=True)
        self.planner_plan = response
        print("===============CrabAgent Subtasks==============")
        print(response)

    def chat(self, observation: str) -> Optional[dict]:
        with decision_scope(self.name, phase="execution"):
            return self._chat(observation)

    # Explicit allowlists keep runtime clients, skills and simulator handles out
    # of memory snapshots. Missing fields support partially initialized agents.
    _REPLAY_AGENT_FIELDS = (
        "robot_type", "task_description", "subtask_description", "initialized",
        "start_act", "planner_plan", "_recovery_actions", "_recovery_feedback",
    )
    _REPLAY_MODEL_FIELDS = ("chat_history", "system_message", "token_usage")
    _REPLAY_MODEL_CONFIG = (
        "model", "window_size", "openai_tools", "actions", "tool_calls_enable",
        "planning_stage", "code_execution", "temperature", "seed",
    )

    @staticmethod
    def _replay_fields(obj, fields):
        return {name: deepcopy(getattr(obj, name)) for name in fields
                if hasattr(obj, name)}

    def export_replay_context(self):
        """Detach decision memory; this is not an environment checkpoint."""
        return {
            "schema_version": 1,
            "agent_name": self.name,
            "agent_config": {"code_execution": self.code_execution,
                             "action_prompt": self.action_prompt},
            "agent_memory": self._replay_fields(self, self._REPLAY_AGENT_FIELDS),
            "model_present": self.llm_model is not None,
            "model_config": self._replay_fields(self.llm_model, self._REPLAY_MODEL_CONFIG),
            "model_memory": self._replay_fields(self.llm_model, self._REPLAY_MODEL_FIELDS),
            "message_pipe": deepcopy(CrabAgent.message_pipe),
        }

    def restore_replay_context(self, context):
        """Restore memory onto compatible existing objects, without model calls.

        The global communication pipe is restored as a whole. Apply contexts in
        original agent_action event order when reconstructing multiple actors.
        """
        if not isinstance(context, dict):
            raise ValueError("Replay context must be an object")
        snapshot = deepcopy(context)
        expected_config = {"code_execution": self.code_execution,
                           "action_prompt": self.action_prompt}
        model_config = self._replay_fields(self.llm_model, self._REPLAY_MODEL_CONFIG)
        saved_model_config = snapshot.get("model_config")
        if isinstance(saved_model_config, dict):
            # Earlier contexts omitted these optional request controls. Their
            # absence matches only an unset current value, including seed=0.
            model_config = {name: value for name, value in model_config.items()
                            if name not in ("temperature", "seed") or value is not None
                            or name in saved_model_config}
        if (type(snapshot.get("schema_version")) is not int
                or type(snapshot.get("model_present")) is not bool
                or snapshot.get("schema_version") != 1
                or snapshot.get("agent_name") != self.name
                or snapshot.get("agent_config") != expected_config
                or snapshot.get("model_present") != (self.llm_model is not None)
                or saved_model_config != model_config):
            raise ValueError("Replay context requires compatible agent/model configuration")
        # Validate every memory key before changing any live state.
        for key, fields in (("agent_memory", self._REPLAY_AGENT_FIELDS),
                            ("model_memory", self._REPLAY_MODEL_FIELDS)):
            if not isinstance(snapshot.get(key), dict) or set(snapshot[key]) - set(fields):
                raise ValueError("Invalid replay memory fields: " + key)
        agent_memory = snapshot["agent_memory"]
        model_memory = snapshot["model_memory"]
        required_agent_fields = set(self._replay_fields(self, self._REPLAY_AGENT_FIELDS))
        required_model_fields = set(self._replay_fields(self.llm_model, self._REPLAY_MODEL_FIELDS))
        if (not required_agent_fields.issubset(agent_memory)
                or set(model_memory) != required_model_fields
                or (self.llm_model is None and model_memory)):
            raise ValueError("Incomplete or model-incompatible replay memory")
        for name in ("robot_type", "task_description", "subtask_description"):
            if name in agent_memory and not isinstance(agent_memory[name], str):
                raise ValueError("Invalid replay text field: " + name)
        for name in ("initialized", "start_act"):
            if name in agent_memory and type(agent_memory[name]) is not bool:
                raise ValueError("Invalid replay boolean field: " + name)
        if not isinstance(agent_memory.get("planner_plan"), (type(None), str, dict, list, tuple)):
            raise ValueError("Invalid replay planner memory")
        feedback = agent_memory.get("_recovery_feedback")
        if not isinstance(feedback, list) or any(not isinstance(item, str) for item in feedback):
            raise ValueError("Invalid replay recovery feedback")
        commands = agent_memory.get("_recovery_actions")
        if not isinstance(commands, list) or any(
                not isinstance(command, dict) or not isinstance(command.get("name"), str)
                or not isinstance(command.get("arguments"), dict) for command in commands):
            raise ValueError("Invalid replay recovery actions")

        def valid_message(message):
            # Live snapshots may contain SDK messages; persisted ones contain
            # their model_dump dictionaries, including nullable content/tools.
            if not isinstance(message, dict) and callable(getattr(message, "model_dump", None)):
                message = message.model_dump()
            if not isinstance(message, dict) or message.get("role") not in (
                    "system", "developer", "user", "assistant", "tool", "function"):
                return False
            content = message.get("content")
            if content is not None and not isinstance(content, (str, list)):
                return False
            if isinstance(content, list) and any(not isinstance(part, dict) for part in content):
                return False
            calls = message.get("tool_calls")
            if calls is not None and (not isinstance(calls, list) or any(
                    not isinstance(call, dict) for call in calls)):
                return False
            return True

        if "chat_history" in model_memory:
            history = model_memory["chat_history"]
            if not isinstance(history, list) or any(
                    not isinstance(group, list) or any(not valid_message(message) for message in group)
                    for group in history):
                raise ValueError("Invalid grouped replay history")
        if "system_message" in model_memory:
            system = model_memory["system_message"]
            if not valid_message(system) or system.get("role") != "system":
                raise ValueError("Invalid replay system message")
        if "token_usage" in model_memory:
            usage = model_memory["token_usage"]
            if type(usage) is not int or usage < 0:
                raise ValueError("Invalid replay token usage")
        pipe = snapshot.get("message_pipe")
        if not isinstance(pipe, dict) or any(
                not isinstance(name, str) or not isinstance(messages, list)
                or any(not isinstance(message, str) for message in messages)
                for name, messages in pipe.items()):
            raise ValueError("Invalid replay communication pipe")
        for name, value in snapshot["agent_memory"].items():
            setattr(self, name, value)
        for name, value in snapshot["model_memory"].items():
            setattr(self.llm_model, name, value)
        CrabAgent.message_pipe = pipe

    def _record_action_output(self, output, transformation):
        memory = {}
        if current_action_id(self.name) is not None:
            memory["replay_context"] = self.export_replay_context()
        record_event(
            "agent_action", agent_name=self.name, phase="execution",
            output=output, transformation=transformation, **memory,
        )
        return output

    def _chat(self, observation: str) -> Optional[dict]:
        if self.name in CrabAgent.message_pipe and CrabAgent.message_pipe[self.name]:
            prompt = " ".join(CrabAgent.message_pipe[self.name])
            observation = str(observation) + " " + prompt
            for message in CrabAgent.message_pipe[self.name]:
                # The pipe remains a list of the original rendered strings.
                # Recover metadata from the exact existing request template.
                source, separator, request = message.partition('" agent sent you requests: "')
                recognized = bool(separator and source.startswith('"') and request.endswith('".'))
                record_event(
                    "communication_receive", agent_name=self.name,
                    phase="execution", sender=source[1:] if recognized else None,
                    recipient=self.name, message=request[:-2] if recognized else message,
                    rendered_message=message, status="consumed",
                    sender_source="request_template" if recognized else "unknown",
                )
            CrabAgent.message_pipe[self.name] = []

        if self._recovery_feedback:
            observation = str(observation) + "\n\n" + "\n".join(self._recovery_feedback)
            self._recovery_feedback = []

        if self._recovery_actions:
            command = self._recovery_actions.pop(0)
            action_name = command["name"]
            parameters = deepcopy(command["arguments"])
            # Keep the forced decision in the same conversation the next model
            # request will see, without inventing a successful tool result.
            self.llm_model.chat_history.append([
                {"role": "user", "content": str(observation)},
                {"role": "assistant", "content": (
                    "Live recovery selected the following tool; its outcome is "
                    "not yet known: " + json.dumps(command, ensure_ascii=False)
                )},
            ])
            record_event(
                "recovery_action", agent_name=self.name, phase="execution",
                action_name=action_name, parameters=parameters,
                source="validated_recovery_command", simulator_reset=False,
            )
        else:
            action_name, parameters = self.llm_model.chat(str(observation))
        record_event(
            "agent_tool_decision", agent_name=self.name, phase="execution",
            action_name=action_name, parameters=parameters,
        )
        if action_name == "send_request":
            target_agent = parameters["target_agent"]
            if target_agent == self.name:  # send request to itself
                record_event(
                    "communication_send", agent_name=self.name,
                    phase="execution", sender=self.name, recipient=target_agent,
                    message=parameters.get("request"), status="suppressed",
                    handling="return_none", reason="self_request",
                )
                return self._record_action_output(None, "self_request_suppressed")
            request = parameters["request"]
            if target_agent not in CrabAgent.message_pipe:
                CrabAgent.message_pipe[target_agent] = []
            prompt = REQUEST_TEMPLATE.format(source_agent=self.name, request=request)
            CrabAgent.message_pipe[target_agent].append(prompt)
            record_event(
                "communication_send", agent_name=self.name, phase="execution",
                sender=self.name, recipient=target_agent, message=request,
                rendered_message=prompt, status="queued", handling="existing_wait",
            )
            return self._record_action_output(
                {"name": "wait", "arguments": ["500"]}, "send_request_to_wait",
            )
        if action_name == "wait":
            return self._record_action_output(
                {"name": "wait", "arguments": ["500"]}, "existing_wait_arguments",
            )
        if action_name in [
            "nav_to_obj",
            "nav_to_goal",
            "nav_to_robot",
            "place",
            "pick",
        ]:
            parameters["robot"] = self.name
            return self._record_action_output(
                {"name": action_name, "arguments": parameters}, "inject_robot_argument",
            )
        else:
            return self._record_action_output(
                {"name": action_name, "arguments": parameters}, "identity",
            )

def _generate_action_prompt(action_space: list[Action], include_arguments: bool = False) -> str:
    if include_arguments:
        return "".join(
            [
                f"[**{action.name}**:\n"
                f"action description: {action.description}\n"
                f"action arguments json schema: {action.parameters.model_json_schema()}\n"
                "]\n"
                for action in action_space
            ]
        )
    else:
        return "".join(
            [f"[{action.name}: {action.description}]\n" for action in action_space]
        )
