from typing import List, Optional

from .crab_core import Action
from ..utils.models import OpenAIModel
from ..utils.trajectory import decision_scope, record_event


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
        print("===============CrabAgent Subtasks==============")
        print(response)

    def chat(self, observation: str) -> Optional[dict]:
        with decision_scope(self.name, phase="execution"):
            return self._chat(observation)

    def _record_action_output(self, output, transformation):
        record_event(
            "agent_action", agent_name=self.name, phase="execution",
            output=output, transformation=transformation,
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
