import copy
import json
import re
from typing import List, Optional

from .crab_core import Action
from .message_bus import AgentMessageBus
from ..utils.models import OpenAIModel


PDDL_TARGET_PATTERN = re.compile(r"^(?:TARGET_)?any_targets\|\d+$")
PDDL_TARGET_ARGUMENTS = {
    "nav_to_obj": ("target_obj",),
    "nav_to_goal": ("target_obj",),
    "pick": ("target_obj",),
    "place": ("target_obj", "target_location"),
}
MAX_ACTION_CORRECTION_RETRIES = 2


def _invalid_pddl_target_arguments(
    action_name: str, parameters: dict
) -> dict[str, object]:
    """Return target arguments that are not valid PDDL entity names."""

    invalid = {}
    for argument_name in PDDL_TARGET_ARGUMENTS.get(action_name, ()):
        value = parameters.get(argument_name)
        if (
            not isinstance(value, str)
            or PDDL_TARGET_PATTERN.fullmatch(value) is None
        ):
            invalid[argument_name] = value
    return invalid


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
    message_pipe = AgentMessageBus.messages

    @classmethod
    def reset_message_pipe(cls) -> None:
        """Reset communication and membership at an episode boundary."""
        AgentMessageBus.reset()

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
        skip_planning: bool = False,
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

        if skip_planning:
            return

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

    def replan(self, subtask_description: str, recovery_context: str) -> None:
        """Replace the assignment and plan from current state, retaining history."""
        self.subtask_description = subtask_description
        self.llm_model.set_system_message(ROBOT_EXECUTION_SYSTEM_PROMPT_TEMPLATE.format(
            robot_type=self.robot_type, robot_key=self.name,
            subtask_description=subtask_description,
        ) + "\nRecovery execution constraints:\n" + recovery_context)
        self.llm_model.chat(
            "Recovery supersedes the previous task assignment and action plan.\n"
            + recovery_context + "\nYour remaining assignment:\n" + subtask_description
            + "\nAvailable actions:\n" + self.action_prompt
            + "\nReplan a full action sequence for this assignment while preserving satisfied goals.",
            crab_planning=True,
        )
        # The next high-level action must receive fresh scene context, not the
        # usual prompt claiming the interrupted low-level action completed.
        self.start_act = False

    def export_replay_context(self) -> dict:
        """Record model memory and communication at a high-level boundary."""
        def encode(value):
            return value.model_dump(mode="json")

        return json.loads(json.dumps({
            "robot_type": self.robot_type,
            "task_description": self.task_description,
            "subtask_description": self.subtask_description,
            "system_message": self.llm_model.system_message,
            "chat_history": self.llm_model.chat_history,
            "start_act": self.start_act,
            "token_usage": self.llm_model.token_usage,
            "messages": AgentMessageBus.messages,
            "members": sorted(AgentMessageBus.members),
            "dropped_agents": sorted(AgentMessageBus.dropped_agents),
            "sent_messages": AgentMessageBus.sent_messages,
        }, default=encode))

    def restore_replay_context(self, context: dict) -> None:
        context = copy.deepcopy(context)
        for field in ("robot_type", "task_description", "subtask_description", "start_act"):
            setattr(self, field, context[field])
        self.llm_model.system_message = context["system_message"]
        self.llm_model.chat_history = context["chat_history"]
        self.llm_model.token_usage = context["token_usage"]
        AgentMessageBus.messages.clear()
        AgentMessageBus.messages.update(context["messages"])
        AgentMessageBus.members = set(context["members"])
        AgentMessageBus.dropped_agents = set(context["dropped_agents"])
        AgentMessageBus.sent_messages = context.get("sent_messages", [])

    def replay_decision(self, decision: dict) -> None:
        if "replay_context" in decision:
            self.restore_replay_context(decision["replay_context"])
        else:
            # Legacy traces lack the original assignment, model conversation and
            # message bus. Keep an explicitly reconstructed decision history.
            self.start_act = True
            self.llm_model.chat_history.append([{
                "role": "user",
                "content": "Recorded high-level action selected at this point: "
                + json.dumps({k: decision[k] for k in ("action", "arguments")})
                + ". This is reconstructed execution history; the original conversation was not recorded.",
            }])

    def chat(self, observation: str) -> Optional[dict]:
        if self.name in AgentMessageBus.dropped_agents:
            return {"name": "wait", "arguments": ["500"]}
        membership = AgentMessageBus.membership_observation()
        if membership:
            observation = str(observation) + " " + membership
        if self.name in CrabAgent.message_pipe and CrabAgent.message_pipe[self.name]:
            prompt = " ".join(CrabAgent.message_pipe[self.name])
            observation = str(observation) + " " + prompt
            CrabAgent.message_pipe[self.name] = []

        action_name = "wait"
        parameters = {}
        for attempt in range(MAX_ACTION_CORRECTION_RETRIES + 1):
            action_name, parameters = self.llm_model.chat(str(observation))
            if action_name == "send_request":
                target_agent = parameters["target_agent"]
                if target_agent == self.name:
                    return None
                error = AgentMessageBus.send_request(
                    self.name, target_agent, parameters["request"]
                )
                if error is None or attempt == MAX_ACTION_CORRECTION_RETRIES:
                    return {"name": "wait", "arguments": ["500"]}
                observation = (
                    f"{observation}\n{error} Choose another action or request "
                    "help from a current team member."
                )
                continue
            invalid_targets = _invalid_pddl_target_arguments(
                action_name, parameters
            )
            if not invalid_targets:
                break

            if attempt == MAX_ACTION_CORRECTION_RETRIES:
                print(
                    f"Agent {self.name} repeatedly returned invalid PDDL "
                    "targets; using wait instead of forwarding an invalid "
                    "action to Habitat."
                )
                return {"name": "wait", "arguments": ["500"]}

            observation = (
                f"Your previous {action_name!r} call used invalid PDDL target "
                f"arguments: {invalid_targets!r}. Scene asset names (for example "
                "'048_hammer_:0000') are descriptions only and cannot be passed "
                "to actions. Retry with one action whose target arguments use "
                "the exact PDDL entity names from the goal conditions, such as "
                "'any_targets|0' or 'TARGET_any_targets|0'."
            )
            print(
                f"Invalid PDDL target from {self.name}; retrying action "
                f"generation ({attempt + 1}/{MAX_ACTION_CORRECTION_RETRIES}): "
                f"{invalid_targets}"
            )

        if action_name == "wait":
            return {"name": "wait", "arguments": ["500"]}
        if action_name in [
            "nav_to_obj",
            "nav_to_goal",
            "nav_to_robot",
            "place",
            "pick",
        ]:
            parameters["robot"] = self.name
            return {"name": action_name, "arguments": parameters}
        else:
            return {"name": action_name, "arguments": parameters}

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
