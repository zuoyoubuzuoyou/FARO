from typing import List, Tuple, Dict, Union, Callable, Optional
import openai
from concurrent.futures import ThreadPoolExecutor, as_completed
import random
import re


# from habitat_mas.agents.llm_agent_base import LLMAgentBase
class DummyAgent:
    def __init__(self, **kwargs):
        self.agent_name = kwargs.get("agent_name", "agent_0")
        self.pddl_problem = kwargs.get("pddl_problem", None)
        # self.all_entities = self.pddl_problem.all_entities
        self.initilized = True

    def _get_action(self, object, receptacle=None):
        return [
            {
                "name": "nav_to_position",
                "arguments": {
                    "target_position": [0.1, 0.1, 0.1],
                    "robot": self.agent_name,
                }
            }
        ]
        # return [
        #         {
        #             "name": "nav_to_obj",
        #             "arguments": {
        #                 "target_obj": receptacle,
        #                 "robot": self.agent_name,
        #             }
        #         },
        #         {
        #             "name": "nav_to_goal",
        #             "arguments": {
        #                 "target_obj": object,
        #                 "robot": self.agent_name,
        #             }
        #         },
        #         {
        #             "name": "pick",
        #             "arguments": {
        #                 "target_obj": object,
        #                 "robot": self.agent_name,
        #             }
        #         },
        #         {
        #             "name": "place",
        #             "arguments": {
        #                 "target_obj": object,
        #                 "target_location": receptacle,
        #                 "robot": self.agent_name,
        #             }
        #         },
        #         {
        #             "name": "reset_arm",
        #             "arguments": {
        #                 "robot": self.agent_name,
        #             }
        #         },
        #     ]

    def get_token_usage(self):
        return 0

    def init_agent(self, **kwargs):
        return 

    def chat(self, skill_name_to_idx) -> Optional[dict]:
        # """
        # Mimic a chatbot always sending the oraclenav action.
        # """
        # object_pattern = r'^any_targets\|\d+$'
        # receptacle_pattern = r'^TARGET_any_targets\|\d+$'
        # objects = [key for key in self.all_entities.keys() if re.match(object_pattern, key)]
        # receptacles = [key for key in self.all_entities.keys() if re.match(receptacle_pattern, key)]

        # if len (receptacles) > 0:
        #     object = random.choice(objects)
        #     receptacle = random.choice(receptacles)
        #     action_list = self._get_action(object, receptacle)
        #     return random.choice(
        #         [action for action in action_list if action['name'] in list(skill_name_to_idx.keys())]
        #     )
        # else:
        #     object = random.choice(objects)
        #     action_list = self._get_action(object)
        #     return random.choice(
        #         [action for action in action_list if action['name'] == 'nav_to_goal']
        #     )
        return {
            "name": "nav_to_position",
            "arguments": {
                "target_position": [0.1, 0.1, 0.1],
                "robot": self.agent_name,
            }
        }