import json
import os
import hashlib
import time
from uuid import uuid4
from ..agents.crab_core import Action
from typing import List
import openai
from .python_interpreter import SubprocessInterpreter
from .trajectory import decision_scope, record_event
# from openai.types.chat.chat_completion import ChatCompletionMessage
# from openai.types.chat.chat_completion_message_tool_call import ChatCompletionMessageToolCall
from pydantic import BaseModel

class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, BaseModel):
            # Pydantic models are not serializable by json.dump by default
            return dict(obj)
        return super().default(obj)
        
class OpenAIModel:
    def __init__(
        self,
        system_prompt: str,
        action_space: List[Action],
        model="gpt-5.5",
        window_size=None,
        discussion_stage=False,
        code_execution=False,
        enable_logging=False,
        logging_file="",
        save_on_each_chat=True,
        agent_name="unknown",
    ) -> None:
        self.system_message = {
            "role": "system",
            "content": system_prompt,
        }
        self._convert_action_to_schema(action_space)
        self.action_map = {action.name: action for action in action_space}
        self.chat_history = []
        self.window_size = window_size
        self.model = model
        self.client = openai.OpenAI()
        self.planning_stage = discussion_stage
        self.code_execution = code_execution
        if self.code_execution:
            self.interpreter = SubprocessInterpreter()
        self.openai_tools = (
            [{"type": "function", "function": action} for action in self.actions]
            if action_space
            else None
        )
        self.tool_calls_enable = True if action_space else False
        self.token_usage = 0
        
        # Debug logging
        self.enable_logging = enable_logging
        self.logging_file = logging_file
        self.save_on_each_chat = save_on_each_chat
        self.agent_name = agent_name

    
    def __del__(self):
        """Save chat history to a file if logging is enabled"""
        if self.enable_logging:
            self.save_chat_history(self.logging_file)
           
    def save_chat_history(self, file_path: str):

        with open(file_path, "w") as f:
            full_history = [self.system_message] + self.chat_history
            json.dump(full_history, f, indent=2, cls=CustomJSONEncoder)
        
        episode_path = os.path.dirname(file_path)
        token_path = os.path.join(episode_path, "token_usage.json")
        if not os.path.exists(token_path):
            with open(token_path, 'w') as f:
                json.dump({}, f)

        with open(token_path, 'r') as f:
            data = json.load(f)
        data[f"{self.agent_name}"] = self.token_usage

        with open(token_path, 'w') as f:
            json.dump(data, f, indent=4)

    def set_system_message(self, system_message: str):
        self.system_message = {"role": "system", "content": system_message}

    def execute_action(self, action_name: str, parameters: dict) -> str:
        print("Internal action: ", action_name, parameters)
        try:
            result = str(self.action_map[action_name].run(**parameters))
        except Exception as error:
            record_event(
                "tool_error", agent_name=self.agent_name, phase="discussion",
                action_name=action_name, parameters=parameters,
                error_type=type(error).__name__, error_message=str(error),
            )
            raise
        record_event(
            "tool_execution", agent_name=self.agent_name, phase="discussion",
            action_name=action_name, parameters=parameters, result=result,
        )
        return result

    def _create_completion(self, request, phase, **kwargs):
        call_id = str(uuid4())
        self._trace_call_id = call_id
        # These values describe only the request we send. Omitted settings are
        # unknown provider defaults, and are never added to the API arguments.
        decoding_names = (
            "temperature", "top_p", "max_tokens", "max_completion_tokens",
            "frequency_penalty", "presence_penalty", "seed", "stop", "n",
            "response_format", "reasoning_effort", "logprobs", "top_logprobs",
            "parallel_tool_calls", "stream",
        )
        client_missing = {}
        try:
            timeout = kwargs.get("timeout", getattr(self.client, "timeout", None))
            if isinstance(timeout, (int, float)):
                timeout_config = {"total": timeout}
            elif timeout is not None:
                timeout_config = {
                    name: getattr(timeout, name, None)
                    for name in ("connect", "read", "write", "pool")
                }
            else:
                timeout_config = None
            max_retries = getattr(self.client, "max_retries", None)
        except Exception:
            timeout_config, max_retries = None, None
        if timeout_config is None:
            client_missing["api_timeout_seconds"] = "SDK client does not expose its configured timeout"
        if max_retries is None:
            client_missing["api_max_retries"] = "SDK client does not expose its configured retry count"
        record_event(
            "model_config", agent_name=self.agent_name, phase=phase,
            call_id=call_id, model=self.model,
            request_kwargs={"model": self.model, **kwargs},
            api_timeout_seconds=timeout_config, api_max_retries=max_retries,
            missing_fields=client_missing,
            decoding_parameters={name: kwargs.get(name) for name in decoding_names},
            parameter_sources={
                name: "explicit_request" if name in kwargs else "provider_default_not_sent"
                for name in decoding_names
            },
            system_prompt_sha256=hashlib.sha256(
                self.system_message["content"].encode("utf-8")
            ).hexdigest() if isinstance(self.system_message.get("content"), str) else None,
        )
        record_event(
            "llm_request", agent_name=self.agent_name, phase=phase,
            call_id=call_id, model=self.model, messages=request,
            request_kwargs={"messages": request, "model": self.model, **kwargs},
            status="started", **kwargs,
        )
        started = time.monotonic()
        try:
            response = self.client.chat.completions.create(
                messages=request, model=self.model, **kwargs,
            )
        except Exception as error:
            record_event(
                "llm_error", agent_name=self.agent_name, phase=phase,
                call_id=call_id, model=self.model, status="error",
                duration_seconds=time.monotonic() - started, usage=None,
                usage_unavailable_reason="request_failed_before_response",
                error_type=type(error).__name__, error_message=str(error),
                handling="rethrow",
            )
            raise
        record_event(
            "llm_response", agent_name=self.agent_name, phase=phase,
            call_id=call_id, model=self.model,
            response_model=getattr(response, "model", None),
            raw_response=response,
            message=response.choices[0].message if response.choices else None,
            usage=response.usage, status="success",
            duration_seconds=time.monotonic() - started,
        )
        return response

    def chat(self, content: str, crab_planning=False):
        phase = "discussion" if self.planning_stage else (
            "planning" if crab_planning else "execution"
        )
        with decision_scope(self.agent_name, phase=phase):
            return self._chat(content, crab_planning=crab_planning)

    def _parse_tool_arguments(self, call, phase):
        try:
            parameters = json.loads(call.function.arguments)
        except Exception as error:
            record_event(
                "llm_parse", agent_name=self.agent_name, phase=phase,
                call_id=self._trace_call_id, status="error",
                parse_stage="tool_arguments", action_name=call.function.name,
                tool_call_id=call.id, raw_arguments=call.function.arguments,
                error_type=type(error).__name__, error_message=str(error),
                handling="rethrow",
            )
            raise
        record_event(
            "llm_parse", agent_name=self.agent_name, phase=phase,
            call_id=self._trace_call_id, status="success",
            parse_stage="tool_arguments", action_name=call.function.name,
            tool_call_id=call.id, raw_arguments=call.function.arguments,
            parameters=parameters, handling="continue",
        )
        return parameters

    def _parse_response_message(self, response, phase):
        try:
            return response.choices[0].message
        except Exception as error:
            record_event(
                "llm_parse", agent_name=self.agent_name, phase=phase,
                call_id=self._trace_call_id, status="error",
                parse_stage="response_message", raw_response=response,
                error_type=type(error).__name__, error_message=str(error),
                handling="rethrow",
            )
            raise

    def _chat(self, content: str, crab_planning=False):
        new_message = {"role": "user", "content": content}

        request = [self.system_message]
        # Add chat_history
        if self.window_size is None:
            for message in self.chat_history:
                request = request + message
        elif self.window_size > 0 and len(self.chat_history) > 0:
            for message in self.chat_history[-self.window_size :]:
                request = request + message

        request.append(new_message)
        self.chat_history.append([new_message])

        if self.planning_stage:
            while True:
                if self.tool_calls_enable:
                    response = self._create_completion(
                        request, phase="discussion",
                        tools=self.openai_tools,
                    )
                else:
                    response = self._create_completion(
                        request, phase="discussion",
                    )
                self.token_usage += response.usage.total_tokens

                response_message = self._parse_response_message(response, "discussion")
                self.chat_history[-1].append(response_message)
                request.append(response_message)

                tool_calls = response_message.tool_calls
                try:
                    codes = _extract_code(response_message.content)
                except Exception as error:
                    record_event(
                        "llm_parse", agent_name=self.agent_name, phase="discussion",
                        call_id=self._trace_call_id, status="error",
                        parse_stage="code_blocks", raw_content=response_message.content,
                        error_type=type(error).__name__, error_message=str(error),
                        handling="rethrow",
                    )
                    raise
                if self.tool_calls_enable and tool_calls is not None:
                    for tool_call in tool_calls:
                        tool_call_result = {
                            "tool_call_id": tool_call.id,
                            "role": "tool",
                            "name": tool_call.function.name,
                            "content": self.execute_action(
                                tool_call.function.name,
                                self._parse_tool_arguments(tool_call, "discussion"),
                            ),
                        }
                        self.chat_history[-1].append(tool_call_result)
                        request.append(tool_call_result)

                elif self.code_execution and codes:
                    # execution_results = []
                    # for code_block, code_type in codes:
                    #     execution_results.append(self.interpreter.run(code_block, code_type))
                    
                    # result_content = "\n".join(execution_results)
                    # print("============Codes============")
                    # for idx, code in enumerate(codes):
                    #     print(f"Code block {idx}: {code[0]}")
                    
                    merged_code = "\n".join([code_block for code_block, code_type in codes
                                             if code_type in self.interpreter._CODE_TYPE_MAPPING
                                             ])
                    try:
                        result_content = self.interpreter.run(merged_code, "python")
                    except Exception as error:
                        record_event(
                            "code_error", agent_name=self.agent_name,
                            phase="discussion", code=merged_code,
                            language="python", error_type=type(error).__name__,
                            error_message=str(error),
                        )
                        raise
                    record_event(
                        "code_execution", agent_name=self.agent_name,
                        phase="discussion", code=merged_code,
                        language="python", result=result_content,
                    )
                
                    print("============merged code============")
                    print(merged_code)
                    print("============Results============")
                    print(result_content)
                    print("==============================")
                    result_message = {"role": "user", "content": result_content}
                    self.chat_history[-1].append(result_message)
                    request.append(result_message)
                else:
                    record_event(
                        "llm_parse", agent_name=self.agent_name, phase="discussion",
                        call_id=self._trace_call_id, status="success",
                        parse_stage="text_content", parsed_output=response_message.content,
                        handling="return_text",
                    )
                    if self.save_on_each_chat:
                        self.save_chat_history(self.logging_file)
                    return response_message.content
        elif crab_planning:
            while True:
                response = self._create_completion(
                    request, phase="planning",
                )
                self.token_usage += response.usage.total_tokens

                response_message = self._parse_response_message(response, "planning")
                self.chat_history[-1].append(response_message)
                request.append(response_message)

                record_event(
                    "llm_parse", agent_name=self.agent_name, phase="planning",
                    call_id=self._trace_call_id, status="success",
                    parse_stage="text_content", parsed_output=response_message.content,
                    handling="return_text",
                )
                if self.save_on_each_chat:
                    self.save_chat_history(self.logging_file)
                return response_message.content
        else:
            response = self._create_completion(
                request, phase="execution",
                tools=[
                    {"type": "function", "function": action} for action in self.actions
                ],
                tool_choice="required",
            )
            self.token_usage += response.usage.total_tokens

            response_message = self._parse_response_message(response, "execution")
            self.chat_history[-1].append(response_message)
            try:
                tool_calls = response_message.tool_calls
                for idx, tool_call in enumerate(tool_calls):
                    self.chat_history[-1].append(
                        {
                            "tool_call_id": tool_call.id,
                            "role": "tool",
                            "name": tool_call.function.name,
                            "content": "Success"
                            if idx == 0
                            else "Didn't execute because only the first tool call is executed",
                        }
                    )  # extend conversation with function response
                    record_event(
                        "tool_response_placeholder", agent_name=self.agent_name,
                        phase="execution", call_id=self._trace_call_id,
                        tool_call_id=tool_call.id, action_name=tool_call.function.name,
                        content=self.chat_history[-1][-1]["content"],
                        is_placeholder=True, actual_execution_observed=False,
                    )

                # if len(tool_calls) != 1:
                #     print(tool_calls)
                #     raise RuntimeError("agent output more than one action per step.")
                call = tool_calls[0]
            except Exception as error:
                record_event(
                    "llm_parse", agent_name=self.agent_name,
                    phase="execution", call_id=self._trace_call_id,
                    status="error", parse_stage="tool_calls", raw_message=response_message,
                    error_type=type(error).__name__, error_message=str(error),
                    handling="rethrow",
                )
                raise
            parameters = self._parse_tool_arguments(call, "execution")
            record_event(
                "tool_selection", agent_name=self.agent_name, phase="execution",
                action_name=call.function.name, parameters=parameters,
                tool_call_id=call.id, call_id=self._trace_call_id,
            )
            
            if self.save_on_each_chat:
                self.save_chat_history(self.logging_file)
            
            return (call.function.name, parameters)

        # function_call_res = []
        # for call in tool_calls:
        #     action_name = call.function.name
        #     action = self.action_map[action_name]
        #     parameters = action.parameters(
        #         json.loads(call.function.arguments)
        #     ).model_dump()
        #     function_call_res.append((action_name, parameters))
        # return function_call_res

    def _convert_action_to_schema(self, action_space):
        self.actions = []
        for action in action_space:
            new_action = action.to_openai_json_schema()
            self.actions.append(new_action)


def _extract_code(content) -> list[tuple[str, str]]:
    codes = []
    texts = []

    lines = content.split("\n")
    idx = 0
    start_idx = 0
    while idx < len(lines):
        while idx < len(lines) and (not lines[idx].lstrip().startswith("```")):
            idx += 1
        text = "\n".join(lines[start_idx:idx]).strip()
        texts.append(text)

        if idx >= len(lines):
            break

        code_type = lines[idx].strip()[3:].strip()
        idx += 1
        start_idx = idx
        while not lines[idx].lstrip().startswith("```"):
            idx += 1
        code = "\n".join(lines[start_idx:idx]).strip()
        codes.append((code, code_type))

        idx += 1
        start_idx = idx

    return codes
