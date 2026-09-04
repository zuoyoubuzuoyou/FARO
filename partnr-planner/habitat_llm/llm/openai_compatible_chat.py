#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree

import json
import os
import socket
import urllib.error
import urllib.parse
import urllib.request
from typing import Dict, List, Optional, Sequence, Union

from omegaconf import DictConfig

from habitat_llm.llm.base_llm import BaseLLM, Prompt


StopSequence = Optional[Union[str, Sequence[str]]]


def _generate_message(multimodal_prompt, image_detail="auto"):
    """Convert PARTNR's multimodal prompt representation to chat content."""
    content = []
    for prompt_type, prompt_value in multimodal_prompt:
        if prompt_type == "text":
            message_item = {"type": "text", "text": prompt_value}
        else:
            message_item = {
                "type": "image_url",
                "image_url": {
                    "url": prompt_value,
                    "detail": image_detail,
                },
            }
        content.append(message_item)
    return {"role": "user", "content": content}


class OpenAICompatibleChat(BaseLLM):
    """Chat Completions backend for an OpenAI-compatible ``/v1`` API.

    This backend intentionally uses Python's standard library instead of the
    OpenAI SDK.  It therefore remains independent of the OpenAI/httpx package
    versions installed in the Habitat environment.
    """

    _REQUEST_PARAMETER_NAMES = (
        "max_tokens",
        "max_completion_tokens",
        "temperature",
        "top_p",
        "frequency_penalty",
        "presence_penalty",
        "reasoning_effort",
    )

    def __init__(self, conf: DictConfig):
        super().__init__(conf)

        self.api_key = self._required_environment_variable("OPENAI_API_KEY")
        base_url = self._required_environment_variable("OPENAI_BASE_URL")
        self.chat_completions_url = self._chat_completions_url(base_url)

        self._validate_conf()
        self.verbose = self.llm_conf.verbose
        self.keep_message_history = self.llm_conf.keep_message_history
        self.message_history: List[Dict] = []
        self.response = ""

    @staticmethod
    def _required_environment_variable(name: str) -> str:
        value = os.getenv(name, "").strip()
        if not value:
            raise ValueError(f"No {name} provided")
        return value

    @staticmethod
    def _chat_completions_url(base_url: str) -> str:
        normalized_base_url = base_url.rstrip("/")
        parsed_url = urllib.parse.urlparse(normalized_base_url)
        if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
            raise ValueError(
                "OPENAI_BASE_URL must be an absolute http(s) URL, for example "
                "http://host:port/v1"
            )
        if parsed_url.query or parsed_url.fragment:
            raise ValueError("OPENAI_BASE_URL must not contain a query or fragment")
        if normalized_base_url.endswith("/chat/completions"):
            return normalized_base_url
        return f"{normalized_base_url}/chat/completions"

    def _validate_conf(self):
        if self.generation_params.get("stream", False):
            raise ValueError("Streaming is not supported")

        model = str(self.generation_params.get("model", "")).strip()
        if not model:
            raise ValueError("generation_params.model must be provided")

        request_timeout = self.generation_params.get("request_timeout", 120)
        if request_timeout is None or float(request_timeout) <= 0:
            raise ValueError("generation_params.request_timeout must be positive")

    def _request_payload(
        self,
        messages: List[Dict],
        stop: StopSequence,
        max_length: Optional[int],
    ) -> Dict:
        payload = {
            "model": str(self.generation_params.model),
            "messages": messages,
            "stream": False,
        }

        for parameter_name in self._REQUEST_PARAMETER_NAMES:
            value = self.generation_params.get(parameter_name)
            if value is not None:
                payload[parameter_name] = value

        if max_length is not None:
            if max_length <= 0:
                raise ValueError("max_length must be positive")
            payload.pop("max_completion_tokens", None)
            payload["max_tokens"] = max_length

        if self.llm_conf.get("send_stop_to_api", False) and stop:
            payload["stop"] = stop

        return payload

    def _post_chat_completion(self, payload: Dict, timeout: float) -> Dict:
        request = urllib.request.Request(
            url=self.chat_completions_url,
            data=json.dumps(payload).encode("utf-8"),
            method="POST",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
        )

        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                response_body = response.read().decode("utf-8")
        except urllib.error.HTTPError as error:
            error_body = error.read().decode("utf-8", errors="replace")
            error_body = error_body.replace(self.api_key, "[REDACTED]")[:4000]
            raise RuntimeError(
                "OpenAI-compatible Chat Completions request failed with HTTP "
                f"{error.code}: {error_body}"
            ) from error
        except (urllib.error.URLError, TimeoutError, socket.timeout) as error:
            raise RuntimeError(
                f"OpenAI-compatible Chat Completions request failed: {error}"
            ) from error

        try:
            result = json.loads(response_body)
        except json.JSONDecodeError as error:
            raise RuntimeError(
                "OpenAI-compatible Chat Completions response was not valid JSON"
            ) from error

        if not isinstance(result, dict):
            raise RuntimeError(
                "OpenAI-compatible Chat Completions response must be a JSON object"
            )
        return result

    @staticmethod
    def _text_from_response(result: Dict) -> str:
        choices = result.get("choices")
        if not isinstance(choices, list) or not choices:
            raise RuntimeError(
                "OpenAI-compatible Chat Completions response contained no choices"
            )

        first_choice = choices[0]
        message = (
            first_choice.get("message") if isinstance(first_choice, dict) else None
        )
        content = message.get("content") if isinstance(message, dict) else None

        if isinstance(content, str) and content:
            return content

        if isinstance(content, list):
            text_parts = []
            for part in content:
                if isinstance(part, dict) and isinstance(part.get("text"), str):
                    text_parts.append(part["text"])
            if text_parts:
                return "".join(text_parts)

        raise RuntimeError(
            "OpenAI-compatible Chat Completions response contained no text content"
        )

    @staticmethod
    def _truncate_at_stop(text: str, stop: StopSequence) -> str:
        if not stop:
            return text
        stop_sequences = [stop] if isinstance(stop, str) else list(stop)
        stop_positions = [
            text.find(sequence)
            for sequence in stop_sequences
            if sequence and text.find(sequence) >= 0
        ]
        if not stop_positions:
            return text
        return text[: min(stop_positions)]

    def generate(
        self,
        prompt: Prompt,
        stop: StopSequence = None,
        max_length: Optional[int] = None,
        generation_args=None,
        request_timeout: Optional[float] = None,
    ) -> str:
        """Generate one non-streaming Chat Completions response."""
        messages = self.message_history.copy()
        if not messages:
            messages.append(
                {"role": "system", "content": self.llm_conf.system_message}
            )

        if isinstance(prompt, str):
            messages.append({"role": "user", "content": prompt})
        else:
            messages.append(_generate_message(prompt, image_detail="low"))

        payload = self._request_payload(messages, stop, max_length)
        timeout = (
            float(request_timeout)
            if request_timeout is not None
            else float(self.generation_params.get("request_timeout", 120))
        )
        if timeout <= 0:
            raise ValueError("request_timeout must be positive")

        result = self._post_chat_completion(payload, timeout)
        text_response = self._text_from_response(result)
        self.response = text_response

        if self.keep_message_history:
            self.message_history = messages.copy()
            self.message_history.append(
                {"role": "assistant", "content": text_response}
            )

        return self._truncate_at_stop(text_response, stop)
