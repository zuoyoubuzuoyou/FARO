#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree

import io
import json
import urllib.error

import pytest
from omegaconf import OmegaConf

from habitat_llm.llm.openai_compatible_chat import OpenAICompatibleChat


class _FakeHTTPResponse:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception, traceback):
        return False

    def read(self):
        return json.dumps(self.payload).encode("utf-8")


def _config(**overrides):
    config = {
        "verbose": False,
        "system_message": "System instruction",
        "keep_message_history": False,
        "send_stop_to_api": False,
        "generation_params": {
            "model": "test-model",
            "max_tokens": 250,
            "stream": False,
            "request_timeout": 120,
        },
    }
    return OmegaConf.merge(OmegaConf.create(config), OmegaConf.create(overrides))


def _set_credentials(monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "http://gateway.test:8317/v1/")
    monkeypatch.setenv("OPENAI_API_KEY", "test-secret-key")


def test_requires_credentials(monkeypatch):
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        OpenAICompatibleChat(_config())

    monkeypatch.setenv("OPENAI_API_KEY", "test-secret-key")
    with pytest.raises(ValueError, match="OPENAI_BASE_URL"):
        OpenAICompatibleChat(_config())


def test_builds_chat_completions_request(monkeypatch):
    _set_credentials(monkeypatch)
    captured = {}

    def fake_urlopen(request, timeout):
        captured["request"] = request
        captured["timeout"] = timeout
        return _FakeHTTPResponse(
            {"choices": [{"message": {"content": "Action: Wait[]"}}]}
        )

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    llm = OpenAICompatibleChat(_config())

    assert llm.generate("Plan the next step") == "Action: Wait[]"

    request = captured["request"]
    payload = json.loads(request.data.decode("utf-8"))
    assert request.full_url == "http://gateway.test:8317/v1/chat/completions"
    assert request.get_header("Authorization") == "Bearer test-secret-key"
    assert payload == {
        "model": "test-model",
        "messages": [
            {"role": "system", "content": "System instruction"},
            {"role": "user", "content": "Plan the next step"},
        ],
        "stream": False,
        "max_tokens": 250,
    }
    assert captured["timeout"] == 120.0


def test_max_length_and_stop_are_applied_locally(monkeypatch):
    _set_credentials(monkeypatch)
    captured = {}

    def fake_urlopen(request, timeout):
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        return _FakeHTTPResponse(
            {
                "choices": [
                    {
                        "message": {
                            "content": "Agent_0_Action: Wait[]\nAssigned!extra"
                        }
                    }
                ]
            }
        )

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    llm = OpenAICompatibleChat(_config())

    response = llm.generate("Plan", stop="Assigned!", max_length=64)

    assert response == "Agent_0_Action: Wait[]\n"
    assert captured["payload"]["max_tokens"] == 64
    assert "stop" not in captured["payload"]


def test_message_history_is_optional(monkeypatch):
    _set_credentials(monkeypatch)
    captured_payloads = []

    def fake_urlopen(request, timeout):
        captured_payloads.append(json.loads(request.data.decode("utf-8")))
        return _FakeHTTPResponse(
            {"choices": [{"message": {"content": "Assistant response"}}]}
        )

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    llm = OpenAICompatibleChat(_config(keep_message_history=True))

    llm.generate("First prompt")
    llm.generate("Second prompt")

    assert captured_payloads[1]["messages"] == [
        {"role": "system", "content": "System instruction"},
        {"role": "user", "content": "First prompt"},
        {"role": "assistant", "content": "Assistant response"},
        {"role": "user", "content": "Second prompt"},
    ]


def test_http_error_redacts_api_key(monkeypatch):
    _set_credentials(monkeypatch)

    def fake_urlopen(request, timeout):
        raise urllib.error.HTTPError(
            request.full_url,
            401,
            "Unauthorized",
            hdrs=None,
            fp=io.BytesIO(b'{"error":"test-secret-key is invalid"}'),
        )

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    llm = OpenAICompatibleChat(_config())

    with pytest.raises(RuntimeError, match="HTTP 401") as error:
        llm.generate("Plan")

    assert "test-secret-key" not in str(error.value)
    assert "[REDACTED]" in str(error.value)


def test_rejects_response_without_text(monkeypatch):
    _set_credentials(monkeypatch)

    def fake_urlopen(request, timeout):
        return _FakeHTTPResponse({"choices": []})

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    llm = OpenAICompatibleChat(_config())

    with pytest.raises(RuntimeError, match="contained no choices"):
        llm.generate("Plan")
