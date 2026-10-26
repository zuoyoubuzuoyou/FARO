"""Model selection reaches the real request builder without network access."""

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from openai.types.chat import ChatCompletion

from habitat_mas.utils import models


class RequestTransport:
    def __init__(self):
        self.requests = []

    def create(self, **kwargs):
        self.requests.append(kwargs)
        return ChatCompletion(
            id="model-selection-test", object="chat.completion", created=0,
            model="test-response-model",
            choices=[{
                "index": 0, "finish_reason": "stop",
                "message": {"role": "assistant", "content": "plan"},
            }],
            usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        )


class LLMModelSelectionTests(unittest.TestCase):
    def requested_model(self, environment_model=None, **kwargs):
        transport = RequestTransport()
        client = SimpleNamespace(chat=SimpleNamespace(completions=transport))
        with patch.dict(os.environ):
            os.environ.pop("EMOS_LLM_MODEL", None)
            if environment_model is not None:
                os.environ["EMOS_LLM_MODEL"] = environment_model
            with patch.object(models.openai, "OpenAI", return_value=client):
                model = models.OpenAIModel(
                    "system", [], save_on_each_chat=False, **kwargs,
                )
                self.assertEqual(
                    model.chat("make a plan", crab_planning=True), "plan",
                )
        return transport.requests[0]["model"]

    def test_environment_is_ignored_when_argument_is_omitted(self):
        self.assertEqual(self.requested_model("gpt-6-sol"), "gpt-5.6-sol")

    def test_explicit_argument_takes_precedence_over_environment(self):
        self.assertEqual(
            self.requested_model("environment-model", model="explicit-model"),
            "explicit-model",
        )

    def test_none_argument_uses_default_despite_environment(self):
        self.assertEqual(
            self.requested_model("gpt-6-sol", model=None), "gpt-5.6-sol",
        )

    def test_unconfigured_model_uses_gpt_5_6_sol_default(self):
        self.assertEqual(self.requested_model(), "gpt-5.6-sol")

    def test_whitespace_environment_uses_gpt_5_6_sol_default(self):
        self.assertEqual(self.requested_model(" \t "), "gpt-5.6-sol")

    def test_nonblank_environment_with_whitespace_is_ignored(self):
        self.assertEqual(self.requested_model(" gpt-6-sol \n"), "gpt-5.6-sol")


if __name__ == "__main__":
    unittest.main()
