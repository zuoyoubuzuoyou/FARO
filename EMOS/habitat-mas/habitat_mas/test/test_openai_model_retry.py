import unittest
from unittest.mock import MagicMock, patch

import httpx
import openai

from habitat_mas.utils.models import OpenAIModel


class OpenAIModelRetryTest(unittest.TestCase):
    @staticmethod
    def _model() -> OpenAIModel:
        model = OpenAIModel.__new__(OpenAIModel)
        model.client = MagicMock()
        model.max_retries = 10
        model.max_requests_per_minute = 50
        model.service_backoff_seconds = 5.0
        model.max_service_backoff_seconds = 60.0
        model.rate_limit_cooldown_seconds = 65.0
        return model

    @staticmethod
    def _response(status_code: int, **headers: str) -> httpx.Response:
        return httpx.Response(
            status_code,
            headers=headers,
            request=httpx.Request(
                "POST", "https://example.test/v1/chat/completions"
            ),
        )

    def _connection_model(self, error_type, failures):
        requests = []

        def handle_request(request):
            requests.append(request)
            if len(requests) <= failures:
                raise error_type("temporary transport failure", request=request)
            return httpx.Response(
                200,
                json={
                    "id": "test-completion",
                    "object": "chat.completion",
                    "created": 0,
                    "model": "test-model",
                    "choices": [{
                        "index": 0,
                        "message": {"role": "assistant", "content": "ready"},
                        "finish_reason": "stop",
                    }],
                    "usage": {
                        "prompt_tokens": 1,
                        "completion_tokens": 1,
                        "total_tokens": 2,
                    },
                },
            )

        model = self._model()
        model.client = openai.OpenAI(
            api_key="test-key",
            base_url="https://example.test/v1",
            max_retries=0,
            http_client=httpx.Client(
                transport=httpx.MockTransport(handle_request)
            ),
        )
        self.addCleanup(model.client.close)
        return model, requests

    @patch("habitat_mas.utils.models._wait_for_request_slot")
    @patch("habitat_mas.utils.models.time.sleep")
    def test_retries_transport_timeout(
        self, sleep: MagicMock, wait_for_slot: MagicMock
    ) -> None:
        model, requests = self._connection_model(httpx.ReadTimeout, failures=1)

        result = model._chat_completion(model="test-model", messages=[])

        self.assertEqual(result.choices[0].message.content, "ready")
        self.assertEqual(len(requests), 2)
        self.assertEqual(requests[0].content, requests[1].content)
        sleep.assert_called_once_with(5.0)
        self.assertEqual(wait_for_slot.call_count, 2)

    @patch("habitat_mas.utils.models._wait_for_request_slot")
    @patch("habitat_mas.utils.models.time.sleep")
    def test_retries_transport_connection_error(
        self, sleep: MagicMock, wait_for_slot: MagicMock
    ) -> None:
        model, requests = self._connection_model(httpx.ConnectError, failures=1)

        result = model._chat_completion(model="test-model", messages=[])

        self.assertEqual(result.choices[0].message.content, "ready")
        self.assertEqual(len(requests), 2)
        sleep.assert_called_once_with(5.0)
        self.assertEqual(wait_for_slot.call_count, 2)

    @patch("habitat_mas.utils.models._wait_for_request_slot")
    @patch("habitat_mas.utils.models.time.sleep")
    def test_timeouts_stop_after_configured_retry_budget(
        self, sleep: MagicMock, wait_for_slot: MagicMock
    ) -> None:
        model, requests = self._connection_model(httpx.ReadTimeout, failures=10)
        model.max_retries = 2

        with self.assertRaises(openai.APITimeoutError) as raised:
            model._chat_completion(model="test-model", messages=[])

        self.assertIsInstance(raised.exception.__cause__, httpx.ReadTimeout)
        self.assertEqual(len(requests), 3)
        self.assertEqual(
            [call.args for call in sleep.call_args_list], [(5.0,), (10.0,)]
        )
        self.assertEqual(wait_for_slot.call_count, 3)

    @patch("habitat_mas.utils.models._wait_for_request_slot")
    @patch("habitat_mas.utils.models.time.sleep")
    def test_zero_retry_budget_propagates_transport_error(
        self, sleep: MagicMock, wait_for_slot: MagicMock
    ) -> None:
        model, requests = self._connection_model(httpx.ReadTimeout, failures=1)
        model.max_retries = 0

        with self.assertRaises(openai.APITimeoutError):
            model._chat_completion(model="test-model", messages=[])

        self.assertEqual(len(requests), 1)
        sleep.assert_not_called()
        wait_for_slot.assert_called_once_with(50)

    @patch("habitat_mas.utils.models._wait_for_request_slot")
    @patch("habitat_mas.utils.models.time.sleep")
    def test_timeout_backoff_is_capped(
        self, sleep: MagicMock, _wait_for_slot: MagicMock
    ) -> None:
        model, requests = self._connection_model(httpx.ReadTimeout, failures=4)
        model.max_service_backoff_seconds = 30.0

        result = model._chat_completion(model="test-model", messages=[])

        self.assertEqual(result.choices[0].message.content, "ready")
        self.assertEqual(len(requests), 5)
        self.assertEqual(
            [call.args for call in sleep.call_args_list],
            [(5.0,), (10.0,), (20.0,), (30.0,)],
        )

    @patch("habitat_mas.utils.models._wait_for_request_slot")
    @patch("habitat_mas.utils.models.time.sleep")
    def test_retries_503_with_exponential_backoff(
        self, sleep: MagicMock, wait_for_slot: MagicMock
    ) -> None:
        model = self._model()
        success = object()
        model.client.chat.completions.create.side_effect = [
            openai.InternalServerError(
                "unavailable", response=self._response(503), body={}
            ),
            openai.InternalServerError(
                "unavailable", response=self._response(503), body={}
            ),
            success,
        ]

        result = model._chat_completion(model="gpt-5.5", messages=[])

        self.assertIs(result, success)
        self.assertEqual(sleep.call_args_list[0].args, (5.0,))
        self.assertEqual(sleep.call_args_list[1].args, (10.0,))
        self.assertEqual(wait_for_slot.call_count, 3)

    @patch("habitat_mas.utils.models._wait_for_request_slot")
    @patch("habitat_mas.utils.models.time.sleep")
    def test_retries_524_with_exponential_backoff(
        self, sleep: MagicMock, wait_for_slot: MagicMock
    ) -> None:
        model = self._model()
        success = object()
        model.client.chat.completions.create.side_effect = [
            openai.InternalServerError(
                "origin timeout", response=self._response(524), body={}
            ),
            success,
        ]

        result = model._chat_completion(model="gpt-5.5", messages=[])

        self.assertIs(result, success)
        sleep.assert_called_once_with(5.0)
        self.assertEqual(wait_for_slot.call_count, 2)

    @patch("habitat_mas.utils.models._wait_for_request_slot")
    @patch("habitat_mas.utils.models.time.sleep")
    def test_429_waits_for_retry_after_or_full_window(
        self, sleep: MagicMock, _wait_for_slot: MagicMock
    ) -> None:
        model = self._model()
        success = object()
        model.client.chat.completions.create.side_effect = [
            openai.RateLimitError(
                "limited",
                response=self._response(429, **{"retry-after": "70"}),
                body={},
            ),
            success,
        ]

        result = model._chat_completion(model="gpt-5.5", messages=[])

        self.assertIs(result, success)
        sleep.assert_called_once_with(70.0)

    @patch("habitat_mas.utils.models._wait_for_request_slot")
    @patch("habitat_mas.utils.models.time.sleep")
    def test_does_not_retry_expired_credentials(
        self, sleep: MagicMock, wait_for_slot: MagicMock
    ) -> None:
        model = self._model()
        model.client.chat.completions.create.side_effect = (
            openai.AuthenticationError(
                "expired", response=self._response(401), body={}
            )
        )

        with self.assertRaises(openai.AuthenticationError):
            model._chat_completion(model="gpt-5.5", messages=[])

        sleep.assert_not_called()
        wait_for_slot.assert_called_once_with(50)


if __name__ == "__main__":
    unittest.main()
