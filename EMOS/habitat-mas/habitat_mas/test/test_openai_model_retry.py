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
