from unittest.mock import patch

import pytest

from habitat_mas.utils.models import OpenAIModel


@pytest.mark.parametrize(
    ("backend", "configured_model", "constructor_model", "expected_backend", "expected_model"),
    [
        (None, None, "gpt-5.5", "gpt", "gpt-5.5"),
        ("qwen", None, "gpt-5.5", "qwen", "gpt-5.5"),
        (None, "qwen-custom", "gpt-5.5", "gpt", "qwen-custom"),
        ("qwen", " qwen-3.5-9b ", "gpt-5.5", "qwen", "qwen-3.5-9b"),
    ],
)
@patch("habitat_mas.utils.models.openai.OpenAI")
def test_request_model_is_independent_from_backend_profile(
    openai_client,
    monkeypatch,
    backend,
    configured_model,
    constructor_model,
    expected_backend,
    expected_model,
):
    if backend is None:
        monkeypatch.delenv("EMOS_LLM_BACKEND", raising=False)
    else:
        monkeypatch.setenv("EMOS_LLM_BACKEND", backend)
    if configured_model is None:
        monkeypatch.delenv("EMOS_LLM_MODEL", raising=False)
    else:
        monkeypatch.setenv("EMOS_LLM_MODEL", configured_model)

    model = OpenAIModel(
        system_prompt="test",
        action_space=[],
        model=constructor_model,
        save_on_each_chat=False,
    )

    assert model.backend == expected_backend
    assert model.model == expected_model
    openai_client.assert_called_once_with()
