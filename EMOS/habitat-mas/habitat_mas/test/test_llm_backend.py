import pytest

from habitat_mas.utils.llm_backend import get_llm_backend


@pytest.mark.parametrize(
    ("configured", "expected"),
    [
        (None, "gpt"),
        ("gpt", "gpt"),
        (" GPT ", "gpt"),
        ("qwen", "qwen"),
        (" QwEn ", "qwen"),
    ],
)
def test_backend_profile_normalizes_supported_families(
    monkeypatch, configured, expected
):
    if configured is None:
        monkeypatch.delenv("EMOS_LLM_BACKEND", raising=False)
    else:
        monkeypatch.setenv("EMOS_LLM_BACKEND", configured)

    assert get_llm_backend() == expected


def test_backend_profile_rejects_unknown_family(monkeypatch):
    monkeypatch.setenv("EMOS_LLM_BACKEND", "llama")

    with pytest.raises(ValueError, match="llama"):
        get_llm_backend()
