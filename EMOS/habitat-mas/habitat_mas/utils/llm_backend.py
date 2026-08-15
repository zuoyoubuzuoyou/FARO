import os


SUPPORTED_LLM_BACKENDS = frozenset({"gpt", "qwen"})


def get_llm_backend() -> str:
    """Return the explicitly selected LLM compatibility profile."""
    backend = os.getenv("EMOS_LLM_BACKEND", "gpt").strip().lower()
    if backend not in SUPPORTED_LLM_BACKENDS:
        supported = ", ".join(sorted(SUPPORTED_LLM_BACKENDS))
        raise ValueError(
            f"Unsupported EMOS_LLM_BACKEND: {backend!r}. Expected one of: {supported}."
        )
    return backend
