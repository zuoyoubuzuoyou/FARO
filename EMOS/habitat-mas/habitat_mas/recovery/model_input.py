"""Build the source-repair model view without changing runtime entrypoint data."""
from copy import deepcopy

from .context_compaction import compact_runtime_context
from .source_view import extract_method_sources


def prepare_model_context(context):
    value = deepcopy(context)
    if not isinstance(value, dict):
        raise ValueError("Source recovery input must be a JSON object")
    code = value.get("code")
    sources = code.get("source_files") if isinstance(code, dict) else None
    if isinstance(sources, dict) and sources:
        views = {item.get("view") if isinstance(item, dict) else None for item in sources.values()}
        if views - {None, "methods-v1"}:
            raise ValueError("Unsupported source input view; complete files or methods-v1 are required")
        if views != {"methods-v1"}:
            if "methods-v1" in views:
                raise ValueError("Source input cannot mix full files and method views")
            code["source_files"] = extract_method_sources(sources, value)
        layout = value.setdefault("input_layout", {})
        if not isinstance(layout, dict):
            raise ValueError("Source input_layout must be an object")
        layout["source_view"] = {
            "format": "methods-v1",
            "content": "Selected complete methods and necessary dependency context; not a complete compilable file.",
            "sha256": "The original FULL FILE hash for base_sha256; content/snippet hashes do not replace it.",
            "methods": "Index into the sole source content, with original line ranges and fragment hashes.",
            "symbol_index": "Methods whose bodies are not sent; omitted bodies are not missing implementations.",
        }
    return compact_runtime_context(value)
