"""Consistent output names for clean and fault-injection runs."""

from __future__ import annotations

import re


def safe_output_component(value: object) -> str:
    """Return a filesystem-safe representation of an output-name component."""

    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")


def output_run_stem(run_label: object, output_type: object) -> str:
    """Prefix a run label with its actual clean/fault type.

    Internal names beginning with ``agent_`` are shortened for human-facing
    filenames. Unknown future fault names are preserved, so callers
    automatically get the same ``<fault>_<label>`` convention.
    """

    raw_type = str(output_type)
    canonical_type = raw_type.removeprefix("agent_")
    safe_type = safe_output_component(canonical_type) or "unknown"
    safe_label = safe_output_component(run_label)
    if not safe_label:
        return safe_type
    if safe_label == safe_type or safe_label.startswith(
        (f"{safe_type}_", f"{safe_type}-", f"{safe_type}.")
    ):
        return safe_label
    return f"{safe_type}_{safe_label}"
