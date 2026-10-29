"""Human-referenced, episode-level system anomaly detection evaluation.

Runtime anomalies and failure attribution are different targets.  Neither native
success nor a previous causal classification is converted into a runtime label.
Precision/recall use fractions (0..1); unreviewed labels and incomplete detector
coverage keep the formal metrics null while exposing the observed counts.
"""

import csv
from pathlib import Path


IDENTITY_FIELDS = ("dataset", "scene_id", "episode_id")
ANNOTATION_FIELDS = (
    *IDENTITY_FIELDS,
    "runtime_system_error",
    "onset_step",
    "offset_step",
    "reason",
    "reviewer",
    "report_path",
    "native_success",
)


def _identity(row):
    result = []
    for field in IDENTITY_FIELDS:
        value = row.get(field)
        if value is None or not str(value).strip():
            raise ValueError("Missing episode identity field: " + field)
        result.append(str(value).strip())
    return tuple(result)


def _label(value):
    if value is None or value == "" or value == "unknown":
        return None
    if not isinstance(value, bool) and value in (0, 1, "0", "1"):
        return int(value)
    raise ValueError("runtime_system_error must be 0, 1, blank or unknown")


def _step(value, field):
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        raise ValueError(field + " must be a nonnegative integer")
    try:
        integer = int(value)
    except (ValueError, TypeError, OverflowError):
        raise ValueError(field + " must be a nonnegative integer") from None
    if integer < 0 or str(integer) != str(value):
        raise ValueError(field + " must be a nonnegative integer")
    return integer


def _native_success(value):
    if value is None or value == "":
        return None
    if value is True or value == 1 or value in ("1", "true", "True"):
        return True
    if value is False or value == 0 or value in ("0", "false", "False"):
        return False
    raise ValueError("native_success must be a boolean, 0, 1 or blank")


def _reference(row):
    result = dict(row)
    result.update(zip(IDENTITY_FIELDS, _identity(row)))
    result["runtime_system_error"] = _label(row.get("runtime_system_error"))
    reviewer = row.get("reviewer") or ""
    if not isinstance(reviewer, str):
        raise ValueError("reviewer must be text")
    result["reviewer"] = reviewer.strip()
    if result["runtime_system_error"] is not None and not result["reviewer"]:
        raise ValueError("A confirmed runtime label requires a human reviewer")
    onset = _step(row.get("onset_step"), "onset_step")
    offset = _step(row.get("offset_step"), "offset_step")
    if onset is not None and offset is not None and offset < onset:
        raise ValueError("offset_step cannot precede onset_step")
    result["onset_step"], result["offset_step"] = onset, offset
    result["native_success"] = _native_success(row.get("native_success"))
    return result


def _references(rows):
    result = {}
    for row in rows:
        normalized = _reference(row)
        key = _identity(normalized)
        if key in result:
            raise ValueError("Duplicate reference episode: " + repr(key))
        result[key] = normalized
    return result


def _prediction_complete(row):
    """Score an aggregate boolean only when its query coverage is complete.

    API, parse and budget errors are missing data rather than negative results.
    Previously saved aggregate rows remain supported independently of the
    current model response schema.
    """
    invalid_status = {
        "error", "failed", "partial", "pending", "not_run", "api_error",
        "parse_error", "budget_exhausted", "incomplete",
    }
    return (
        row.get("complete") is True
        and isinstance(row.get("detected"), bool)
        and not row.get("error")
        and row.get("status") not in invalid_status
    )


def score_episodes(references, predictions, scope="all_episodes"):
    """Score one aggregate prediction per episode and method against humans.

    ``references`` require dataset/scene_id/episode_id, runtime_system_error
    (0/1 or unknown), and reviewer for reviewed labels. ``predictions`` require
    the same identity, method, detected (bool or None) and complete (bool).
    A prediction may hold many alarms; it still contributes at most one TP/FP.
    Duplicate aggregate rows are rejected so accidental reruns cannot inflate
    episode counts. Missing or incomplete predictions never become negatives.

    Formal scores are returned only when every episode in the selected scope
    has a reviewed label and completed prediction. Counts are also provided for
    the reviewed, completed subset when coverage is partial.
    """
    if scope not in ("all_episodes", "failed_episodes_only"):
        raise ValueError("Unknown evaluation scope: " + str(scope))
    all_references = _references(references)
    selected = {}
    for key, row in all_references.items():
        if scope == "failed_episodes_only":
            if row["native_success"] is None:
                raise ValueError("failed_episodes_only requires native_success")
            if row["native_success"]:
                continue
        selected[key] = row

    methods = {}
    for row in predictions:
        key = _identity(row)
        if key not in all_references:
            raise ValueError("Prediction has no reference episode: " + repr(key))
        method = row.get("method")
        if not isinstance(method, str) or not method.strip():
            raise ValueError("Prediction method must be nonempty text")
        method = method.strip()
        grouped = methods.setdefault(method, {})
        if key in grouped:
            raise ValueError("Duplicate prediction for method/episode: " + repr((method, key)))
        if row.get("detected") is not None and not isinstance(row["detected"], bool):
            raise ValueError("detected must be a boolean or null")
        if not isinstance(row.get("complete"), bool):
            raise ValueError("Prediction complete must be a boolean")
        _step(row.get("first_alarm_step"), "first_alarm_step")
        grouped[key] = row

    reviewed = sum(row["runtime_system_error"] is not None for row in selected.values())
    unknown = len(selected) - reviewed
    reference_counts = {
        "episodes": len(selected),
        "reviewed": reviewed,
        "unknown_labels": unknown,
        "positives": sum(row["runtime_system_error"] == 1 for row in selected.values()),
        "negatives": sum(row["runtime_system_error"] == 0 for row in selected.values()),
    }
    output = {}
    for method, grouped in sorted(methods.items()):
        counts = dict(tp=0, fp=0, fn=0, tn=0)
        missing = incomplete = complete = scored = 0
        for key, reference in selected.items():
            prediction = grouped.get(key)
            if prediction is None:
                missing += 1
                continue
            if not _prediction_complete(prediction):
                incomplete += 1
                continue
            complete += 1
            label = reference["runtime_system_error"]
            if label is None:
                continue
            scored += 1
            bucket = (
                "tp" if prediction["detected"] else "fn"
            ) if label == 1 else (
                "fp" if prediction["detected"] else "tn"
            )
            counts[bucket] += 1
        if not selected or unknown:
            status = "awaiting_manual_labels"
        elif missing or incomplete:
            status = "partial"
        else:
            status = "complete"
        precision_denominator = counts["tp"] + counts["fp"]
        recall_denominator = counts["tp"] + counts["fn"]
        output[method] = {
            "status": status,
            **counts,
            "precision": counts["tp"] / precision_denominator
            if status == "complete" and precision_denominator else None,
            "recall": counts["tp"] / recall_denominator
            if status == "complete" and recall_denominator else None,
            "coverage": {
                **reference_counts,
                "complete_predictions": complete,
                "missing_predictions": missing,
                "incomplete_predictions": incomplete,
                "scored": scored,
            },
            "count_scope": "reviewed_labels_with_complete_predictions",
        }
    return {"scope": scope, "reference_counts": reference_counts, "methods": output}


def export_csv(path, rows, fieldnames=None):
    """Write UTF-8 BOM CSV so Chinese text opens correctly in Excel."""
    rows = list(rows)
    if fieldnames is None:
        fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_annotation_template(path, episodes):
    """Create blank runtime labels, including successes; preserve existing work.

    Explicit existing paths are rejected to avoid overwriting an annotator's
    answers during a repeated detector run. Other old labels/reasons/reviewers
    from ``episodes`` are deliberately never copied.
    """
    path = Path(path)
    if path.exists():
        raise FileExistsError("Annotation file already exists: " + str(path))
    rows = []
    seen = set()
    for episode in episodes:
        key = _identity(episode)
        if key in seen:
            raise ValueError("Duplicate template episode: " + repr(key))
        seen.add(key)
        row = dict.fromkeys(ANNOTATION_FIELDS, "")
        row.update(zip(IDENTITY_FIELDS, key))
        row["report_path"] = episode.get("report_path") or ""
        success = _native_success(episode.get("native_success"))
        row["native_success"] = "" if success is None else int(success)
        rows.append(row)
    export_csv(path, rows, ANNOTATION_FIELDS)


def load_annotations(path):
    """Read human annotations with strict identity, label and interval checks."""
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {*IDENTITY_FIELDS, "runtime_system_error", "reviewer"}
        if not required.issubset(reader.fieldnames or ()):
            raise ValueError("Annotation CSV is missing required columns")
        if len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise ValueError("Annotation CSV has duplicate columns")
        rows = list(reader)
    if any(None in row for row in rows):
        raise ValueError("Annotation CSV row has more values than columns")
    return list(_references(rows).values())
