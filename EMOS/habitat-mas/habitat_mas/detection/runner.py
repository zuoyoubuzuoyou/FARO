"""Replay both monitoring families without exposing outcomes to detectors."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
import csv
import hashlib
import json
import os
from pathlib import Path
from uuid import uuid4

from .evaluation import _references, export_csv, load_annotations, score_episodes, write_annotation_template
from .heuristics import RepeatActionDetector, TeamProgressDetector
from .llm import LLMDetector
from .readers import iter_frames


DEFAULT_CONFIG = {"sample_every": 50, "window": 10, "check_every": 5,
    "repeat_threshold": 10, "min_stall_steps": 100, "reset_budget": 100,
    "model": "gpt-6-sol", "max_llm_calls": None, "timeout": 60}
METHODS = ("repeat_action", "team_progress", "llm")


def _json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2,
        allow_nan=False) + "\n", encoding="utf-8")


def _tail(path):
    size = path.stat().st_size
    if not size:
        raise ValueError("Empty trajectory: " + str(path))
    with path.open("rb") as handle:
        length = min(size, 65536)
        while True:
            handle.seek(size - length)
            lines = handle.read().rstrip(b"\r\n").split(b"\n")
            if len(lines) > 1 or length == size:
                return json.loads(lines[-1])
            length = min(size, length * 2)


def load_dataset_episodes(config_path, repo_root):
    """Use terminal outcomes for reporting only, never observation contexts."""
    root = Path(repo_root).resolve()
    spec = json.loads(Path(config_path).read_text(encoding="utf-8"))
    episodes, seen = [], set()
    for dataset in spec["datasets"]:
        manifest_path = root / dataset["manifest"]
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        entries = manifest["episodes"]
        if len(entries) != dataset["expected_episodes"]:
            raise ValueError("Manifest episode count differs: " + dataset["dataset"])
        for entry in entries:
            raw = manifest_path.parent / (entry.get("raw") or entry["path"])
            with raw.open(encoding="utf-8-sig") as handle:
                first = json.loads(handle.readline())
            final = _tail(raw)
            if final.get("done") is not True:
                raise ValueError("Manifest contains incomplete episode: " + str(raw))
            success = (final.get("metrics") or {}).get("pddl_success")
            if success not in (0, 1):
                raise ValueError("Missing native success: " + str(raw))
            identifier, scene = str(first["episode_id"]), str(first["scene_id"])
            if identifier != str(entry["episode_id"]):
                raise ValueError("Manifest/raw episode identity mismatch")
            key = (dataset["dataset"], scene, identifier)
            if key in seen:
                raise ValueError("Duplicate dataset episode: " + repr(key))
            seen.add(key)
            goals = {}
            facts_path = manifest_path.parent / "data" / ("episode_" + identifier + ".json")
            if facts_path.is_file() and dataset["task"] in ("mobility", "manipulation"):
                facts = json.loads(facts_path.read_text(encoding="utf-8"))
                if dataset["task"] == "mobility":
                    goal = (facts.get("first") or {}).get("goal_position")
                    if goal is not None:
                        goals["TARGET_any_targets|0"] = goal
                else:
                    goals = facts.get("goal_positions") or {}
            episodes.append({"dataset": dataset["dataset"], "task": dataset["task"],
                "episode_id": identifier, "scene_id": scene,
                "native_success": bool(success), "raw_path": str(raw.resolve()),
                "report_path": str((manifest_path.parent / entry["report"]).resolve()),
                "goal_positions": goals,
                "source_sha256": entry.get("raw_sha256", entry.get("sha256")),
                "source_sha256_status": "manifest_expected_not_rehashed"})
    return episodes


def _episode_token(episode):
    scene_hash = hashlib.sha256(episode["scene_id"].encode()).hexdigest()[:10]
    identifier = str(episode["episode_id"])
    if "/" in identifier or "\\" in identifier or identifier in (".", ".."):
        identifier = hashlib.sha256(identifier.encode()).hexdigest()[:16]
    return "episode_" + identifier + "_" + scene_hash


def run_episode(episode, config, output_dir, methods, llm_client=None):
    """Return one aggregate prediction per method; preserve all observed alarms."""
    cfg = {**DEFAULT_CONFIG, **config}
    for key in ("sample_every", "window", "check_every", "repeat_threshold"):
        if isinstance(cfg[key], bool) or not isinstance(cfg[key], int) or cfg[key] < 1:
            raise ValueError(key + " must be a positive integer")
    if not methods or len(set(methods)) != len(methods) or any(m not in METHODS for m in methods):
        raise ValueError("Select unique known detector methods")
    output = Path(output_dir)
    dataset = str(episode["dataset"])
    if "/" in dataset or "\\" in dataset or dataset in (".", ".."):
        raise ValueError("Dataset must be a single path component")
    token = _episode_token(episode)
    factories = {
        "repeat_action": lambda: RepeatActionDetector(cfg["repeat_threshold"]),
        "team_progress": lambda: TeamProgressDetector(window=cfg["window"],
            check_every=cfg["check_every"], min_stall_steps=cfg["min_stall_steps"],
            reset_budget=cfg["reset_budget"]),
        "llm": lambda: LLMDetector(model=cfg["model"], window=cfg["window"],
            check_every=cfg["check_every"], client=llm_client,
            cache_dir=output / "cache", audit_path=output / "llm_requests" / dataset / (token + ".jsonl"),
            max_calls=cfg["max_llm_calls"], timeout=cfg["timeout"]),
    }
    detectors = {method: factories[method]() for method in methods}
    alarms = {method: [] for method in methods}
    failures = {method: 0 for method in methods}
    pending = {"decisions": [], "messages": []}
    samples = frames = 0
    stream_error = None
    try:
        for index, frame in enumerate(iter_frames(episode["raw_path"], episode["task"], episode.get("goal_positions"))):
            frames += 1
            for key in pending:
                for item in frame.get(key) or []:
                    pending[key].append({**item, "step": item.get("step", frame["step"])})
            if index % cfg["sample_every"]:
                continue
            sampled = deepcopy(frame)
            for key in pending:
                sampled[key], pending[key] = pending[key], []
            samples += 1
            for method, detector in detectors.items():
                if alarms[method]:
                    continue
                decision = detector.observe(sampled)
                if decision and decision.get("status") == "system_error":
                    alarms[method].append(decision)
                elif decision and decision.get("status") in ("error", "budget_exhausted"):
                    failures[method] += 1
            if all(alarms.values()):
                break
    except (OSError, ValueError, KeyError, TypeError) as error:
        stream_error = type(error).__name__ + ": " + str(error)
    fingerprint = hashlib.sha256(json.dumps({"identity": [dataset, episode["scene_id"], episode["episode_id"]],
        "source_sha256": episode.get("source_sha256"), "config": cfg,
        "methods": sorted(methods)}, sort_keys=True).encode()).hexdigest()
    rows = []
    for method, detector in detectors.items():
        positive = bool(alarms[method])
        complete = positive or not (failures[method] or stream_error or not frames)
        row = {key: str(episode[key]) for key in ("dataset", "scene_id", "episode_id")}
        row.update(task=episode["task"], method=method,
            detected=positive if complete else None, complete=complete,
            status="completed" if complete else "partial",
            first_alarm_step=alarms[method][0]["step"] if positive else None,
            alarms=alarms[method], stats=dict(getattr(detector, "stats", {})),
            sampled_frames=samples, frames_read=frames, short_window=samples < cfg["window"],
            config=cfg, fingerprint=fingerprint, source_sha256=episode.get("source_sha256"),
            source_sha256_status=episode.get("source_sha256_status", "manifest_expected_not_rehashed"))
        if not complete:
            row["error"] = stream_error or "incomplete_query_coverage" if frames else "empty_trajectory"
        rows.append(row)
    _json(output / "episodes" / dataset / (token + ".json"), rows)
    return rows


def _ensure_annotation_rows(path, episodes):
    """Expand a cohort without altering existing annotator-entered cell values."""
    if not path.exists():
        write_annotation_template(path, episodes)
        return
    existing_bytes = path.read_bytes()
    known = _references(load_annotations(path))
    missing = [episode for episode in episodes if tuple(str(episode[k]).strip()
        for k in ("dataset", "scene_id", "episode_id")) not in known]
    if not missing:
        return
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields, old_rows = list(reader.fieldnames), list(reader)
    temporary = path.with_name(path.name + "." + str(uuid4()) + ".tmp")
    try:
        write_annotation_template(temporary, missing)
        with temporary.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            new_fields, new_rows = list(reader.fieldnames), list(reader)
        fields.extend(field for field in new_fields if field not in fields)
        export_csv(temporary, old_rows + new_rows, fields)
        if path.read_bytes() != existing_bytes:
            raise RuntimeError("Annotation file changed while expanding cohort")
        os.replace(str(temporary), str(path))
    finally:
        if temporary.exists():
            temporary.unlink()


def run_evaluation(episodes, config, output_dir, methods, annotations=None,
                   llm_client=None, workers=1, progress=None):
    """Replay a selected cohort and write auditable predictions and score tables."""
    if workers < 1:
        raise ValueError("workers must be positive")
    episodes = list(episodes)
    if not episodes:
        raise ValueError("No episodes selected")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    labels_path = output / "manual_labels.csv"
    _ensure_annotation_rows(labels_path, episodes)
    labels = load_annotations(labels_path) if annotations is None else list(annotations)
    identity = lambda r: tuple(str(r[k]) for k in ("dataset", "scene_id", "episode_id"))
    label_map = _references(labels)
    references = []
    for episode in episodes:
        reference = {**episode, "runtime_system_error": None, "reviewer": ""}
        reference.update(label_map.get(identity(episode), {}))
        reference["native_success"] = episode["native_success"]
        references.append(reference)
    references = list(_references(references).values())
    selected = {identity(episode) for episode in episodes}
    unmatched = [dict(zip(("dataset", "scene_id", "episode_id"), key))
        for key in label_map if key not in selected]
    _json(output / "references_used.json", {
        "target": "runtime_system_error", "scope": "all_episodes",
        "annotation_source": str(labels_path) if annotations is None else "provided_annotations",
        "references": references, "unmatched_annotation_count": len(unmatched),
        "unmatched_annotation_identities": unmatched})
    def replay(episode):
        return run_episode(episode, config, output, methods, llm_client)
    if workers == 1:
        batches = []
        for done, episode in enumerate(episodes, 1):
            batches.append(replay(episode))
            if progress is not None:
                progress(done, len(episodes), episode)
    else:
        batches = [None] * len(episodes)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(replay, episode): index
                for index, episode in enumerate(episodes)}
            for done, future in enumerate(as_completed(futures), 1):
                index = futures[future]
                batches[index] = future.result()
                if progress is not None:
                    progress(done, len(episodes), episodes[index])
    predictions = [row for batch in batches for row in batch]
    metrics, main_rows, failed_rows = {}, [], []
    for dataset in dict.fromkeys(row["dataset"] for row in episodes):
        refs = [r for r in references if r["dataset"] == dataset]
        preds = [p for p in predictions if p["dataset"] == dataset]
        metrics[dataset] = score_episodes(refs, preds)
        failed_scores = score_episodes(refs, preds, scope="failed_episodes_only")
        metrics[dataset]["failed_episodes_only"] = failed_scores
        failed = sum(not r["native_success"] for r in refs)
        failed_labels = [r["runtime_system_error"] for r in refs if not r["native_success"]]
        runtime_share = sum(label in (1, "1") for label in failed_labels) / failed \
            if failed and all(label in (0, 1, "0", "1") for label in failed_labels) else None
        for method, scores in metrics[dataset]["methods"].items():
            main_rows.append({"dataset": dataset, "task": refs[0]["task"],
                "episodes": len(refs), "failures": failed, "failure_rate": failed / len(refs),
                "system_failure_share": None,
                "runtime_error_share_among_failures": runtime_share,
                "method": method,
                **{k: scores[k] for k in ("status", "tp", "fp", "fn", "tn", "precision", "recall")},
                **scores["coverage"]})
            secondary = failed_scores["methods"][method]
            failed_rows.append({"dataset": dataset, "task": refs[0]["task"],
                "scope": "failed_episodes_only", "method": method,
                **{k: secondary[k] for k in ("status", "tp", "fp", "fn", "tn", "precision", "recall")},
                **secondary["coverage"]})
    overall = score_episodes(references, predictions)
    _json(output / "predictions.json", predictions)
    export_csv(output / "predictions_excel.csv", [{k: v for k, v in row.items()
        if k not in ("alarms", "stats", "config")} for row in predictions])
    with (output / "alarms.jsonl").open("w", encoding="utf-8") as handle:
        for row in predictions:
            for alarm in row["alarms"]:
                handle.write(json.dumps({"dataset": row["dataset"], "scene_id": row["scene_id"],
                    "episode_id": row["episode_id"], **alarm}, ensure_ascii=False) + "\n")
    _json(output / "metrics.json", {"target": "runtime_system_error", "unit": "episode",
        "datasets": metrics, "overall": overall})
    _json(output / "run_config.json", {"config": {**DEFAULT_CONFIG, **config},
        "methods": list(methods), "episode_count": len(episodes), "workers": workers})
    export_csv(output / "main_table_excel.csv", main_rows)
    export_csv(output / "failed_episodes_only_table_excel.csv", failed_rows)
    (output / "README.md").write_text(
        "# 系统运行异常检测\n\n"
        "两类检测覆盖所选任务：team_progress为结合团队目标的启发式，llm为语言模型；"
        "repeat_action是仅数连续动作名的弱基线。\n\n"
        "先填写manual_labels.csv：runtime_system_error填0或1，reviewer填人工核查者，"
        "可补充onset_step、offset_step和中文原因。包括成功episode；旧失败归因不作真值。"
        "初始标签全部为空，标签或预测覆盖不足时正式Precision/Recall为null。"
        "重新运行时保留人工表，并复用LLM缓存；可用--labels导入另存的标注。\n\n"
        "metrics.json和main_table_excel.csv按episode统计TP/FP/FN/TN，"
        "Precision=TP/(TP+FP)，Recall=TP/(TP+FN)，数值采用0到1比例。"
        "LLM只返回system_error布尔值；false表示当前未检出系统异常，"
        "请求/解析失败和预算耗尽记为覆盖不完整。\n\n"
        "sample_every为仿真帧采样间隔，window/check_every为采样后的观测点数。"
        "默认50帧采样、10点窗口跨度450步（约500步），首查在第450步，之后每250步检查。"
        "短轨迹不在终局补查。到首次报警为止的原文请求和响应在llm_requests/，"
        "逐集结果在episodes/，配置在run_config.json。状态与当前原生谓词可能包含模拟器"
        "特权观测；不向检测器提供最终结果或人工标签。\n\n"
        "references_used.json保存本次评分实际采用的人工标签、核查者与未匹配条目数量。"
        "扩展同一输出目录的任务范围会追加空白标注行，保留已有人工单元格内容。"
        "source_sha256仅是原manifest的预期值，状态manifest_expected_not_rehashed表示"
        "本回放工具没有重新读取全部原文计算校验值。\n\n"
        "Failure rate单独按原生成败计算；系统失败归因占比与运行异常不是同一标签，"
        "因此本表system_failure_share留空。runtime_error_share_among_failures仅在全部"
        "失败episode的新人工运行异常标签已知时计算；它不表示因果归因。"
        "failed_episodes_only_table_excel.csv是仅失败集的独立范围，主表含全部episode。\n", encoding="utf-8")
    return {"metrics": metrics, "overall": overall, "episode_count": len(episodes),
        "predictions": predictions, "output_dir": str(output)}
