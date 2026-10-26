"""Capture evaluation provenance without resolving config or reading secrets.

This module only observes declared configuration and local source bytes. Model
defaults are deliberately left unknown until the model wrapper reports the
actual request at runtime.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields, is_dataclass
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
from typing import Any


_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
_SOURCE_FILES = (
    "habitat-mas/habitat_mas/utils/run_config.py",
    "habitat-mas/habitat_mas/utils/trajectory.py",
    "habitat-mas/habitat_mas/utils/trajectory_storage.py",
    "habitat-mas/habitat_mas/utils/models.py",
    "habitat-mas/habitat_mas/agents/crab_agent.py",
    "habitat-baselines/habitat_baselines/run.py",
    "habitat-baselines/habitat_baselines/rl/multi_agent/habitat_mas_evaluator.py",
    "habitat-baselines/habitat_baselines/rl/multi_agent/multi_llm_policy.py",
    "habitat-baselines/habitat_baselines/rl/hrl/hl/llm_policy.py",
    "habitat-baselines/habitat_baselines/rl/hrl/hierarchical_policy.py",
    "habitat-baselines/habitat_baselines/rl/hrl/skills/skill.py",
    "habitat-lab/habitat/core/environments.py",
)
_PROMPT_FILES = (
    "habitat-baselines/habitat_baselines/rl/multi_agent/multi_llm_policy.py",
    "habitat-mas/habitat_mas/agents/crab_agent.py",
    "habitat-baselines/habitat_baselines/rl/hrl/hl/llm_policy.py",
)
_CONFIG_FILES = (
    "habitat-baselines/habitat_baselines/config/default_structured_configs.py",
    "habitat-lab/habitat/config/default_structured_configs.py",
)
_SECRET_KEY = re.compile(
    r"(?:api[_-]?key|secret|password|passwd|authorization|credential|"
    r"access[_-]?token|refresh[_-]?token|bearer[_-]?token|private[_-]?key|^token$)",
    re.IGNORECASE,
)
_URL = re.compile(r"[a-z][a-z0-9+.-]*://[^\s\"'<>]+", re.IGNORECASE)
_SECRET_VALUE = re.compile(r"\b(?:sk-[a-z0-9_-]{8,}|Bearer\s+[^\s\"'<>]+)", re.IGNORECASE)
_DECODE_KEYS = (
    "temperature", "top_p", "max_tokens", "max_completion_tokens", "seed",
    "frequency_penalty", "presence_penalty", "stop", "n", "reasoning_effort",
)


def _plain(value: Any) -> Any:
    """Copy config without invoking interpolations, properties, or repr()."""
    try:
        from omegaconf import OmegaConf
    except ImportError:
        OmegaConf = None
    if OmegaConf is not None and OmegaConf.is_config(value):
        return OmegaConf.to_container(value, resolve=False)
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: _plain(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    attributes = getattr(value, "__dict__", None)
    if isinstance(attributes, dict):
        return {key: _plain(item) for key, item in attributes.items()
                if not key.startswith("_")}
    return None


def _safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: "[REDACTED]" if _SECRET_KEY.search(key) else _safe(item)
                for key, item in value.items()}
    if isinstance(value, list):
        return [_safe(item) for item in value]
    if isinstance(value, str):
        # URLs are unnecessary for reproducing policy settings and may contain
        # credentials in userinfo, path segments, or query parameters.
        return _SECRET_VALUE.sub("[REDACTED]", _URL.sub("[REDACTED_URL]", value))
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _get(config: Any, path: str, default=None):
    current = config
    for key in path.split("."):
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    # Keep unresolved expressions in the safe snapshot, but don't pretend they
    # are observed settings in the summary fields.
    if isinstance(current, str) and (current == "???" or "${" in current):
        return default
    return current


def _hash_files(root: Path, paths) -> dict:
    result = {}
    for entry in sorted(set(str(path) for path in paths)):
        path = Path(entry)
        path = path if path.is_absolute() else root / path
        if path.is_file():
            try:
                label = str(path.relative_to(root))
            except ValueError:
                label = str(path)
            result[label] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def _hydra_context(missing: dict) -> tuple:
    try:
        from hydra.core.hydra_config import HydraConfig
        if not HydraConfig.initialized():
            missing["config_path"] = "HydraConfig is not initialized; source config path is unavailable."
            return None, {}, []
        hydra = _plain(HydraConfig.get())
        choices = _get(hydra, "runtime.choices", {}) or {}
        name = _get(hydra, "job.config_name")
        roots = [source["path"] for source in
                 (_get(hydra, "runtime.config_sources", []) or [])
                 if source.get("schema") == "file" and source.get("path")]
        if name is None:
            missing["config_path"] = "Initialized Hydra config does not report job.config_name."
        return name, choices, roots
    except Exception as error:
        missing["config_path"] = f"Hydra source config path unavailable ({type(error).__name__})."
        return None, {}, []


def _version(root: Path, config: dict, config_name, choices, hydra_roots, missing):
    version = {"git_head": None, "git_dirty": None}
    for field, args in (
        ("git_head", ["rev-parse", "HEAD"]),
        ("git_dirty", ["status", "--porcelain", "--untracked-files=normal"]),
    ):
        try:
            output = subprocess.run(
                ["git", "-C", str(root), *args], check=True,
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
                timeout=5,
            ).stdout.strip()
            version[field] = bool(output) if field == "git_dirty" else output
        except (OSError, subprocess.SubprocessError) as error:
            missing[f"version.{field}"] = f"Git metadata unavailable ({type(error).__name__})."
    source_paths = list(_SOURCE_FILES)
    # Action schemas are part of the effective prompt sent to execution agents.
    action_dir = root / "habitat-mas/habitat_mas/agents/actions"
    action_sources = list(action_dir.glob("*.py")) if action_dir.is_dir() else []
    source_paths.extend(action_sources)
    prompt_paths = [*_PROMPT_FILES, *action_sources]
    config_paths = list(_CONFIG_FILES)
    search_roots = [Path(path) for path in hydra_roots]
    search_roots.extend((root / "habitat-baselines/habitat_baselines/config",
                         root / "habitat-lab/habitat/config"))
    selections = ([str(config_name)] if config_name else [])
    selections.extend(f"{key.split('@', 1)[0].lstrip('/')}/{value}"
                      for key, value in choices.items()
                      if value is not None and not key.startswith("hydra/"))
    for selection in selections:
        for config_root in search_roots:
            candidate = config_root / selection
            if candidate.suffix not in (".yaml", ".yml", ".json"):
                candidate = candidate.with_suffix(".yaml")
            if candidate.is_file():
                config_paths.append(candidate)
    for field in ("habitat.dataset.robot_config", "habitat.simulator.json_path"):
        path = _get(config, field)
        if isinstance(path, str) and path:
            config_paths.append(path)
    task_spec = _get(config, "habitat.task.task_spec")
    task_base = _get(config, "habitat.task.task_spec_base_path")
    if task_spec and task_base:
        task_file = str(Path(task_base) / task_spec)
        for config_root in search_roots:
            config_paths.append(config_root / f"{task_file}.yaml")
    version["source_files"] = _hash_files(root, source_paths)
    version["config_files"] = _hash_files(root, config_paths)
    version["prompt_sources"] = _hash_files(root, prompt_paths)
    version["config_snapshot_sha256"] = hashlib.sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":"),
                   ensure_ascii=False, allow_nan=False).encode("utf-8")
    ).hexdigest()
    if config_name is None:
        missing["version.composed_config_files"] = (
            "Hydra composition provenance unavailable; hashes include known "
            "structured configs and declared local robot/task files."
        )
    return version


def _models(config: dict, missing: dict) -> dict:
    models = {}
    for module in ("group_discussion", "agent_reflection", "execution"):
        # No model settings are currently forwarded through the stock policy
        # config. Record declared settings only, never constructor defaults.
        declared = _get(config, f"habitat_baselines.eval.models.{module}")
        if not isinstance(declared, dict):
            declared = {}
        model = declared.get("model", declared.get("model_id"))
        decoding = declared.get("decoding")
        if decoding is None:
            decoding = {key: declared[key] for key in _DECODE_KEYS if key in declared} or None
        models[module] = {"model": model, "decoding": decoding,
                          "observation_status": "declared_config_only",
                          "config_source": f"habitat_baselines.eval.models.{module}" if declared else None}
        if model is None:
            missing[f"models.{module}.model"] = (
                "No explicit module model in evaluation config; model wrapper "
                "defaults are reported from actual runtime requests."
            )
        if decoding is None:
            missing[f"models.{module}.decoding"] = (
                "No explicit decoding settings in evaluation config; omitted "
                "provider defaults are unknown."
            )
        if not declared:
            missing[f"models.{module}.config_source"] = "No explicit module model configuration."
    return models


def build_run_metadata(config, checkpoint_index=None, step_id=None, *, repository_root=None):
    """Return a JSON-safe manifest of the observed run configuration."""
    missing = {}
    try:
        snapshot = _safe(_plain(config))
        if not isinstance(snapshot, dict):
            raise TypeError("Expected a configuration mapping")
    except Exception as error:
        snapshot = None
        missing["config_snapshot"] = f"Safe unresolved config snapshot unavailable ({type(error).__name__})."
    observed = snapshot or {}
    config_name, choices, hydra_roots = _hydra_context(missing)
    config_path = None
    if config_name is not None:
        for directory in hydra_roots:
            candidate = Path(directory) / str(config_name)
            if candidate.suffix not in (".yaml", ".yml"):
                candidate = candidate.with_suffix(".yaml")
            if candidate.is_file():
                config_path = str(candidate)
                break
        if config_path is None:
            missing["config_path"] = "Hydra reports config_name but no accessible local source file."
    benchmark = _get(observed, "benchmark", _get(observed, "habitat.benchmark"))
    if benchmark is None:
        benchmark = next((value for key, value in choices.items()
                          if key.split("@", 1)[0].lstrip("/").startswith("benchmark/")), None)
    category = _get(observed, "task_category", _get(observed, "habitat.dataset.mode"))
    dataset = _get(observed, "habitat.dataset")
    agent_configs = _get(observed, "habitat.simulator.agents", {}) or {}
    order = _get(observed, "habitat.simulator.agents_order")
    if order is None:
        missing["agents_order"] = "Simulator agents_order is not explicitly available."
    identities = []
    if isinstance(agent_configs, dict):
        ids = order if isinstance(order, list) else list(agent_configs)
        for agent_id in ids:
            agent = agent_configs.get(agent_id, {}) or {}
            robot_type = agent.get("articulated_agent_type", agent.get("robot_type"))
            identities.append({"agent_id": agent_id, "robot_type": robot_type})
            if robot_type is None:
                missing[f"agents.{agent_id}.robot_type"] = "Agent config does not declare a robot type."
    elif isinstance(agent_configs, list):
        identities = [{"agent_id": None, "robot_type": entry.get("articulated_agent_type")}
                      for entry in agent_configs]
        missing["agents.identities"] = "Simulator agent configs have no declared mapping identities."
    success_measure = _get(observed, "habitat.task.success_measure")
    success_config = _get(observed, f"habitat.task.measurements.{success_measure}") if isinstance(success_measure, str) else None
    termination_paths = {
        "max_episode_steps": "habitat.environment.max_episode_steps",
        "max_episode_seconds": "habitat.environment.max_episode_seconds",
        "end_on_success": "habitat.task.end_on_success",
        "should_terminate_on_wait": "habitat.dataset.should_terminate_on_wait",
    }
    termination = {key: _get(observed, path) for key, path in termination_paths.items()}
    termination["task_conditions"] = {
        key: value for key, value in (_get(observed, "habitat.task", {}) or {}).items()
        if "terminate" in key or "ends_episode" in key or key.startswith("end_on_")
    }
    termination["measurement_configs"] = _get(observed, "habitat.task.measurements")
    metadata = {
        "checkpoint_index": checkpoint_index, "checkpoint_step": step_id,
        "benchmark": benchmark, "dataset": dataset, "task_category": category,
        "task_category_source": ("task_category" if _get(observed, "task_category") is not None else "habitat.dataset.mode") if category is not None else None,
        "task_spec": _get(observed, "habitat.task.task_spec"),
        "config_path": config_path, "config_name": config_name,
        "config_snapshot": snapshot, "hydra_choices": choices,
        "seed": _get(observed, "habitat.seed"),
        "simulator_seed": _get(observed, "habitat.simulator.seed"),
        "agents_order": order, "agents": identities,
        "agent_count": len(agent_configs),
        "models": _models(observed, missing), "termination": termination,
        "model_provenance": "models contains declared config only; config.json models_used contains actual runtime requests. Stock policies do not forward eval.models settings.",
        "success_measure": success_measure,
        "success_definition": {"measure_name": success_measure, "measure_config": success_config},
    }
    for key in ("benchmark", "dataset", "task_category", "task_category_source",
                "task_spec", "seed", "simulator_seed", "success_measure"):
        if metadata[key] is None:
            missing[key] = "Not declared in the observed evaluation configuration."
    if success_config is None:
        missing["success_definition.measure_config"] = "Configured original success measurement specification is unavailable."
    if success_measure is None:
        missing["success_definition.measure_name"] = "No original success measure configured."
    if config_name is None:
        missing["config_name"] = "Hydra job.config_name is unavailable."
    if checkpoint_index is None:
        missing["checkpoint_index"] = "Evaluator did not supply a checkpoint index."
    if step_id is None:
        missing["checkpoint_step"] = "Evaluator did not supply a checkpoint step."
    for key, value in termination.items():
        if value is None:
            missing[f"termination.{key}"] = "Not declared in the observed evaluation configuration."
    root = Path(repository_root) if repository_root is not None else _REPOSITORY_ROOT
    metadata["version"] = _version(root, observed, config_name, choices, hydra_roots, missing)
    metadata["missing_fields"] = missing
    return _safe(metadata)
