"""Stage-aware fault injection for the EMOS MVP.

The simulator and scene graph remain ground truth. Faults are applied only to
the agent-visible text, task assignments, executed action, or verification
result, and every successful injection is written as an auditable JSON record.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple


FAULT_DEFINITIONS = {
    "WrongObjectLocation": {
        "fault_type": "PerceptionFault",
        "phase": "observation",
        "required": ("faulty_agent", "affected_object", "observed_location"),
        "prefix": "F1_wrong_location",
        "recovery": "refresh_observation_or_ask_teammate",
    },
    "ObjectRecognitionError": {
        "fault_type": "PerceptionFault",
        "phase": "observation",
        "required": ("faulty_agent", "affected_object", "observed_object"),
        "prefix": "F2_wrong_recognition",
        "recovery": "reobserve_object_or_ask_teammate",
    },
    "WrongSubgoal": {
        "fault_type": "PlanningFault",
        "phase": "assignment",
        "required": ("faulty_agent", "wrong_subgoal"),
        "prefix": "F3_wrong_subgoal",
        "recovery": "replan_or_reject_subgoal",
    },
    "MissingSubgoal": {
        "fault_type": "PlanningFault",
        "phase": "assignment",
        "required": ("faulty_agent",),
        "prefix": "F4_missing_subgoal",
        "recovery": "detect_unsatisfied_precondition_and_replan",
    },
    "MessageDelayOrStale": {
        "fault_type": "CommunicationFault",
        "phase": "observation",
        "required": ("faulty_agent", "stale_message"),
        "prefix": "F5_stale_message",
        "recovery": "request_fresh_message_or_cross_check",
    },
    "DuplicateAssignment": {
        "fault_type": "CoordinationFault",
        "phase": "assignment",
        "required": ("faulty_agent", "source_agent"),
        "prefix": "F6_duplicate_assignment",
        "recovery": "deduplicate_assignments_and_reallocate",
    },
    "ActionNoOpOrFalseSuccess": {
        "fault_type": "ActionFault",
        "phase": "control",
        "required": ("faulty_agent",),
        "prefix": "F7_action_noop",
        "recovery": "verify_world_state_and_retry_action",
    },
    "MissingOrFalseVerification": {
        "fault_type": "VerificationFault",
        "phase": "control",
        "required": ("faulty_agent",),
        "prefix": "F8_false_verification",
        "recovery": "cross_agent_verification_or_reobserve",
    },
}

OBSERVATION_SUBTYPES = {
    subtype
    for subtype, definition in FAULT_DEFINITIONS.items()
    if definition["phase"] == "observation"
}
ASSIGNMENT_SUBTYPES = {
    subtype
    for subtype, definition in FAULT_DEFINITIONS.items()
    if definition["phase"] == "assignment"
}
CONTROL_SUBTYPES = {
    subtype
    for subtype, definition in FAULT_DEFINITIONS.items()
    if definition["phase"] == "control"
}


@dataclass(frozen=True)
class FaultInjectionResult:
    """Result of applying matching faults to one text observation."""

    scene_description: str
    records: Tuple[Dict[str, Any], ...] = ()

    @property
    def changed(self) -> bool:
        return any(record.get("status") == "injected" for record in self.records)


@dataclass(frozen=True)
class AssignmentFaultInjectionResult:
    """Result of applying planning/coordination faults to assignments."""

    assignments: Dict[str, str]
    records: Tuple[Dict[str, Any], ...] = ()

    @property
    def changed(self) -> bool:
        return any(record.get("status") == "injected" for record in self.records)


@dataclass(frozen=True)
class ControlFaultInjectionResult:
    """Control overrides requested by F7/F8."""

    force_noop: bool = False
    false_success: bool = False
    verification_result: Optional[bool] = None
    records: Tuple[Dict[str, Any], ...] = ()

    @property
    def changed(self) -> bool:
        return any(record.get("status") == "injected" for record in self.records)


def _config_value(config: Any, key: str, default: Any = None) -> Any:
    if isinstance(config, Mapping):
        return config.get(key, default)
    return getattr(config, key, default)


def _safe_path_component(value: Any) -> str:
    component = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._")
    return component or "unknown"


class TextObservationFaultInjector:
    """Inject the eight first-stage EMOS MVP faults.

    The historical class name is retained for compatibility. In addition to
    :meth:`inject`, the class now exposes assignment and control phase APIs.
    """

    def __init__(
        self,
        faults: Iterable[Mapping[str, Any]],
        log_dir: str,
        *,
        enabled: bool = True,
        strict: bool = True,
    ) -> None:
        self.enabled = enabled
        self.log_dir = Path(log_dir)
        self.strict = strict
        self.faults = [
            self._normalize_fault(fault, index)
            for index, fault in enumerate(faults)
        ]
        self._observation_cache: Dict[
            Tuple[str, int, str, str], FaultInjectionResult
        ] = {}
        self._assignment_cache: Dict[
            Tuple[str, int, str], AssignmentFaultInjectionResult
        ] = {}
        self._control_cache: Dict[
            Tuple[str, int, str, str], ControlFaultInjectionResult
        ] = {}

    @classmethod
    def from_config(cls, config: Any) -> Optional["TextObservationFaultInjector"]:
        """Build an injector from ``habitat.dataset.fault_injection``."""

        if config is None or not bool(_config_value(config, "enabled", False)):
            return None
        schedule_path = str(_config_value(config, "schedule_path", "")).strip()
        if not schedule_path:
            raise ValueError(
                "Fault injection is enabled but fault_injection.schedule_path is empty."
            )
        with open(schedule_path, "r", encoding="utf-8") as schedule_file:
            schedule = json.load(schedule_file)
        if isinstance(schedule, list):
            faults = schedule
        elif isinstance(schedule, dict) and "faults" in schedule:
            faults = schedule["faults"]
        elif isinstance(schedule, dict):
            faults = [schedule]
        else:
            raise ValueError("Fault schedule must be a JSON object or list.")
        return cls(
            faults=faults,
            log_dir=str(_config_value(config, "log_dir", "fault_injection_output")),
            enabled=True,
            strict=bool(_config_value(config, "strict", True)),
        )

    @staticmethod
    def _normalize_fault(fault: Mapping[str, Any], index: int) -> Dict[str, Any]:
        normalized = dict(fault)
        subtype = normalized.get("fault_subtype", "WrongObjectLocation")
        if subtype not in FAULT_DEFINITIONS:
            raise ValueError(
                f"Unsupported fault_subtype {subtype!r}. Supported values: "
                f"{sorted(FAULT_DEFINITIONS)}"
            )
        definition = FAULT_DEFINITIONS[subtype]
        normalized.setdefault("fault_subtype", subtype)
        normalized.setdefault("fault_type", definition["fault_type"])
        normalized.setdefault(
            "fault_id", f"{definition['prefix']}_{index + 1:03d}"
        )
        normalized.setdefault("injected_at_step", normalized.pop("inject_at_step", 0))
        normalized.setdefault("severity", "medium")
        normalized.setdefault("expected_recovery", definition["recovery"])
        normalized.setdefault("enabled", True)

        if normalized["fault_type"] != definition["fault_type"]:
            raise ValueError(
                f"Fault {normalized['fault_id']!r} has type "
                f"{normalized['fault_type']!r}; subtype {subtype!r} requires "
                f"{definition['fault_type']!r}."
            )
        for required in definition["required"]:
            if not str(normalized.get(required, "")).strip():
                raise ValueError(
                    f"Fault {normalized['fault_id']!r} is missing {required!r}."
                )
        normalized["injected_at_step"] = int(normalized["injected_at_step"])
        if definition["phase"] in ("observation", "assignment") and normalized[
            "injected_at_step"
        ] != 0:
            raise ValueError(
                f"{definition['phase'].capitalize()} fault "
                f"{normalized['fault_id']!r} must use injected_at_step=0 "
                "because EMOS creates scene context and assignments at episode "
                "initialization."
            )
        if subtype == "MissingOrFalseVerification":
            mode = normalized.setdefault("verification_mode", "false_positive")
            if mode not in ("false_positive", "false_negative", "missing"):
                raise ValueError(
                    "verification_mode must be false_positive, false_negative, "
                    "or missing."
                )
        return normalized

    def inject(
        self,
        scene_description: str,
        *,
        episode_id: Any,
        step: int,
        observer: str,
    ) -> FaultInjectionResult:
        """Inject F1, F2, and F5 into one agent-visible observation."""

        if not self.enabled:
            return FaultInjectionResult(scene_description)
        cache_key = (str(episode_id), int(step), observer, scene_description)
        if cache_key in self._observation_cache:
            return self._observation_cache[cache_key]

        faulty_description = scene_description
        records: List[Dict[str, Any]] = []
        for fault in self.faults:
            if fault["fault_subtype"] not in OBSERVATION_SUBTYPES or not self._matches(
                fault, episode_id, step, observer
            ):
                continue
            subtype = fault["fault_subtype"]
            if subtype == "WrongObjectLocation":
                replacement = self._replace_object_location(
                    faulty_description,
                    affected_object=fault["affected_object"],
                    observed_location=fault["observed_location"],
                )
            elif subtype == "ObjectRecognitionError":
                replacement = self._replace_object_identity(
                    faulty_description,
                    affected_object=fault["affected_object"],
                    observed_object=fault["observed_object"],
                )
            else:
                stale_message = str(fault["stale_message"])
                observed = f'[Teammate message to {observer}]: "{stale_message}"'
                replacement = (
                    faulty_description.rstrip() + "\n" + observed + "\n",
                    "no delayed message in current context",
                    observed,
                )

            record = self._record_replacement(
                fault, episode_id=episode_id, replacement=replacement
            )
            if replacement is not None:
                faulty_description = replacement[0]
            records.append(record)

        result = FaultInjectionResult(faulty_description, tuple(records))
        self._observation_cache[cache_key] = result
        return result

    def inject_assignments(
        self,
        assignments: Mapping[str, str],
        *,
        episode_id: Any,
        step: int,
    ) -> AssignmentFaultInjectionResult:
        """Inject F3, F4, and F6 after leader planning."""

        original_key = json.dumps(dict(assignments), sort_keys=True)
        cache_key = (str(episode_id), int(step), original_key)
        if cache_key in self._assignment_cache:
            return self._assignment_cache[cache_key]
        faulty_assignments = dict(assignments)
        records: List[Dict[str, Any]] = []
        for fault in self.faults:
            if fault["fault_subtype"] not in ASSIGNMENT_SUBTYPES or not self._matches(
                fault, episode_id, step, fault["faulty_agent"]
            ):
                continue
            subtype = fault["fault_subtype"]
            target_agent = fault["faulty_agent"]
            ground_truth = faulty_assignments.get(target_agent)
            observed: Optional[str] = None
            if ground_truth is None:
                replacement = None
            elif subtype == "WrongSubgoal":
                observed = str(fault["wrong_subgoal"])
                faulty_assignments[target_agent] = observed
                replacement = ("", ground_truth, observed)
            elif subtype == "MissingSubgoal":
                missing = str(
                    fault.get("missing_subgoal")
                    or fault.get("affected_subtask")
                    or ""
                )
                if missing and missing in ground_truth:
                    observed = ground_truth.replace(missing, "").strip(" ,;.")
                    if not observed:
                        observed = "Nothing to do"
                else:
                    observed = "Nothing to do"
                faulty_assignments[target_agent] = observed
                replacement = ("", ground_truth, observed)
            else:
                source_agent = fault["source_agent"]
                source_task = faulty_assignments.get(source_agent)
                if source_task is None:
                    replacement = None
                else:
                    observed = source_task
                    faulty_assignments[target_agent] = observed
                    replacement = ("", ground_truth, observed)

            record = self._record_replacement(
                fault, episode_id=episode_id, replacement=replacement
            )
            records.append(record)

        result = AssignmentFaultInjectionResult(faulty_assignments, tuple(records))
        self._assignment_cache[cache_key] = result
        return result

    def inject_control(
        self,
        *,
        episode_id: Any,
        step: int,
        observer: str,
        requested_action: str,
    ) -> ControlFaultInjectionResult:
        """Return F7/F8 action and verification overrides."""

        cache_key = (str(episode_id), int(step), observer, requested_action)
        if cache_key in self._control_cache:
            return self._control_cache[cache_key]
        force_noop = False
        false_success = False
        verification_result: Optional[bool] = None
        records: List[Dict[str, Any]] = []
        for fault in self.faults:
            if fault["fault_subtype"] not in CONTROL_SUBTYPES or not self._matches(
                fault, episode_id, step, observer
            ):
                continue
            if fault["fault_subtype"] == "ActionNoOpOrFalseSuccess":
                force_noop = True
                false_success = bool(fault.get("false_success", True))
                ground_truth = (
                    f"requested_action={requested_action}; "
                    "executed_action=no_op; action_success=false"
                )
                observed = "executed_action=no_op"
                if false_success:
                    observed += "; reported_success=true"
            else:
                mode = fault["verification_mode"]
                if mode == "false_positive":
                    verification_result = True
                    ground_truth = "verification_result=false"
                    observed = "verification_result=true; mode=false_positive"
                elif mode == "false_negative":
                    verification_result = False
                    ground_truth = "verification_result=true"
                    observed = "verification_result=false; mode=false_negative"
                else:
                    verification_result = False
                    ground_truth = "verification_required=true"
                    observed = "verification=missing; verification_result=false"
            record = self._make_record(
                fault,
                episode_id=episode_id,
                ground_truth_state=ground_truth,
                agent_observed_state=observed,
                status="injected",
            )
            self._write_record(record)
            records.append(record)
        result = ControlFaultInjectionResult(
            force_noop=force_noop,
            false_success=false_success,
            verification_result=verification_result,
            records=tuple(records),
        )
        self._control_cache[cache_key] = result
        return result

    def _record_replacement(
        self,
        fault: Mapping[str, Any],
        *,
        episode_id: Any,
        replacement: Optional[Tuple[str, str, str]],
    ) -> Dict[str, Any]:
        if replacement is None:
            message = (
                f"Fault target was not found for {fault['fault_id']!r} "
                f"in episode {episode_id!r}."
            )
            if self.strict:
                raise ValueError(message)
            record = self._make_record(
                fault,
                episode_id=episode_id,
                ground_truth_state=None,
                agent_observed_state=None,
                status="not_injected_target_not_found",
            )
        else:
            _, ground_truth, observed = replacement
            record = self._make_record(
                fault,
                episode_id=episode_id,
                ground_truth_state=ground_truth,
                agent_observed_state=observed,
                status="injected",
            )
        self._write_record(record)
        return record

    @staticmethod
    def _matches(
        fault: Mapping[str, Any], episode_id: Any, step: int, observer: str
    ) -> bool:
        if not fault.get("enabled", True):
            return False
        if int(fault["injected_at_step"]) != int(step):
            return False
        if fault["faulty_agent"] not in ("*", observer):
            return False
        configured_episode = fault.get("episode_id")
        configured_episodes = fault.get("episode_ids")
        if configured_episode is not None and str(configured_episode) != str(episode_id):
            return False
        if configured_episodes is not None and str(episode_id) not in {
            str(item) for item in configured_episodes
        }:
            return False
        return True

    @staticmethod
    def _replace_object_location(
        scene_description: str,
        *,
        affected_object: str,
        observed_location: str,
    ) -> Optional[Tuple[str, str, str]]:
        escaped_object = re.escape(affected_object)
        object_token = f'"{affected_object}"'
        lines = scene_description.splitlines(keepends=True)
        explicit_location = re.compile(
            rf'The object "{escaped_object}" is located .*?\.(?=\s+The height|\s*$)'
        )
        for index, line in enumerate(lines):
            if object_token not in line:
                continue
            match = explicit_location.search(line)
            if match is None:
                continue
            ground_truth = match.group(0)
            observed = (
                f'The object "{affected_object}" is located at '
                f'"{observed_location}".'
            )
            lines[index] = line[: match.start()] + observed + line[match.end() :]
            return "".join(lines), ground_truth, observed
        for index, line in enumerate(lines):
            if object_token not in line or not (
                "The height of" in line or "horizontal distance" in line
            ):
                continue
            ground_truth = line.rstrip("\r\n")
            observed = (
                f'The object "{affected_object}" is located at '
                f'"{observed_location}".'
            )
            newline = line[len(line.rstrip("\r\n")) :]
            lines[index] = f"{observed} {ground_truth}{newline}"
            return "".join(lines), ground_truth, observed
        return None

    @staticmethod
    def _replace_object_identity(
        scene_description: str,
        *,
        affected_object: str,
        observed_object: str,
    ) -> Optional[Tuple[str, str, str]]:
        ground_truth_token = f'"{affected_object}"'
        if ground_truth_token not in scene_description:
            return None
        observed_token = f'"{observed_object}"'
        return (
            scene_description.replace(ground_truth_token, observed_token),
            f"object_identity={affected_object}",
            f"object_identity={observed_object}",
        )

    @staticmethod
    def _make_record(
        fault: Mapping[str, Any],
        *,
        episode_id: Any,
        ground_truth_state: Optional[str],
        agent_observed_state: Optional[str],
        status: str,
    ) -> Dict[str, Any]:
        return {
            "fault_id": fault["fault_id"],
            "fault_type": fault["fault_type"],
            "fault_subtype": fault["fault_subtype"],
            "episode_id": str(episode_id),
            "injected_at_step": fault["injected_at_step"],
            "faulty_agent": fault["faulty_agent"],
            "affected_object": fault.get("affected_object"),
            "affected_subtask": fault.get("affected_subtask"),
            "severity": fault["severity"],
            "ground_truth_state": ground_truth_state,
            "agent_observed_state": agent_observed_state,
            "expected_recovery": fault["expected_recovery"],
            "status": status,
        }

    def _write_record(self, record: Mapping[str, Any]) -> None:
        episode_dir = self.log_dir / _safe_path_component(record["episode_id"])
        episode_dir.mkdir(parents=True, exist_ok=True)
        filename = _safe_path_component(record["fault_id"]) + ".json"
        output_path = episode_dir / filename
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=episode_dir,
            prefix=f".{filename}.",
            suffix=".tmp",
            delete=False,
        ) as temp_file:
            json.dump(record, temp_file, ensure_ascii=False, indent=2)
            temp_file.write("\n")
            temp_path = temp_file.name
        os.replace(temp_path, output_path)
