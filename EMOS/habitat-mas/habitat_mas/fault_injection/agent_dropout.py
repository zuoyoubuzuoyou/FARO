"""Agent-dropout structure fault for EMOS multi-agent evaluation."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from habitat_mas.fault_injection.naming import output_run_stem


class FaultInjectionError(ValueError):
    """Raised when an enabled structure fault is invalid or never fires."""


def _config_value(config: Any, name: str, default: Any) -> Any:
    if config is None:
        return default
    if isinstance(config, Mapping):
        return config.get(name, default)
    return getattr(config, name, default)


@dataclass(frozen=True)
class AgentDropoutFaultSpec:
    """Validated runtime view of an agent-dropout Hydra configuration."""

    enabled: bool = False
    fault_id: str = "SF_agent_dropout_001"
    episode_id: str = ""
    agent: str = "agent_0"
    trigger_domain: str = "simulator_step"
    simulator_step: int = 0
    simulator_step_schedule: str = ""
    decision_step: int = 0

    @classmethod
    def from_config(cls, config: Any) -> "AgentDropoutFaultSpec":
        return cls(
            enabled=bool(_config_value(config, "enabled", False)),
            fault_id=str(
                _config_value(config, "fault_id", "SF_agent_dropout_001")
            ),
            episode_id=str(_config_value(config, "episode_id", "")),
            agent=str(_config_value(config, "agent", "agent_0")),
            trigger_domain=str(
                _config_value(config, "trigger_domain", "simulator_step")
            ),
            simulator_step=int(
                _config_value(config, "simulator_step", 0)
            ),
            simulator_step_schedule=str(
                _config_value(config, "simulator_step_schedule", "")
            ),
            decision_step=int(_config_value(config, "decision_step", 0)),
        )

    def simulator_step_schedule_map(self) -> dict[str, int]:
        """Parse ``episode@step;...`` into an episode-specific schedule."""

        schedule: dict[str, int] = {}
        raw_schedule = self.simulator_step_schedule.strip()
        if not raw_schedule:
            return schedule
        for raw_entry in raw_schedule.split(";"):
            entry = raw_entry.strip()
            parts = entry.split("@")
            if len(parts) != 2 or not parts[0].strip():
                raise FaultInjectionError(
                    "agent_dropout.simulator_step_schedule must use "
                    "episode_id@step entries separated by semicolons"
                )
            episode_id, raw_step = (part.strip() for part in parts)
            if episode_id in schedule:
                raise FaultInjectionError(
                    "agent_dropout.simulator_step_schedule contains duplicate "
                    f"episode_id {episode_id!r}"
                )
            try:
                step = int(raw_step)
            except ValueError as error:
                raise FaultInjectionError(
                    "agent_dropout.simulator_step_schedule step must be an "
                    f"integer, got {raw_step!r}"
                ) from error
            if step < 0:
                raise FaultInjectionError(
                    "agent_dropout.simulator_step_schedule steps must be >= 0"
                )
            schedule[episode_id] = step
        return schedule

    def validate(self) -> None:
        if not self.enabled:
            return
        if re.fullmatch(r"agent_\d+", self.agent) is None:
            raise FaultInjectionError(
                "agent_dropout.agent must use the form agent_N"
            )
        if self.trigger_domain not in {
            "simulator_step",
            "high_level_decision",
        }:
            raise FaultInjectionError(
                "agent_dropout.trigger_domain must be simulator_step or "
                "high_level_decision"
            )
        if self.simulator_step < 0:
            raise FaultInjectionError(
                "agent_dropout.simulator_step must be >= 0"
            )
        if self.decision_step < 0:
            raise FaultInjectionError(
                "agent_dropout.decision_step must be >= 0"
            )
        schedule = self.simulator_step_schedule_map()
        if schedule and self.trigger_domain != "simulator_step":
            raise FaultInjectionError(
                "agent_dropout.simulator_step_schedule requires "
                "trigger_domain=simulator_step"
            )
        if schedule and self.episode_id:
            raise FaultInjectionError(
                "agent_dropout.episode_id and simulator_step_schedule are "
                "mutually exclusive"
            )

    def episode_matches(self, episode_id: Any) -> bool:
        schedule = self.simulator_step_schedule_map()
        if schedule:
            return str(episode_id) in schedule
        return not self.episode_id or self.episode_id == str(episode_id)

    def target_simulator_step(self, episode_id: Any) -> int:
        schedule = self.simulator_step_schedule_map()
        if schedule:
            try:
                return schedule[str(episode_id)]
            except KeyError as error:
                raise FaultInjectionError(
                    f"episode {episode_id!r} is not in simulator_step_schedule"
                ) from error
        return self.simulator_step

    def is_active(
        self,
        *,
        episode_id: Any,
        agent: str,
        simulator_step: int,
        decision_step: int | None = None,
        is_high_level_decision: bool = False,
    ) -> bool:
        if not self.enabled:
            return False
        self.validate()
        if not (
            self.episode_matches(episode_id) and self.agent == str(agent)
        ):
            return False
        if self.trigger_domain == "simulator_step":
            return int(simulator_step) >= self.target_simulator_step(
                episode_id
            )
        return (
            is_high_level_decision
            and decision_step is not None
            and int(decision_step) == self.decision_step
        )

    def is_trigger(
        self,
        *,
        episode_id: Any,
        agent: str,
        simulator_step: int,
        decision_step: int | None = None,
        is_high_level_decision: bool = False,
    ) -> bool:
        if not self.is_active(
            episode_id=episode_id,
            agent=agent,
            simulator_step=simulator_step,
            decision_step=decision_step,
            is_high_level_decision=is_high_level_decision,
        ):
            return False
        if self.trigger_domain == "simulator_step":
            return int(simulator_step) == self.target_simulator_step(
                episode_id
            )
        return True


def build_agent_dropout_event(
    spec: AgentDropoutFaultSpec,
    *,
    episode_id: Any,
    simulator_step: int | None = None,
    decision_step: int | None = None,
) -> dict[str, Any]:
    """Build the one-shot audit event emitted when dropout becomes active."""

    spec.validate()
    configured_simulator_step = spec.target_simulator_step(episode_id)
    actual_simulator_step = (
        configured_simulator_step
        if simulator_step is None
        else int(simulator_step)
    )
    actual_decision_step = (
        spec.decision_step
        if spec.trigger_domain == "high_level_decision"
        and decision_step is None
        else decision_step
    )
    return {
        "schema_version": 1,
        "type": "fault_injected",
        "fault_id": spec.fault_id,
        "fault_type": "StructureFault",
        "fault_subtype": "AgentDropout",
        "trigger_domain": spec.trigger_domain,
        "phase": (
            "PRE_HIGH_LEVEL_DECISION"
            if spec.trigger_domain == "high_level_decision"
            else "PRE_ACTION"
        ),
        "episode_id": str(episode_id),
        "agent": spec.agent,
        "simulator_step": actual_simulator_step,
        "decision_step": actual_decision_step,
        "high_level_decisions_completed": actual_decision_step,
        "configured_simulator_step": configured_simulator_step,
        "simulator_step_schedule": spec.simulator_step_schedule,
        "configured_decision_step": spec.decision_step,
        "duration": "until_episode_end",
        "silent": False,
        "effects": {
            "execution_policy": "not_called",
            "incoming_messages": "rejected_agent_unavailable",
            "queued_messages": "discarded",
            "team_membership": "removed",
            "peer_observation": "membership_update",
            "policy_state": "frozen",
            "environment_action": "zero_noop",
        },
        "simulator_agent_removed": False,
        "logical_agent_active": False,
    }


class AgentDropoutEventRecorder:
    """Write agent-dropout audit events as JSON Lines."""

    def __init__(
        self,
        output_dir: str,
        run_label: str,
        *,
        fault_name: str = "agent_dropout",
    ) -> None:
        run_stem = output_run_stem(run_label, fault_name)
        self.path = (
            Path(output_dir)
            / f"{run_stem}_events.jsonl"
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("", encoding="utf-8")
        self.count = 0
        self.episode_ids: set[str] = set()

    def record(self, event: Mapping[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as output:
            output.write(json.dumps(dict(event), ensure_ascii=False))
            output.write("\n")
            output.flush()
        self.count += 1
        if "episode_id" in event:
            self.episode_ids.add(str(event["episode_id"]))
