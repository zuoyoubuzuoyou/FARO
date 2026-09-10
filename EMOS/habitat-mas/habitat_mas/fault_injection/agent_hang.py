"""Logical agent suspension until episode end."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from habitat_mas.fault_injection.agent_dropout import (
    AgentDropoutEventRecorder,
    AgentDropoutFaultSpec,
    FaultInjectionError,
    _config_value,
    build_agent_dropout_event,
)


@dataclass(frozen=True)
class AgentHangFaultSpec:
    """Hang triggered only by a zero-based simulator step."""

    enabled: bool = False
    fault_id: str = "SF_agent_hang_001"
    episode_id: str = ""
    agent: str = "agent_0"
    simulator_step: int = 0
    simulator_step_schedule: str = ""

    @classmethod
    def from_config(cls, config: Any) -> "AgentHangFaultSpec":
        if (
            _config_value(config, "trigger_domain", "simulator_step")
            != "simulator_step"
        ):
            raise FaultInjectionError(
                "agent_hang only supports simulator_step triggers"
            )
        if _config_value(config, "decision_step", None) is not None:
            raise FaultInjectionError(
                "agent_hang.decision_step is unsupported; use simulator_step"
            )
        return cls(
            enabled=bool(_config_value(config, "enabled", False)),
            fault_id=str(
                _config_value(config, "fault_id", "SF_agent_hang_001")
            ),
            episode_id=str(_config_value(config, "episode_id", "")),
            agent=str(_config_value(config, "agent", "agent_0")),
            simulator_step=int(_config_value(config, "simulator_step", 0)),
            simulator_step_schedule=str(
                _config_value(config, "simulator_step_schedule", "")
            ),
        )

    def _trigger_spec(self) -> AgentDropoutFaultSpec:
        # Share episode/schedule validation without exposing decision triggers.
        return AgentDropoutFaultSpec(**asdict(self))

    def validate(self) -> None:
        try:
            self._trigger_spec().validate()
        except FaultInjectionError as error:
            raise FaultInjectionError(
                str(error).replace("agent_dropout", "agent_hang")
            ) from error

    def simulator_step_schedule_map(self) -> dict[str, int]:
        return self._trigger_spec().simulator_step_schedule_map()

    def is_trigger(
        self, *, episode_id: Any, agent: str, simulator_step: int
    ) -> bool:
        self.validate()
        return self._trigger_spec().is_trigger(
            episode_id=episode_id,
            agent=agent,
            simulator_step=simulator_step,
        )


def build_agent_hang_event(
    spec: AgentHangFaultSpec,
    *,
    episode_id: Any,
    simulator_step: int,
    decision_step: int,
) -> dict[str, Any]:
    spec.validate()
    event = build_agent_dropout_event(
        spec._trigger_spec(),
        episode_id=episode_id,
        simulator_step=simulator_step,
        decision_step=decision_step,
    )
    event.pop("configured_decision_step")
    event.update(
        fault_subtype="AgentHang",
        logical_agent_active=True,
        logical_agent_responsive=False,
    )
    return event


@dataclass
class AgentHangState:
    """Latch one silent hang until the policy resets the episode state."""

    active_event: dict[str, Any] | None = None

    def advance(
        self,
        spec: AgentHangFaultSpec,
        *,
        episode_id: Any,
        agent: str,
        simulator_step: int,
        decision_step: int,
    ) -> dict[str, Any] | None:
        if self.active_event is None and spec.is_trigger(
            episode_id=episode_id,
            agent=agent,
            simulator_step=simulator_step,
        ):
            self.active_event = build_agent_hang_event(
                spec,
                episode_id=episode_id,
                simulator_step=simulator_step,
                decision_step=decision_step,
            )
            return self.active_event
        return None


class AgentHangEventRecorder(AgentDropoutEventRecorder):
    def __init__(self, output_dir: str, run_label: str) -> None:
        super().__init__(output_dir, run_label, fault_name="agent_hang")
