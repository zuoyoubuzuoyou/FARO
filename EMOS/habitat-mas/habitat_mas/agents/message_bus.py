"""Episode-scoped communication for the single-environment agent team."""

from __future__ import annotations


class AgentMessageBus:
    messages: dict[str, list[str]] = {}
    members: set[str] = set()
    dropped_agents: set[str] = set()
    sent_messages: list[dict[str, str]] = []
    pending_messages: list[dict] = []
    delivery_events: list[dict] = []
    simulator_step: int = 0
    delay: dict | None = None
    _next_message_id: int = 0

    @classmethod
    def reset(cls) -> None:
        cls.messages.clear()
        cls.members.clear()
        cls.dropped_agents.clear()
        cls.sent_messages.clear()
        cls.pending_messages.clear()
        cls.delivery_events.clear()
        cls.simulator_step = 0
        cls.delay = None
        cls._next_message_id = 0

    @classmethod
    def begin_step(cls, step: int, delay: dict | None = None) -> None:
        """Deliver before either agent runs; disabling delay keeps queued due times."""
        cls.simulator_step, cls.delay = int(step), delay
        waiting = []
        for message in cls.pending_messages:
            if message["delivery_step"] > step:
                waiting.append(message)
                continue
            if message["target"] in cls.dropped_agents:
                cls.delivery_events.append({**message, "type": "message_dropped",
                                            "simulator_step": step, "reason": "agent_unavailable"})
            else:
                cls._deliver(message["source"], message["target"], message["request"])
                cls.delivery_events.append({**message, "type": "message_delivered",
                                            "simulator_step": step, "actual_delivery_step": step})
        cls.pending_messages = waiting

    @classmethod
    def drain_delivery_events(cls) -> list[dict]:
        events, cls.delivery_events = cls.delivery_events, []
        return events

    @classmethod
    def finish_episode(cls, step: int) -> None:
        for message in cls.pending_messages:
            cls.delivery_events.append({**message, "type": "message_dropped",
                                        "simulator_step": step, "reason": "episode_ended_before_delivery"})
        cls.pending_messages.clear()
        cls.delay = None

    @classmethod
    def _deliver(cls, source, target, request) -> None:
        cls.messages.setdefault(target, []).append(
            f'"{source}" agent sent you requests: "{request}".'
        )

    @classmethod
    def active_members(cls) -> list[str]:
        return sorted(cls.members - cls.dropped_agents)

    @classmethod
    def drop_agent(cls, name: str) -> None:
        """Remove a logical member and discard its undeliverable inbox."""
        cls.dropped_agents.add(name)
        cls.messages.pop(name, None)
        remaining = []
        for message in cls.pending_messages:
            if message["target"] == name:
                cls.delivery_events.append({**message, "type": "message_dropped",
                                            "simulator_step": cls.simulator_step, "reason": "agent_unavailable"})
            else:
                remaining.append(message)
        cls.pending_messages = remaining

    @classmethod
    def send_request(cls, source: str, target: str, request: str) -> str | None:
        """Queue a request, or return an immediate transport failure.

        A hung agent retains membership and its inbox. Queueing a message does
        not mean that the recipient has processed it or acknowledged it.
        """
        if source in cls.dropped_agents or target in cls.dropped_agents:
            return (
                f"send_request failed: agent_unavailable. {target!r} cannot "
                "receive this request because the sender or recipient has "
                "left the team. The request was not queued."
            )
        if cls.delay is not None and cls.delay["agent"] == target:
            message = {
                "message_id": cls._next_message_id, "source": source, "target": target,
                "request": request, "send_step": cls.simulator_step,
                "delivery_step": cls.simulator_step + cls.delay["delay_steps"],
                "fault_id": cls.delay["fault_id"],
            }
            cls._next_message_id += 1
            cls.pending_messages.append(message)
            cls.delivery_events.append({**message, "type": "message_queued",
                                        "simulator_step": cls.simulator_step})
        else:
            cls._deliver(source, target, request)
        cls.sent_messages.append({"source": source, "target": target, "request": request})
        return None

    @classmethod
    def membership_observation(cls) -> str:
        if not cls.dropped_agents:
            return ""
        return (
            f"Team membership update: {', '.join(sorted(cls.dropped_agents))} "
            "left the team and cannot receive requests or execute tasks. "
            f"Current team members: {', '.join(cls.active_members()) or 'none'}."
        )
