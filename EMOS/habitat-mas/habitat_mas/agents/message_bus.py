"""Episode-scoped communication for the single-environment agent team."""

from __future__ import annotations


class AgentMessageBus:
    messages: dict[str, list[str]] = {}
    members: set[str] = set()
    dropped_agents: set[str] = set()
    sent_messages: list[dict[str, str]] = []

    @classmethod
    def reset(cls) -> None:
        cls.messages.clear()
        cls.members.clear()
        cls.dropped_agents.clear()
        cls.sent_messages.clear()

    @classmethod
    def active_members(cls) -> list[str]:
        return sorted(cls.members - cls.dropped_agents)

    @classmethod
    def drop_agent(cls, name: str) -> None:
        """Remove a logical member and discard its undeliverable inbox."""
        cls.dropped_agents.add(name)
        cls.messages.pop(name, None)

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
        cls.messages.setdefault(target, []).append(
            f'"{source}" agent sent you requests: "{request}".'
        )
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
