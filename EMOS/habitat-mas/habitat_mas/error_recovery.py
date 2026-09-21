"""Minimal, deterministic recovery for permanent agent dropout.

Task ownership is inferred from exact entity references in existing assignments.
Goal completion always comes from the simulator, never from LLM claims or waits.
"""

from __future__ import annotations

import json
import re
from pathlib import Path


class RecoveryError(ValueError):
    """The detected dropout cannot be handled by this recovery baseline."""


def snapshot_goals(problem):
    """Evaluate every atomic goal in an AND tree, without short circuiting."""
    goals = []

    def visit(expr):
        if hasattr(expr, "sub_exprs"):
            if expr.expr_type.value != "and" or expr.quantifier is not None:
                raise RecoveryError("Recovery supports only conjunctions of atomic goals")
            for child in expr.sub_exprs:
                visit(child)
        else:
            goals.append({"id": expr.compact_str,
                          "entities": [arg.name for arg in expr._arg_values],
                          "complete": bool(expr.is_true(problem.sim_info))})

    visit(problem.goal)
    return goals


def _mentions(text, entity):
    return re.search(r"(?<![\w|])" + re.escape(entity) + r"(?![\w|])", text) is not None


def build_recovery_plan(goals, previous, online, failed, state):
    """Keep live owners and balance orphaned goals by unfinished task count."""
    online = sorted(online)
    if not online:
        raise RecoveryError("No online agents remain")
    assigned = {name: [] for name in online}
    preserved = {name: [] for name in online}
    completed, unfinished, unassigned, transfers = [], [], [], []
    owners, pending = {}, []
    for goal in goals:
        task = goal["id"]
        entities = goal["entities"]
        required = [entity for entity in entities if re.fullmatch(r"agent_\d+", entity)]
        targets = [entity for entity in entities if entity not in required]
        # Recovery-generated assignments identify predicates exactly. Original
        # natural-language assignments can only be matched by entity references.
        candidates = [name for name, text in previous.items() if task in text.splitlines()]
        if not candidates:
            candidates = [name for name, text in previous.items()
                          if targets and all(_mentions(text, entity) for entity in targets)]
        if goal["complete"]:
            completed.append(task)
            maintainers = required or (candidates if len(candidates) == 1 else [])
            for name in maintainers:
                if name in preserved:
                    preserved[name].append(task)
            continue
        if any(name not in online for name in required):
            raise RecoveryError(f"Goal {task} requires unavailable agent {required}")
        if len(set(required)) > 1:
            raise RecoveryError(f"Goal {task} requires coordinated agent-specific execution")
        for name in failed:
            held = state.get("agents", {}).get(name, {}).get("held_object")
            if held and held.get("goal_name") in entities:
                raise RecoveryError(f"Goal {task} depends on an object held by {name}")
        owners[task] = candidates
        if any(name in failed for name in candidates):
            unfinished.append(task)
        if len(candidates) != 1:
            unassigned.append(task)
        live_owners = [name for name in candidates if name in online] if len(candidates) == 1 else []
        if required or live_owners:
            owner = (required or sorted(live_owners))[0]
            assigned[owner].append(task)
        else:
            pending.append(task)
    # Count retained work before distributing failed/unassigned work.
    for task in pending:
        owner = min(online, key=lambda name: (len(assigned[name]) + len(preserved[name]), name))
        assigned[owner].append(task)
        transfers.append({"task": task, "from": owners[task], "to": owner})
    return {"assignments": assigned, "previous_owners": owners,
            "preserved_tasks": preserved,
            "completed_tasks": completed, "unfinished_tasks": unfinished,
            "unassigned_tasks": unassigned, "transfers": transfers}


class DropoutRecovery:
    """Episode-scoped recovery and audit trail; policy/LLM objects are injected."""

    def __init__(self, event_path=None, record_event=None):
        self.event_path = None if event_path is None else Path(event_path)
        self.record_event = record_event
        if self.event_path is not None:
            self.event_path.parent.mkdir(parents=True, exist_ok=True)
            self.event_path.write_text("", encoding="utf-8")
        self.reset()

    def reset(self):
        self.attempts = 0
        self.recoveries = 0
        self.transferred_tasks = 0

    def _record(self, event):
        if self.event_path is not None:
            with self.event_path.open("a", encoding="utf-8") as output:
                output.write(json.dumps(event, ensure_ascii=False) + "\n")
        if self.record_event is not None:
            self.record_event(event)

    def metrics(self):
        return {"error_recovery.attempts": float(self.attempts),
                "error_recovery.recoveries": float(self.recoveries),
                "error_recovery.transferred_tasks": float(self.transferred_tasks)}

    def recover(self, error, detector, policies, text_context, text_goal, get_context):
        """Reassign and replan before acknowledging the latched dropout.

        RecoveryError keeps the existing crash path. Unexpected API/configuration
        errors keep their original exception and are not classified as dropout.
        """
        if error.event["reason"] != "agent_dropout":
            raise error
        self.attempts += 1
        base = {key: error.event[key] for key in ("scene_id", "episode_id", "simulator_step")}
        failed = sorted(name for name, health in detector.health.items() if not health["online"])
        online = sorted(name for name, health in detector.health.items() if health["online"])
        self._record({**base, "type": "failure_detected", "reason": "agent_dropout", "failed_agents": failed})
        try:
            context = get_context()
            if context.get("unsupported_reason"):
                raise RecoveryError(context["unsupported_reason"])
            previous = {f"agent_{i}": getattr(policy._high_level_policy.llm_agent,
                                             "subtask_description", "")
                        for i, policy in enumerate(policies)
                        if policy._high_level_policy.llm_agent.initialized}
            plan = build_recovery_plan(context["goals"], previous, online, failed,
                                       context["simulator_state"])
            self._record({**base, "type": "tasks_reassigned", "failed_agents": failed, **plan})
            resume = json.loads(text_context["robot_resume"])
            scene = context["scene_description"]
            recovery_context = (
                f"Agents {failed} are offline permanently; active agents: {online}. "
                "Do not send requests to offline agents. The previous action was interrupted, "
                "not necessarily completed. Continue from the current physical state. "
                "Execute your remaining assigned goal conditions. The completed_tasks list "
                "means conditions satisfied NOW, not permanently completed actions. Preserve "
                "these conditions, especially your preserved_tasks; re-establish them if "
                "movement invalidates them. All goal conditions must hold at episode end.\n"
                + json.dumps({"completed_tasks": plan["completed_tasks"],
                              "preserved_tasks": plan["preserved_tasks"],
                              "team_assignments": plan["assignments"],
                              "simulator_state": context["simulator_state"]}, ensure_ascii=False)
                + "\nCurrent scene:\n" + scene
            )
            for i, policy in enumerate(policies):
                name = f"agent_{i}"
                if name not in online:
                    continue
                llm = policy._high_level_policy.llm_agent
                tasks = plan["assignments"][name]
                subtask = "Complete these remaining goal conditions:\n" + "\n".join(tasks) if tasks else "Nothing to do"
                if llm.initialized:
                    llm.replan(subtask, recovery_context)
                else:
                    llm.init_agent(robot_type=resume[name]["robot_type"],
                                   task_description=text_goal,
                                   subtask_description=subtask + "\n" + recovery_context)
                    # Keep the task ledger separate from shared scene/context tokens.
                    llm.subtask_description = subtask
                policy._cur_call_high_level.fill_(True)
                self._record({**base, "type": "agent_replanned", "agent": name, "tasks": tasks})
            detector.acknowledge_dropout(failed)
            self.recoveries += 1
            self.transferred_tasks += len(plan["transfers"])
            self._record({**base, "type": "recovery_completed", "failed_agents": failed,
                          "active_agents": online, "status": "ready_to_resume"})
            text_context["scene_description"] = scene
            return plan
        except RecoveryError as failure:
            self._record({**base, "type": "recovery_failed", "reason": str(failure)})
            error.event["recovery_failure"] = str(failure)
            raise error from failure
