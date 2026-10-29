"""Configuration shared by Hydra evaluation and standalone recovery clients."""
from dataclasses import dataclass, fields
import math


@dataclass
class RecoveryConfig:
    enabled: bool = False
    mode: str = "source"
    persist_source_changes: bool = False
    max_policy_retries: int = 3
    task: str = "auto"
    detector: str = "llm"
    model: str = "gpt-6-sol"
    sample_every: int = 50
    window: int = 10
    check_every: int = 5
    min_stall_steps: int = 100
    reset_budget: int = 100
    history_turns: int = 10
    max_iterations: int = 1
    cooldown_steps: int = 250
    verify_after_steps: int = 250
    model_timeout: float = 60.0
    execution_timeout: float = 1.0
    modification_instruction: str = (
        "先根据任务、当前观测和原始执行代码判断是否存在系统级运行异常；仅在存在异常时，"
        "在允许范围内提出必要的最小源码修复。不改变任务目标、评分、能力或真实观测，"
        "保留已完成工作、物理状态、模型与对话，在同一运行实例中Resume；观察恢复后的实际反馈。"
    )

    @classmethod
    def from_value(cls, value):
        if isinstance(value, cls):
            return value.validate()
        defaults = cls()
        return cls(**{field.name: (value.get(field.name, getattr(defaults, field.name))
                                  if hasattr(value, "get") else getattr(value, field.name, getattr(defaults, field.name)))
                      for field in fields(cls)}).validate()

    def validate(self):
        if self.mode not in {"source", "extension"}:
            raise ValueError("Recovery mode must be source or extension")
        if not isinstance(self.persist_source_changes, bool):
            raise ValueError("persist_source_changes must be a boolean")
        if self.detector not in {"llm", "team_progress"}:
            raise ValueError("Recovery detector must be llm or team_progress")
        for name in ("sample_every", "window", "check_every", "min_stall_steps", "reset_budget",
                     "history_turns", "max_iterations", "max_policy_retries", "cooldown_steps", "verify_after_steps"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(name + " must be a positive integer")
        for name in ("model_timeout", "execution_timeout"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value <= 0:
                raise ValueError(name + " must be finite and positive")
        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError("Recovery model is required")
        if not isinstance(self.modification_instruction, str) or not self.modification_instruction.strip():
            raise ValueError("Recovery modification_instruction is required")
        return self
