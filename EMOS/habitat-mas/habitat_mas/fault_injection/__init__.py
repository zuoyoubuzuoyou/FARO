"""Fault injection primitives for EMOS experiments."""

from habitat_mas.fault_injection.object_localization import (
    FaultInjectionError,
    FaultEventRecorder,
    ObjectLocalizationFaultSpec,
    inject_object_localization_error,
)
from habitat_mas.fault_injection.trajectory import TrajectoryRecorder

__all__ = [
    "FaultEventRecorder",
    "FaultInjectionError",
    "ObjectLocalizationFaultSpec",
    "TrajectoryRecorder",
    "inject_object_localization_error",
]
