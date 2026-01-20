from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Optional, Protocol, Sequence

from models import Student


@dataclass(slots=True)
class OperationMetrics:
    """Timing and resource metrics captured for a single structure/operation."""

    in_memory_ns: int = 0
    file_io_ns: int = 0
    memory_bytes: int = 0
    disk_bytes: int = 0


@dataclass(slots=True)
class OperationResult:
    """Result payload produced by a backend operation for both structures."""

    operation: str
    student_id: int
    lsm_metrics: OperationMetrics
    bptree_metrics: OperationMetrics
    lsm_result: Optional[str] = None
    bptree_result: Optional[str] = None


@dataclass(slots=True)
class Snapshot:
    """Unified snapshot returned by storage backends."""

    lsm: Mapping[str, object]
    bptree: Mapping[str, object]
    benchmarks: Mapping[str, Mapping[str, Mapping[str, float]]]
    queue_depth: int
    memory_bytes: int = 0
    disk_bytes: int = 0


class SimulationBackend(Protocol):
    """Backend capable of executing LSM/B+ tree workloads."""

    def execute(self, operation: str, payload: Student | int) -> OperationResult:
        ...

    def enqueue(self, operation: str, payload: Student | int) -> None:
        ...

    def extend_queue(self, operations: Sequence[tuple[str, Student | int]]) -> None:
        ...

    def step(self) -> Optional[OperationResult]:
        ...

    def drain(self) -> Sequence[OperationResult]:
        ...

    def queue_depth(self) -> int:
        ...

    def clear_queue(self) -> None:
        ...

    def flush(self) -> OperationMetrics:
        ...

    def compact(self) -> OperationMetrics:
        ...

    def snapshot(self) -> Snapshot:
        ...

    def export_state(self) -> Mapping[str, object]:
        ...

    def reset(self) -> None:
        ...


__all__ = [
    "OperationMetrics",
    "OperationResult",
    "Snapshot",
    "SimulationBackend",
]
