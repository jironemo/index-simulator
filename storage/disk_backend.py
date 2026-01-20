from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, List, Optional, Sequence

from config import BPTREE_ORDER, DISK_BPTREE_SNAPSHOT_LIMIT, LSM_LEVELS, LSM_MEMTABLE_THRESHOLD
from interfaces.storage import OperationMetrics, OperationResult, SimulationBackend, Snapshot
from models import Student
from services.benchmark import BenchmarkRecorder
from storage.bptree_disk_store import DiskBPlusTreeStore
from storage.in_memory_backend import QueueEntry
from storage.lsm_disk_store import DiskLSMStore

OperationPayload = Student | int


@dataclass(slots=True)
class DiskBackendConfig:
    """Configuration for disk-backed simulations."""

    base_path: Path
    memtable_threshold: int = LSM_MEMTABLE_THRESHOLD
    num_levels: int = LSM_LEVELS
    bptree_order: int = BPTREE_ORDER
    bptree_snapshot_limit: int = DISK_BPTREE_SNAPSHOT_LIMIT


class DiskSimulationBackend(SimulationBackend):
    """Simulation backend operating directly on disk-backed data structures."""

    def __init__(self, config: DiskBackendConfig) -> None:
        self.config = config
        self.config.base_path.mkdir(parents=True, exist_ok=True)
        self._queue: Deque[QueueEntry] = deque()
        self.benchmark = BenchmarkRecorder()
        self.lsm_store = DiskLSMStore(
            self.config.base_path / "lsm",
            memtable_threshold=self.config.memtable_threshold,
            num_levels=self.config.num_levels,
        )
        self.bptree_store = DiskBPlusTreeStore(
            self.config.base_path / "bptree",
            self.config.bptree_order,
            snapshot_limit=self.config.bptree_snapshot_limit,
            flush_threshold=self.config.memtable_threshold,
        )

    # ------------------------------------------------------------------
    # Queue management
    # ------------------------------------------------------------------
    def enqueue(self, operation: str, payload: OperationPayload) -> None:
        self._queue.append((operation, payload))

    def extend_queue(self, operations: Sequence[tuple[str, OperationPayload]]) -> None:
        self._queue.extend(operations)

    def step(self) -> Optional[OperationResult]:
        if not self._queue:
            return None
        operation, payload = self._queue.popleft()
        return self.execute(operation, payload)

    def drain(self) -> Sequence[OperationResult]:
        results: List[OperationResult] = []
        while self._queue:
            result = self.step()
            if result:
                results.append(result)
        return results

    def queue_depth(self) -> int:
        return len(self._queue)

    def clear_queue(self) -> None:
        self._queue.clear()

    # ------------------------------------------------------------------
    # Immediate operations
    # ------------------------------------------------------------------
    def execute(self, operation: str, payload: OperationPayload) -> OperationResult:
        normalized = operation.lower()
        if normalized not in {"insert", "update", "delete", "search"}:
            raise ValueError(f"Unsupported operation: {operation}")
        if isinstance(payload, Student):
            student_payload = payload
            student_id = payload.id
        else:
            student_payload = None
            student_id = int(payload)

        if normalized in {"insert", "update"} and student_payload is None:
            raise ValueError(f"{operation.title()} operation requires a Student payload.")

        lsm_metrics: OperationMetrics
        bpt_metrics: OperationMetrics
        lsm_record: Optional[Student] = None
        bpt_record: Optional[Student] = None

        if normalized == "insert":
            assert student_payload is not None
            lsm_metrics = self.lsm_store.insert(student_payload)
            bpt_metrics = self.bptree_store.insert(student_payload)
        elif normalized == "update":
            assert student_payload is not None
            lsm_metrics = self.lsm_store.update(student_payload)
            bpt_metrics = self.bptree_store.update(student_payload)
        elif normalized == "delete":
            lsm_metrics = self.lsm_store.delete(student_id)
            bpt_metrics = self.bptree_store.delete(student_id)
        else:  # search
            lsm_record, lsm_metrics = self.lsm_store.search(student_id)
            bpt_record, bpt_metrics = self.bptree_store.search(student_id)

        self.benchmark.record("lsm", normalized, self._metrics_to_dict(lsm_metrics))
        self.benchmark.record("bptree", normalized, self._metrics_to_dict(bpt_metrics))

        return OperationResult(
            operation=normalized,
            student_id=student_id,
            lsm_metrics=lsm_metrics,
            bptree_metrics=bpt_metrics,
            lsm_result=lsm_record.to_summary() if lsm_record else None,
            bptree_result=bpt_record.to_summary() if bpt_record else None,
        )

    # ------------------------------------------------------------------
    # Maintenance operations
    # ------------------------------------------------------------------
    def flush(self) -> OperationMetrics:
        metrics = self.lsm_store.flush_memtable()
        # Flushing does not impact B+ tree state, so we keep its metrics unchanged.
        metrics.memory_bytes += self.bptree_store.memory_bytes()
        metrics.disk_bytes += self.bptree_store.disk_bytes()
        return metrics

    def compact(self) -> OperationMetrics:
        metrics = self.lsm_store.compact_levels()
        metrics.memory_bytes += self.bptree_store.memory_bytes()
        metrics.disk_bytes += self.bptree_store.disk_bytes()
        return metrics

    # ------------------------------------------------------------------
    # State helpers
    # ------------------------------------------------------------------
    def snapshot(self) -> Snapshot:
        lsm_snapshot = self.lsm_store.snapshot()
        bptree_snapshot = self.bptree_store.snapshot()
        summary = self.benchmark.summary()
        memory_usage = self.lsm_store.memory_bytes() + self.bptree_store.memory_bytes()
        disk_usage = self.lsm_store.disk_bytes() + self.bptree_store.disk_bytes()
        return Snapshot(
            lsm=lsm_snapshot,
            bptree=bptree_snapshot,
            benchmarks=summary,
            queue_depth=len(self._queue),
            memory_bytes=memory_usage,
            disk_bytes=disk_usage,
        )

    def export_state(self) -> Dict[str, object]:
        return {
            "lsm": self.lsm_store.export_memtable(),
            "bptree": self.bptree_store.export_records(),
        }

    def reset(self) -> None:
        self.lsm_store.reset()
        self.bptree_store.reset()
        self.benchmark.reset()
        self._queue.clear()

    def _metrics_to_dict(self, metrics: OperationMetrics) -> Dict[str, int]:
        return {
            "in_memory_ns": metrics.in_memory_ns,
            "file_io_ns": metrics.file_io_ns,
            "memory_bytes": metrics.memory_bytes,
            "disk_bytes": metrics.disk_bytes,
        }
