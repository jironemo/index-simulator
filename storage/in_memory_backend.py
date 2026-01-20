from __future__ import annotations

import sys
import time
from collections import deque
from typing import Deque, Dict, List, Optional, Sequence, Tuple

from config import BPTREE_ORDER, LSM_LEVELS, LSM_MEMTABLE_THRESHOLD
from interfaces.storage import OperationMetrics, OperationResult, SimulationBackend, Snapshot
from models import BPlusTreeModel, LSMTreeModel, Student
from services.benchmark import BenchmarkRecorder

OperationPayload = Student | int
QueueEntry = Tuple[str, OperationPayload]


class InMemorySimulationBackend(SimulationBackend):
    """Simulation backend that reuses in-memory LSM/B+ tree implementations."""

    def __init__(
        self,
        *,
        memtable_threshold: int = LSM_MEMTABLE_THRESHOLD,
        num_levels: int = LSM_LEVELS,
        bptree_order: int = BPTREE_ORDER,
    ) -> None:
        self.lsm_model = LSMTreeModel(memtable_threshold, num_levels)
        self.bptree_model = BPlusTreeModel(bptree_order)
        self.benchmark = BenchmarkRecorder()
        self._queue: Deque[QueueEntry] = deque()

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
        result = self.execute(operation, payload)
        return result

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

        lsm_metrics: Dict[str, int]
        bpt_metrics: Dict[str, int]
        lsm_record: Optional[Student] = None
        bpt_record: Optional[Student] = None

        if normalized == "insert":
            assert student_payload is not None  # for type-checkers
            lsm_metrics = self.lsm_model.insert(student_payload)
            bpt_metrics = self.bptree_model.insert(student_payload)
        elif normalized == "update":
            assert student_payload is not None
            lsm_metrics = self.lsm_model.update(student_payload)
            bpt_metrics = self.bptree_model.update(student_payload)
        elif normalized == "delete":
            lsm_metrics = self.lsm_model.delete(student_id)
            bpt_metrics = self.bptree_model.delete(student_id)
        else:  # search
            t0 = time.perf_counter_ns()
            lsm_record = self.lsm_model.search(student_id)
            t1 = time.perf_counter_ns()
            bpt_record = self.bptree_model.search(student_id)
            t2 = time.perf_counter_ns()
            lsm_metrics = {"in_memory_ns": t1 - t0}
            bpt_metrics = {"in_memory_ns": t2 - t1}

        lsm_op_metrics = self._to_operation_metrics(lsm_metrics, structure="lsm")
        bpt_op_metrics = self._to_operation_metrics(bpt_metrics, structure="bptree")

        self.benchmark.record("lsm", normalized, self._metrics_to_dict(lsm_op_metrics))
        self.benchmark.record("bptree", normalized, self._metrics_to_dict(bpt_op_metrics))

        return OperationResult(
            operation=normalized,
            student_id=student_id,
            lsm_metrics=lsm_op_metrics,
            bptree_metrics=bpt_op_metrics,
            lsm_result=lsm_record.to_summary() if lsm_record else None,
            bptree_result=bpt_record.to_summary() if bpt_record else None,
        )

    # ------------------------------------------------------------------
    # Maintenance operations
    # ------------------------------------------------------------------
    def flush(self) -> OperationMetrics:
        duration = self.lsm_model.flush_memtable()
        metrics = OperationMetrics(in_memory_ns=0, file_io_ns=duration)
        metrics.memory_bytes = self._estimate_memory_usage()
        metrics.disk_bytes = 0
        return metrics

    def compact(self) -> OperationMetrics:
        duration = self.lsm_model.compact_levels()
        metrics = OperationMetrics(in_memory_ns=0, file_io_ns=duration)
        metrics.memory_bytes = self._estimate_memory_usage()
        metrics.disk_bytes = 0
        return metrics

    # ------------------------------------------------------------------
    # State helpers
    # ------------------------------------------------------------------
    def snapshot(self) -> Snapshot:
        lsm_snapshot = self.lsm_model.snapshot()
        bptree_snapshot = self.bptree_model.snapshot()
        summary = self.benchmark.summary()
        memory_usage = self._estimate_memory_usage()
        return Snapshot(
            lsm=lsm_snapshot,
            bptree=bptree_snapshot,
            benchmarks=summary,
            queue_depth=len(self._queue),
            memory_bytes=memory_usage,
            disk_bytes=0,
        )

    def export_state(self) -> Dict[str, object]:
        return {
            "lsm": self.lsm_model.export_memtable(),
            "bptree": self.bptree_model.export_records(),
        }

    def reset(self) -> None:
        self.lsm_model.reset()
        self.bptree_model.reset()
        self.benchmark.reset()
        self._queue.clear()

    def _metrics_to_dict(self, metrics: OperationMetrics) -> Dict[str, int]:
        return {
            "in_memory_ns": metrics.in_memory_ns,
            "file_io_ns": metrics.file_io_ns,
            "memory_bytes": metrics.memory_bytes,
            "disk_bytes": metrics.disk_bytes,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _to_operation_metrics(self, metrics: Dict[str, int], *, structure: str) -> OperationMetrics:
        op_metrics = OperationMetrics(
            in_memory_ns=int(metrics.get("in_memory_ns", 0)),
            file_io_ns=int(metrics.get("file_io_ns", 0)),
        )
        memory_usage = self._estimate_memory_usage()
        op_metrics.memory_bytes = memory_usage
        op_metrics.disk_bytes = 0
        return op_metrics

    def _estimate_memory_usage(self) -> int:
        return self._estimate_lsm_memory() + self._estimate_bptree_memory()

    def _estimate_lsm_memory(self) -> int:
        size = sys.getsizeof(self.lsm_model.memtable)
        for entry in self.lsm_model.memtable.values():
            size += sys.getsizeof(entry)
            if entry.student is not None:
                size += sys.getsizeof(entry.student)
        size += sys.getsizeof(self.lsm_model.levels)
        for level in self.lsm_model.levels:
            size += sys.getsizeof(level)
            for table in level:
                size += sys.getsizeof(table)
                for entry in table:
                    size += sys.getsizeof(entry)
                    if entry.student is not None:
                        size += sys.getsizeof(entry.student)
        return size

    def _estimate_bptree_memory(self) -> int:
        return self._estimate_bptree_node(self.bptree_model.root)

    def _estimate_bptree_node(self, node) -> int:
        size = sys.getsizeof(node)
        size += sys.getsizeof(node.keys)
        for key in node.keys:
            size += sys.getsizeof(key)
        if node.is_leaf:
            size += sys.getsizeof(node.values)
            for value in node.values:
                size += sys.getsizeof(value)
        else:
            size += sys.getsizeof(node.children)
            for child in node.children:
                size += self._estimate_bptree_node(child)
        return size
