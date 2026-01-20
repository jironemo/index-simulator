from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Union

from config import BPTREE_ORDER, LSM_LEVELS, LSM_MEMTABLE_THRESHOLD
from interfaces.storage import OperationMetrics, OperationResult, SimulationBackend, Snapshot
from models import Student
from storage import DiskBackendConfig, DiskSimulationBackend, InMemorySimulationBackend

OperationPayload = Union[Student, int]


class SimulationController:
    """Coordinates simulation backends and surfaces results to the UI layer."""

    def __init__(
        self,
        *,
        memtable_threshold: int = LSM_MEMTABLE_THRESHOLD,
        num_levels: int = LSM_LEVELS,
        bptree_order: int = BPTREE_ORDER,
        backend: Optional[SimulationBackend] = None,
        backend_type: str = "memory",
        disk_path: Optional[Union[str, Path]] = None,
    ) -> None:
        if backend is not None:
            self.backend = backend
        else:
            if backend_type.lower() == "disk":
                base_path = Path(disk_path or "./storage_data").resolve()
                config = DiskBackendConfig(
                    base_path=base_path,
                    memtable_threshold=memtable_threshold,
                    num_levels=num_levels,
                    bptree_order=bptree_order,
                )
                self.backend = DiskSimulationBackend(config)
            else:
                self.backend = InMemorySimulationBackend(
                    memtable_threshold=memtable_threshold,
                    num_levels=num_levels,
                    bptree_order=bptree_order,
                )

    # ------------------------------------------------------------------
    # Queue management
    # ------------------------------------------------------------------
    def enqueue(self, operation: str, payload: OperationPayload) -> None:
        self.backend.enqueue(operation, payload)

    def extend_queue(self, operations: Iterable[tuple[str, OperationPayload]]) -> None:
        seq = list(operations)
        self.backend.extend_queue(seq)

    def queue_depth(self) -> int:
        return self.backend.queue_depth()

    def clear_queue(self) -> None:
        self.backend.clear_queue()

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------
    def step(self) -> Optional[Dict[str, object]]:
        result = self.backend.step()
        if result is None:
            return None
        payload = self._operation_result_to_dict(result)
        payload["queued_operations"] = self.backend.queue_depth()
        return payload

    def drain_all(self) -> List[Dict[str, object]]:
        drained = self.backend.drain()
        results: List[Dict[str, object]] = []
        for result in drained:
            payload = self._operation_result_to_dict(result)
            payload["queued_operations"] = self.backend.queue_depth()
            results.append(payload)
        return results

    def execute_now(self, operation: str, payload: OperationPayload) -> Dict[str, object]:
        result = self.backend.execute(operation, payload)
        payload_dict = self._operation_result_to_dict(result)
        payload_dict["queued_operations"] = self.backend.queue_depth()
        return payload_dict

    def force_flush(self) -> Dict[str, object]:
        metrics = self.backend.flush()
        snapshot = self.snapshot()
        return {
            "flush_metrics": self._metrics_to_dict(metrics),
            "lsm": snapshot["lsm"],
        }

    def force_compact(self) -> Dict[str, object]:
        metrics = self.backend.compact()
        snapshot = self.snapshot()
        return {
            "compaction_metrics": self._metrics_to_dict(metrics),
            "lsm": snapshot["lsm"],
        }

    # ------------------------------------------------------------------
    # State inspection helpers
    # ------------------------------------------------------------------
    def snapshot(self) -> Dict[str, object]:
        snap = self.backend.snapshot()
        return {
            "lsm": snap.lsm,
            "bptree": snap.bptree,
            "benchmarks": snap.benchmarks,
            "queue_depth": snap.queue_depth,
            "memory_bytes": snap.memory_bytes,
            "disk_bytes": snap.disk_bytes,
        }

    def export_state(self) -> Dict[str, object]:
        return dict(self.backend.export_state())

    def reset(self) -> None:
        self.backend.reset()

    # ------------------------------------------------------------------
    # Compatibility accessors
    # ------------------------------------------------------------------
    @property
    def lsm_model(self) -> object:
        backend_model = getattr(self.backend, "lsm_model", None)
        if backend_model is None:
            backend_model = getattr(self.backend, "lsm_store", None)
        if backend_model is None:
            raise AttributeError("Active backend does not expose an LSM model")
        return backend_model

    @property
    def bptree_model(self) -> object:
        backend_model = getattr(self.backend, "bptree_model", None)
        if backend_model is None:
            backend_model = getattr(self.backend, "bptree_store", None)
        if backend_model is None:
            raise AttributeError("Active backend does not expose a B+ tree model")
        return backend_model

    @property
    def backend_mode(self) -> str:
        if isinstance(self.backend, DiskSimulationBackend):
            return "disk"
        return "memory"

    @property
    def backend_base_path(self) -> Optional[Path]:
        if isinstance(self.backend, DiskSimulationBackend):
            return self.backend.config.base_path
        return None

    # ------------------------------------------------------------------
    # Conversion helpers
    # ------------------------------------------------------------------
    def _operation_result_to_dict(self, result: OperationResult) -> Dict[str, object]:
        return {
            "operation": result.operation,
            "student_id": result.student_id,
            "lsm_metrics": self._metrics_to_dict(result.lsm_metrics),
            "bptree_metrics": self._metrics_to_dict(result.bptree_metrics),
            "lsm_result": result.lsm_result,
            "bptree_result": result.bptree_result,
        }

    def _metrics_to_dict(self, metrics: OperationMetrics) -> Dict[str, int]:
        return {
            "in_memory_ns": metrics.in_memory_ns,
            "file_io_ns": metrics.file_io_ns,
            "memory_bytes": metrics.memory_bytes,
            "disk_bytes": metrics.disk_bytes,
        }
