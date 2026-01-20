from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import DefaultDict, Dict, Union


Numeric = Union[int, float]


@dataclass
class MetricTotals:
    count: int = 0
    in_memory_ns: int = 0
    file_io_ns: int = 0
    memory_bytes: int = 0
    disk_bytes: int = 0

    def record(self, timings: Dict[str, int]) -> None:
        self.count += 1
        self.in_memory_ns += int(timings.get("in_memory_ns", 0))
        self.file_io_ns += int(timings.get("file_io_ns", 0))
        self.memory_bytes += int(timings.get("memory_bytes", 0))
        self.disk_bytes += int(timings.get("disk_bytes", 0))

    def report(self) -> Dict[str, Numeric]:
        if self.count == 0:
            avg_in_mem = 0.0
            avg_file = 0.0
            avg_mem_bytes = 0.0
            avg_disk_bytes = 0.0
        else:
            avg_in_mem = self.in_memory_ns / self.count
            avg_file = self.file_io_ns / self.count
            avg_mem_bytes = self.memory_bytes / self.count
            avg_disk_bytes = self.disk_bytes / self.count

        return {
            "count": self.count,
            "in_memory_ns": avg_in_mem,
            "file_io_ns": avg_file,
            "memory_bytes": avg_mem_bytes,
            "disk_bytes": avg_disk_bytes,
            "total_in_memory_ns": self.in_memory_ns,
            "total_file_io_ns": self.file_io_ns,
            "total_memory_bytes": self.memory_bytes,
            "total_disk_bytes": self.disk_bytes,
        }


OPERATIONS = ("insert", "update", "delete", "search")


class BenchmarkRecorder:
    """Aggregates timing metrics for different structures and operations."""

    def __init__(self) -> None:
        self._totals: DefaultDict[str, DefaultDict[str, MetricTotals]] = defaultdict(
            lambda: defaultdict(MetricTotals)
        )

    def record(self, structure: str, operation: str, timings: Dict[str, int]) -> None:
        self._totals[structure][operation].record(timings)

    def summary(self) -> Dict[str, Dict[str, Dict[str, Numeric]]]:
        report: Dict[str, Dict[str, Dict[str, Numeric]]] = {}
        for structure, per_op in self._totals.items():
            structure_report: Dict[str, Dict[str, Numeric]] = {}
            for operation in OPERATIONS:
                totals = per_op.get(operation)
                if totals is None:
                    structure_report[operation] = MetricTotals().report()
                else:
                    structure_report[operation] = totals.report()
            report[structure] = structure_report
        return report

    def reset(self) -> None:
        self._totals.clear()
