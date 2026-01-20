from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

from models import Student
from interfaces.storage import OperationMetrics


@dataclass(slots=True)
class _DiskEntry:
    id: int
    student: Optional[Student]
    is_tombstone: bool
    timestamp: int

    def to_json(self) -> Dict[str, object]:
        return {
            "id": self.id,
            "is_tombstone": self.is_tombstone,
            "timestamp": self.timestamp,
            "student": asdict(self.student) if self.student else None,
        }

    @classmethod
    def from_json(cls, payload: Dict[str, object]) -> "_DiskEntry":
        student_payload = payload.get("student")
        student = Student(**student_payload) if student_payload else None
        return cls(
            id=int(payload["id"]),
            student=student,
            is_tombstone=bool(payload.get("is_tombstone", False)),
            timestamp=int(payload.get("timestamp", 0)),
        )


class DiskLSMStore:
    """Disk-backed LSM tree with JSON SSTables and manifest metadata."""

    MANIFEST_FILENAME = "manifest.json"

    def __init__(
        self,
        base_path: Path,
        *,
        memtable_threshold: int,
        num_levels: int,
    ) -> None:
        self.base_path = base_path
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.memtable_threshold = max(4, memtable_threshold)
        self.num_levels = max(1, num_levels)
        self.level_table_limits = [max(1, 4 * (i + 1)) for i in range(self.num_levels)]
        self.memtable: Dict[int, _DiskEntry] = {}
        self.levels: List[List[str]] = [[] for _ in range(self.num_levels)]
        self.flush_count = 0
        self.compaction_count = 0
        self._timestamp = 0
        self._manifest_path = self.base_path / self.MANIFEST_FILENAME
        self._load_manifest()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def insert(self, student: Student) -> OperationMetrics:
        t0 = time.perf_counter_ns()
        entry = self._make_entry(student.id, student, is_tombstone=False)
        self.memtable[student.id] = entry
        t1 = time.perf_counter_ns()
        metrics = OperationMetrics(in_memory_ns=t1 - t0)
        if len(self.memtable) >= self.memtable_threshold:
            flush_ns = self._flush_memtable()
            if flush_ns:
                metrics.file_io_ns += flush_ns
        metrics.memory_bytes = self._memory_bytes()
        metrics.disk_bytes = self._disk_bytes()
        return metrics

    def update(self, student: Student) -> OperationMetrics:
        return self.insert(student)

    def delete(self, student_id: int) -> OperationMetrics:
        t0 = time.perf_counter_ns()
        entry = self._make_entry(student_id, None, is_tombstone=True)
        self.memtable[student_id] = entry
        t1 = time.perf_counter_ns()
        metrics = OperationMetrics(in_memory_ns=t1 - t0)
        if len(self.memtable) >= self.memtable_threshold:
            flush_ns = self._flush_memtable()
            if flush_ns:
                metrics.file_io_ns += flush_ns
        metrics.memory_bytes = self._memory_bytes()
        metrics.disk_bytes = self._disk_bytes()
        return metrics

    def search(self, student_id: int) -> tuple[Optional[Student], OperationMetrics]:
        t0 = time.perf_counter_ns()
        entry = self.memtable.get(student_id)
        if entry is not None:
            t1 = time.perf_counter_ns()
            metrics = OperationMetrics(in_memory_ns=t1 - t0)
            metrics.memory_bytes = self._memory_bytes()
            metrics.disk_bytes = self._disk_bytes()
            return (None if entry.is_tombstone else entry.student, metrics)

        for level in self.levels:
            for table_name in reversed(level):
                table_entries = self._read_table(table_name)
                found = self._binary_search(table_entries, student_id)
                if found:
                    t1 = time.perf_counter_ns()
                    metrics = OperationMetrics(file_io_ns=t1 - t0)
                    metrics.memory_bytes = self._memory_bytes()
                    metrics.disk_bytes = self._disk_bytes()
                    return (None if found.is_tombstone else found.student, metrics)
        t1 = time.perf_counter_ns()
        metrics = OperationMetrics(file_io_ns=t1 - t0)
        metrics.memory_bytes = self._memory_bytes()
        metrics.disk_bytes = self._disk_bytes()
        return (None, metrics)

    def flush_memtable(self) -> OperationMetrics:
        flush_ns = self._flush_memtable()
        metrics = OperationMetrics(file_io_ns=flush_ns)
        metrics.memory_bytes = self._memory_bytes()
        metrics.disk_bytes = self._disk_bytes()
        return metrics

    def compact_levels(self) -> OperationMetrics:
        t0 = time.perf_counter_ns()
        for level_index in range(self.num_levels):
            self._maybe_compact_level(level_index)
        t1 = time.perf_counter_ns()
        metrics = OperationMetrics(file_io_ns=t1 - t0)
        metrics.memory_bytes = self._memory_bytes()
        metrics.disk_bytes = self._disk_bytes()
        return metrics

    def snapshot(self) -> Dict[str, object]:
        snapshot_levels: List[List[List[Dict[str, object]]]] = []
        for level in self.levels:
            level_snapshot: List[List[Dict[str, object]]] = []
            for table_name in level:
                entries = self._read_table(table_name)
                level_snapshot.append([self._entry_to_dict(entry) for entry in entries[:100]])
            snapshot_levels.append(level_snapshot)
        memtable_snapshot = [self._entry_to_dict(entry) for entry in self._sorted_memtable()]
        return {
            "memtable": memtable_snapshot,
            "levels": snapshot_levels,
            "flush_count": self.flush_count,
            "compaction_count": self.compaction_count,
        }

    def export_memtable(self) -> List[Dict[str, object]]:
        return [self._entry_to_dict(entry) for entry in self._sorted_memtable() if not entry.is_tombstone]

    def reset(self) -> None:
        self.memtable.clear()
        for level in self.levels:
            for table_name in level:
                self._remove_table(table_name)
        self.levels = [[] for _ in range(self.num_levels)]
        self.flush_count = 0
        self.compaction_count = 0
        self._timestamp = 0
        self._write_manifest()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _make_entry(self, student_id: int, student: Optional[Student], *, is_tombstone: bool) -> _DiskEntry:
        self._timestamp += 1
        return _DiskEntry(id=student_id, student=student, is_tombstone=is_tombstone, timestamp=self._timestamp)

    def _sorted_memtable(self) -> List[_DiskEntry]:
        return sorted(self.memtable.values(), key=lambda entry: entry.id)

    def _flush_memtable(self) -> int:
        if not self.memtable:
            return 0
        t0 = time.perf_counter_ns()
        entries = self._sorted_memtable()
        table_name = self._generate_table_name(level=0)
        self._write_table(table_name, entries)
        self.memtable.clear()
        self.levels[0].append(table_name)
        self.flush_count += 1
        self._write_manifest()
        self._maybe_compact_level(0)
        t1 = time.perf_counter_ns()
        return t1 - t0

    def _maybe_compact_level(self, level_index: int) -> None:
        if level_index >= self.num_levels:
            return
        level = self.levels[level_index]
        limit = self.level_table_limits[level_index]
        if len(level) <= limit:
            return
        if not level:
            return

        merged: Dict[int, _DiskEntry] = {}
        for table_name in level:
            for entry in self._read_table(table_name):
                existing = merged.get(entry.id)
                if existing is None or entry.timestamp > existing.timestamp:
                    merged[entry.id] = entry

        for table_name in level:
            self._remove_table(table_name)
        self.levels[level_index] = []

        merged_entries = sorted(merged.values(), key=lambda entry: entry.id)
        next_level = min(level_index + 1, self.num_levels - 1)
        table_name = self._generate_table_name(level=next_level)
        self._write_table(table_name, merged_entries)
        self.levels[next_level].append(table_name)
        self.compaction_count += 1
        self._write_manifest()

        if next_level != level_index:
            self._maybe_compact_level(next_level)

    def _generate_table_name(self, *, level: int) -> str:
        timestamp = int(time.time() * 1_000_000)
        return f"level{level}_{timestamp}.jsonl"

    def _write_table(self, table_name: str, entries: List[_DiskEntry]) -> None:
        path = self.base_path / table_name
        with path.open("w", encoding="utf-8") as handle:
            for entry in entries:
                json.dump(entry.to_json(), handle, separators=(",", ":"))
                handle.write("\n")

    def _read_table(self, table_name: str) -> List[_DiskEntry]:
        path = self.base_path / table_name
        if not path.exists():
            return []
        entries: List[_DiskEntry] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                payload = json.loads(line)
                entries.append(_DiskEntry.from_json(payload))
        return entries

    def _remove_table(self, table_name: str) -> None:
        path = self.base_path / table_name
        if path.exists():
            try:
                path.unlink()
            except OSError:
                pass

    def _binary_search(self, entries: List[_DiskEntry], student_id: int) -> Optional[_DiskEntry]:
        low = 0
        high = len(entries) - 1
        while low <= high:
            mid = (low + high) // 2
            mid_id = entries[mid].id
            if mid_id == student_id:
                return entries[mid]
            if mid_id < student_id:
                low = mid + 1
            else:
                high = mid - 1
        return None

    def _entry_to_dict(self, entry: _DiskEntry) -> Dict[str, object]:
        if entry.is_tombstone or entry.student is None:
            return {
                "id": entry.id,
                "gender": "-",
                "race_ethnicity": "-",
                "math_score": "-",
                "reading_score": "-",
                "writing_score": "-",
                "tombstone": True,
            }
        data = asdict(entry.student)
        data["tombstone"] = False
        return data

    def _memory_bytes(self) -> int:
        size = sys.getsizeof(self.memtable)
        for entry in self.memtable.values():
            size += sys.getsizeof(entry)
            if entry.student:
                size += sys.getsizeof(entry.student)
        return size

    def _disk_bytes(self) -> int:
        total = 0
        for level in self.levels:
            for table_name in level:
                path = self.base_path / table_name
                if path.exists():
                    try:
                        total += path.stat().st_size
                    except OSError:
                        continue
        if self._manifest_path.exists():
            try:
                total += self._manifest_path.stat().st_size
            except OSError:
                pass
        return total

    def _load_manifest(self) -> None:
        if not self._manifest_path.exists():
            self._write_manifest()
            return
        try:
            with self._manifest_path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
        except Exception:
            self._write_manifest()
            return
        self.levels = [list(level) for level in data.get("levels", [[] for _ in range(self.num_levels)])]
        if len(self.levels) < self.num_levels:
            self.levels.extend([[] for _ in range(self.num_levels - len(self.levels))])
        sanitized: List[List[str]] = []
        for level in self.levels:
            sanitized.append([name for name in level if (self.base_path / name).exists()])
        self.levels = sanitized
        self.flush_count = int(data.get("flush_count", 0))
        self.compaction_count = int(data.get("compaction_count", 0))
        self._timestamp = int(data.get("timestamp", 0))

    def _write_manifest(self) -> None:
        data = {
            "levels": self.levels,
            "flush_count": self.flush_count,
            "compaction_count": self.compaction_count,
            "timestamp": self._timestamp,
        }
        with self._manifest_path.open("w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2)

    def memory_bytes(self) -> int:
        return self._memory_bytes()

    def disk_bytes(self) -> int:
        return self._disk_bytes()


__all__ = ["DiskLSMStore"]
