from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional

from .student import Student


@dataclass(slots=True)
class _LSMEntry:
    """Internal representation of a row (or tombstone) within the LSM."""

    id: int
    student: Optional[Student]
    is_tombstone: bool
    timestamp: int

    def to_snapshot_dict(self) -> Dict[str, object]:
        if self.is_tombstone or self.student is None:
            return {
                "id": self.id,
                "gender": "—",
                "race_ethnicity": "—",
                "math_score": "—",
                "reading_score": "—",
                "writing_score": "—",
                "tombstone": True,
            }
        data = asdict(self.student)
        data["tombstone"] = False
        return data


class LSMTreeModel:
    """In-memory LSM tree with leveled SSTables and tombstone-aware compaction."""

    def __init__(self, memtable_threshold: int, num_levels: int) -> None:
        self.memtable_threshold = max(4, memtable_threshold)
        self.num_levels = max(1, num_levels)
        self.memtable: Dict[int, _LSMEntry] = {}
        self.levels: List[List[List[_LSMEntry]]] = [[] for _ in range(self.num_levels)]
        self.flush_count = 0
        self.compaction_count = 0
        self._level_table_limits = [max(1, 4 * (i + 1)) for i in range(self.num_levels)]
        self._timestamp = 0

    # ------------------------------------------------------------------
    # Mutating operations
    # ------------------------------------------------------------------
    def insert(self, student: Student) -> Dict[str, int]:
        t0 = time.perf_counter_ns()
        entry = self._make_entry(student.id, student, is_tombstone=False)
        self.memtable[student.id] = entry
        t1 = time.perf_counter_ns()
        metrics = {"in_memory_ns": t1 - t0}
        if len(self.memtable) >= self.memtable_threshold:
            flush_ns = self._flush_memtable()
            if flush_ns:
                metrics["file_io_ns"] = flush_ns
        return metrics

    def update(self, student: Student) -> Dict[str, int]:
        return self.insert(student)

    def delete(self, student_id: int) -> Dict[str, int]:
        t0 = time.perf_counter_ns()
        entry = self._make_entry(student_id, None, is_tombstone=True)
        self.memtable[student_id] = entry
        t1 = time.perf_counter_ns()
        metrics = {"in_memory_ns": t1 - t0}
        if len(self.memtable) >= self.memtable_threshold:
            flush_ns = self._flush_memtable()
            if flush_ns:
                metrics["file_io_ns"] = flush_ns
        return metrics

    def flush_memtable(self) -> int:
        """Force a memtable flush and return the duration in nanoseconds."""
        if not self.memtable:
            return 0
        return self._flush_memtable()

    def compact_levels(self) -> int:
        """Force compaction across all levels and return total duration (ns)."""
        t0 = time.perf_counter_ns()
        for level_index in range(self.num_levels):
            self._maybe_compact_level(level_index)
        t1 = time.perf_counter_ns()
        return t1 - t0

    def search(self, student_id: int) -> Optional[Student]:
        in_mem = self.memtable.get(student_id)
        if in_mem:
            return None if in_mem.is_tombstone else in_mem.student
        for level in self.levels:
            # Search newer SSTables first because we append fresh runs to the tail
            for table in reversed(level):
                entry = self._binary_search_table(table, student_id)
                if entry is not None:
                    return None if entry.is_tombstone else entry.student
        return None

    def reset(self) -> None:
        self.memtable.clear()
        self.levels = [[] for _ in range(self.num_levels)]
        self.flush_count = 0
        self.compaction_count = 0
        self._timestamp = 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _make_entry(self, student_id: int, student: Optional[Student], *, is_tombstone: bool) -> _LSMEntry:
        self._timestamp += 1
        return _LSMEntry(id=student_id, student=student, is_tombstone=is_tombstone, timestamp=self._timestamp)

    def _flush_memtable(self) -> int:
        if not self.memtable:
            return 0
        t0 = time.perf_counter_ns()
        sorted_entries = sorted(self.memtable.values(), key=lambda entry: entry.id)
        self.memtable.clear()
        self.levels[0].append(sorted_entries)
        self.flush_count += 1
        self._maybe_compact_level(0)
        t1 = time.perf_counter_ns()
        return t1 - t0

    def _maybe_compact_level(self, level_index: int) -> None:
        if level_index >= self.num_levels:
            return
        tables = self.levels[level_index]
        limit = self._level_table_limits[level_index] if level_index < len(self._level_table_limits) else None
        if limit is not None and len(tables) <= limit:
            return
        if not tables:
            return

        merged_map: Dict[int, _LSMEntry] = {}
        for table in tables:
            for entry in table:
                existing = merged_map.get(entry.id)
                if existing is None or entry.timestamp > existing.timestamp:
                    merged_map[entry.id] = entry

        merged_table = sorted(merged_map.values(), key=lambda entry: entry.id)
        self.levels[level_index] = []

        next_level_index = min(level_index + 1, self.num_levels - 1)
        self.levels[next_level_index].append(merged_table)
        self.compaction_count += 1

        if next_level_index != level_index:
            self._maybe_compact_level(next_level_index)

    @staticmethod
    def _binary_search_table(table: List[_LSMEntry], student_id: int) -> Optional[_LSMEntry]:
        low = 0
        high = len(table) - 1
        while low <= high:
            mid = (low + high) // 2
            mid_id = table[mid].id
            if mid_id == student_id:
                return table[mid]
            if mid_id < student_id:
                low = mid + 1
            else:
                high = mid - 1
        return None

    # ------------------------------------------------------------------
    # Snapshot utilities
    # ------------------------------------------------------------------
    def snapshot(self) -> Dict[str, object]:
        return {
            "memtable": [entry.to_snapshot_dict() for entry in self._sorted_memtable()],
            "levels": [
                [[entry.to_snapshot_dict() for entry in table] for table in level]
                for level in self.levels
            ],
            "flush_count": self.flush_count,
            "compaction_count": self.compaction_count,
        }

    def export_memtable(self) -> List[Dict[str, object]]:
        return [entry.to_snapshot_dict() for entry in self._sorted_memtable() if not entry.is_tombstone]

    def _sorted_memtable(self) -> List[_LSMEntry]:
        return sorted(self.memtable.values(), key=lambda entry: entry.id)

    # ------------------------------------------------------------------
    # Serialization helpers for disk persistence
    # ------------------------------------------------------------------
    def to_state(self) -> Dict[str, object]:
        return {
            "memtable": [self._entry_to_state(entry) for entry in self.memtable.values()],
            "levels": [
                [[self._entry_to_state(entry) for entry in table] for table in level]
                for level in self.levels
            ],
            "flush_count": self.flush_count,
            "compaction_count": self.compaction_count,
            "timestamp": self._timestamp,
            "memtable_threshold": self.memtable_threshold,
            "num_levels": self.num_levels,
        }

    @classmethod
    def from_state(cls, state: Dict[str, object]) -> "LSMTreeModel":
        memtable_threshold = int(state.get("memtable_threshold", 4))
        num_levels = int(state.get("num_levels", 1))
        model = cls(memtable_threshold, num_levels)

        model.memtable = {
            entry_state["id"]: model._entry_from_state(entry_state)
            for entry_state in state.get("memtable", [])
        }

        levels_state = state.get("levels", [])
        model.levels = []
        for level_tables in levels_state:
            tables: List[List[_LSMEntry]] = []
            for table_entries in level_tables:
                table = [model._entry_from_state(entry_state) for entry_state in table_entries]
                tables.append(table)
            model.levels.append(tables)

        model.flush_count = int(state.get("flush_count", 0))
        model.compaction_count = int(state.get("compaction_count", 0))
        model._timestamp = int(state.get("timestamp", 0))
        return model

    def _entry_to_state(self, entry: _LSMEntry) -> Dict[str, object]:
        return {
            "id": entry.id,
            "is_tombstone": entry.is_tombstone,
            "timestamp": entry.timestamp,
            "student": asdict(entry.student) if entry.student else None,
        }

    def _entry_from_state(self, state: Dict[str, object]) -> _LSMEntry:
        student_state = state.get("student")
        student = Student(**student_state) if student_state else None
        return _LSMEntry(
            id=int(state.get("id")),
            student=student,
            is_tombstone=bool(state.get("is_tombstone", False)),
            timestamp=int(state.get("timestamp", 0)),
        )
