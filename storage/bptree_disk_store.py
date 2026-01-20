from __future__ import annotations

import bisect
import json
import shutil
import time
from collections import deque
from dataclasses import asdict
from pathlib import Path
from typing import Deque, Dict, List, Optional, Set, Tuple

from interfaces.storage import OperationMetrics
from models import Student

_NODE_FILENAME_TEMPLATE = "node_{node_id}.json"
_MANIFEST_FILENAME = "manifest.json"


class DiskBPlusTreeStore:
    """Disk-backed B+ tree implemented with JSON-serialized nodes."""

    def __init__(
        self,
        base_path: Path,
        order: int,
        snapshot_limit: Optional[int] = None,
        flush_threshold: Optional[int] = 32,
    ) -> None:
        self.base_path = base_path
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.nodes_path = self.base_path / "nodes"
        self.nodes_path.mkdir(parents=True, exist_ok=True)
        self.order = max(3, int(order))
        self.max_leaf_keys = self.order - 1
        self.max_internal_children = self.order
        self.snapshot_limit = snapshot_limit if snapshot_limit and snapshot_limit > 0 else None

        self.flush_threshold = int(flush_threshold or 0)
        if self.flush_threshold < 0:
            self.flush_threshold = 0
        self._pending_ops = 0
        self._node_cache: Dict[int, Dict[str, object]] = {}
        self._dirty_nodes: Set[int] = set()
        self._manifest_cache: Dict[str, object] = {}
        self._manifest_dirty = False

        self.operation_log: Deque[str] = deque(maxlen=100)

        self.root_id: int
        self.leaf_head_id: int
        self.next_node_id: int

        self._load_manifest()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def insert(self, student: Student) -> OperationMetrics:
        t0 = time.perf_counter_ns()
        action = self._upsert(student, log_action=True)
        self._after_write()
        t1 = time.perf_counter_ns()
        metrics = OperationMetrics(file_io_ns=t1 - t0)
        metrics.memory_bytes = self.memory_bytes()
        metrics.disk_bytes = self._disk_bytes()
        return metrics

    def update(self, student: Student) -> OperationMetrics:
        return self.insert(student)

    def delete(self, student_id: int) -> OperationMetrics:
        t0 = time.perf_counter_ns()
        removed = self._delete_record(student_id)
        if removed:
            self._after_write()
        t1 = time.perf_counter_ns()
        metrics = OperationMetrics(file_io_ns=t1 - t0)
        metrics.memory_bytes = self.memory_bytes()
        metrics.disk_bytes = self._disk_bytes()
        if removed:
            self._log(f"Student {student_id} deleted (disk).")
        else:
            self._log(f"Student {student_id} delete skipped (disk).")
        return metrics

    def search(self, student_id: int) -> tuple[Optional[Student], OperationMetrics]:
        t0 = time.perf_counter_ns()
        node_id = self.root_id
        while True:
            node = self._read_node(node_id)
            if node["type"] == "leaf":
                keys: List[int] = node["keys"]
                idx = bisect.bisect_left(keys, student_id)
                if idx < len(keys) and keys[idx] == student_id:
                    student = self._student_from_record(node["records"][idx])
                    t1 = time.perf_counter_ns()
                    metrics = OperationMetrics(file_io_ns=t1 - t0)
                    metrics.memory_bytes = self.memory_bytes()
                    metrics.disk_bytes = self._disk_bytes()
                    self._log(f"Student {student_id} located on disk.")
                    return student, metrics
                break
            else:
                keys: List[int] = node["keys"]
                children: List[int] = node["children"]
                index = bisect.bisect_right(keys, student_id)
                node_id = children[index]

        t1 = time.perf_counter_ns()
        metrics = OperationMetrics(file_io_ns=t1 - t0)
        metrics.memory_bytes = self.memory_bytes()
        metrics.disk_bytes = self._disk_bytes()
        self._log(f"Student {student_id} not present (disk).")
        return None, metrics

    def snapshot(self) -> Dict[str, object]:
        leaves: List[List[Dict[str, object]]] = []
        total_records = 0
        node_id = self.leaf_head_id
        collected = 0
        while node_id:
            leaf = self._read_leaf(node_id)
            records = [record.copy() for record in leaf["records"]]
            total_records += len(records)
            if self.snapshot_limit is not None:
                remaining = self.snapshot_limit - collected
                if remaining <= 0:
                    break
                if len(records) > remaining:
                    records = records[:remaining]
                collected += len(records)
            leaves.append(records)
            node_id = leaf.get("next_leaf") or 0
        return {
            "order": self.order,
            "leaves": leaves,
            "record_count": total_records,
            "operation_log": list(self.operation_log)[-10:],
        }

    def export_records(self) -> List[Dict[str, object]]:
        records: List[Dict[str, object]] = []
        node_id = self.leaf_head_id
        while node_id:
            leaf = self._read_leaf(node_id)
            records.extend(record.copy() for record in leaf["records"])
            node_id = leaf.get("next_leaf") or 0
        return records

    def reset(self) -> None:
        self._reinitialize_tree()
        self.operation_log.clear()

    def close(self) -> None:
        self._flush_dirty_nodes(force=True)

    def memory_bytes(self) -> int:
        total = 0
        for node in self._node_cache.values():
            total += len(json.dumps(self._node_payload(node)))
        if self._manifest_cache:
            total += len(json.dumps(self._manifest_cache))
        return total

    def disk_bytes(self) -> int:
        return self._disk_bytes()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _upsert(self, student: Student, *, log_action: bool) -> str:
        path = self._find_path(student.id)
        leaf_id, leaf = path[-1]
        keys: List[int] = leaf["keys"]
        records: List[Dict[str, object]] = leaf["records"]
        record_dict = self._record_from_student(student)
        idx = bisect.bisect_left(keys, student.id)
        if idx < len(keys) and keys[idx] == student.id:
            records[idx] = record_dict
            action = "updated"
        else:
            keys.insert(idx, student.id)
            records.insert(idx, record_dict)
            action = "inserted"
        self._write_leaf(leaf)

        if len(keys) > self.max_leaf_keys:
            promoted_key, new_leaf_id = self._split_leaf(leaf)
            self._propagate_split(path[:-1], promoted_key, new_leaf_id)

        self._write_manifest()
        if log_action:
            self._log(f"Student {student.id} {action} (disk).")
        return action

    def _split_leaf(self, leaf: Dict[str, object]) -> Tuple[int, int]:
        records: List[Dict[str, object]] = leaf["records"]
        split_index = len(records) // 2
        right_records = records[split_index:]
        left_records = records[:split_index]

        new_leaf_id = self._allocate_node_id()
        new_leaf = {
            "type": "leaf",
            "id": new_leaf_id,
            "records": right_records,
            "keys": [record["id"] for record in right_records],
            "next_leaf": leaf.get("next_leaf"),
        }

        leaf["records"] = left_records
        leaf["keys"] = [record["id"] for record in left_records]
        leaf["next_leaf"] = new_leaf_id

        self._write_leaf(leaf)
        self._write_leaf(new_leaf)

        promoted_key = new_leaf["keys"][0] if new_leaf["keys"] else leaf["keys"][-1]
        return promoted_key, new_leaf_id

    def _propagate_split(self, path: List[Tuple[int, Dict[str, object]]], promoted_key: int, new_child_id: int) -> None:
        for node_id, node in reversed(path):
            keys: List[int] = node["keys"]
            children: List[int] = node["children"]
            insert_index = bisect.bisect_right(keys, promoted_key)
            keys.insert(insert_index, promoted_key)
            children.insert(insert_index + 1, new_child_id)

            if len(children) <= self.max_internal_children:
                self._write_internal(node)
                self._write_manifest()
                return

            promoted_key, new_child_id = self._split_internal(node)
            self._write_internal(node)

        self._create_new_root(promoted_key, new_child_id)
        self._write_manifest()

    def _split_internal(self, node: Dict[str, object]) -> Tuple[int, int]:
        keys: List[int] = node["keys"]
        children: List[int] = node["children"]
        mid_index = len(keys) // 2
        promoted_key = keys[mid_index]

        right_keys = keys[mid_index + 1 :]
        right_children = children[mid_index + 1 :]

        node["keys"] = keys[:mid_index]
        node["children"] = children[: mid_index + 1]

        new_internal_id = self._allocate_node_id()
        new_internal = {
            "type": "internal",
            "id": new_internal_id,
            "keys": right_keys,
            "children": right_children,
        }
        self._write_internal(new_internal)
        return promoted_key, new_internal_id

    def _create_new_root(self, promoted_key: int, right_child_id: int) -> None:
        old_root_id = self.root_id
        new_root_id = self._allocate_node_id()
        new_root = {
            "type": "internal",
            "id": new_root_id,
            "keys": [promoted_key],
            "children": [old_root_id, right_child_id],
        }
        self._write_internal(new_root)
        self.root_id = new_root_id

    def _find_path(self, key: int) -> List[Tuple[int, Dict[str, object]]]:
        path: List[Tuple[int, Dict[str, object]]] = []
        node_id = self.root_id
        while True:
            node = self._read_node(node_id)
            path.append((node_id, node))
            if node["type"] == "leaf":
                break
            keys: List[int] = node["keys"]
            children: List[int] = node["children"]
            index = bisect.bisect_right(keys, key)
            node_id = children[index]
        return path

    def _gather_all_students(self) -> List[Student]:
        students: List[Student] = []
        node_id = self.leaf_head_id
        while node_id:
            leaf = self._read_leaf(node_id)
            students.extend(self._student_from_record(record) for record in leaf["records"])
            node_id = leaf.get("next_leaf") or 0
        return students

    def _delete_record(self, student_id: int) -> bool:
        path = self._find_path(student_id)
        leaf_id, leaf = path[-1]
        keys: List[int] = leaf["keys"]
        records: List[Dict[str, object]] = leaf["records"]
        idx = bisect.bisect_left(keys, student_id)
        if idx >= len(keys) or keys[idx] != student_id:
            return False

        next_leaf_id = int(leaf.get("next_leaf", 0) or 0)
        removed_first = idx == 0
        records.pop(idx)
        # Writing the leaf recalculates keys from records
        self._write_leaf(leaf)

        if leaf_id == self.leaf_head_id and not records and next_leaf_id:
            self.leaf_head_id = next_leaf_id

        if len(path) > 1 and records and removed_first:
            parent_id, parent = path[-2]
            parent = self._read_node(parent_id)
            try:
                child_index = parent["children"].index(leaf_id)
            except ValueError:
                child_index = -1
            if child_index > 0:
                parent_keys: List[int] = parent["keys"]
                parent_keys[child_index - 1] = records[0]["id"]
                self._write_internal(parent)

        self._write_manifest()
        return True

    def _reinitialize_tree(self) -> None:
        self._node_cache.clear()
        self._dirty_nodes.clear()
        self._pending_ops = 0
        if self.nodes_path.exists():
            shutil.rmtree(self.nodes_path)
        self.nodes_path.mkdir(parents=True, exist_ok=True)
        self.root_id = 1
        self.leaf_head_id = 1
        self.next_node_id = 2
        empty_leaf = {
            "type": "leaf",
            "id": self.root_id,
            "records": [],
            "keys": [],
            "next_leaf": 0,
        }
        self._write_leaf(empty_leaf)
        self._write_manifest()
        self._flush_dirty_nodes(force=True)

    def _allocate_node_id(self) -> int:
        node_id = self.next_node_id
        self.next_node_id += 1
        return node_id

    def _load_manifest(self) -> None:
        manifest_path = self.base_path / _MANIFEST_FILENAME
        if not manifest_path.exists():
            self._reinitialize_tree()
            return
        try:
            with manifest_path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
            self.root_id = int(data.get("root_id", 1))
            self.leaf_head_id = int(data.get("leaf_head_id", self.root_id))
            self.next_node_id = int(data.get("next_node_id", 2))
            self._manifest_cache = {
                "order": self.order,
                "root_id": self.root_id,
                "leaf_head_id": self.leaf_head_id,
                "next_node_id": self.next_node_id,
            }
            self._manifest_dirty = False
        except Exception:
            self._reinitialize_tree()
            return
        try:
            self._read_node(self.root_id)
        except FileNotFoundError:
            self._reinitialize_tree()

    def _write_manifest(self) -> None:
        if not self._manifest_cache:
            self._manifest_cache = {}
        self._manifest_cache["order"] = self.order
        self._manifest_cache["root_id"] = self.root_id
        self._manifest_cache["leaf_head_id"] = self.leaf_head_id
        self._manifest_cache["next_node_id"] = self.next_node_id
        self._manifest_dirty = True
        if self.flush_threshold == 0:
            self._persist_manifest()
            self._manifest_dirty = False

    def _persist_manifest(self) -> None:
        manifest_path = self.base_path / _MANIFEST_FILENAME
        payload = self._manifest_cache or {
            "order": self.order,
            "root_id": self.root_id,
            "leaf_head_id": self.leaf_head_id,
            "next_node_id": self.next_node_id,
        }
        with manifest_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)

    def _record_from_student(self, student: Student) -> Dict[str, object]:
        record = asdict(student)
        record["id"] = int(record["id"])
        return record

    def _student_from_record(self, record: Dict[str, object]) -> Student:
        return Student(
            id=int(record["id"]),
            gender=str(record.get("gender", "")),
            race_ethnicity=str(record.get("race_ethnicity", "")),
            parental_level_of_education=str(record.get("parental_level_of_education", "")),
            lunch=str(record.get("lunch", "")),
            test_preparation_course=str(record.get("test_preparation_course", "")),
            math_score=int(record.get("math_score", 0)),
            reading_score=int(record.get("reading_score", 0)),
            writing_score=int(record.get("writing_score", 0)),
        )

    def _read_node(self, node_id: int) -> Dict[str, object]:
        cached = self._node_cache.get(node_id)
        if cached is not None:
            return cached
        path = self.nodes_path / _NODE_FILENAME_TEMPLATE.format(node_id=node_id)
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if data.get("type") == "leaf":
            node = self._normalize_leaf_dict(data)
        else:
            node = self._normalize_internal_dict(data)
        self._store_node(node, dirty=False)
        return node

    def _read_leaf(self, node_id: int) -> Dict[str, object]:
        node = self._read_node(node_id)
        if node.get("type") != "leaf":
            raise ValueError(f"Node {node_id} is not a leaf")
        return node

    def _write_leaf(self, leaf: Dict[str, object]) -> None:
        leaf["type"] = "leaf"
        leaf["keys"] = [record["id"] for record in leaf["records"]]
        leaf["next_leaf"] = int(leaf.get("next_leaf", 0) or 0)
        self._store_node(leaf, dirty=True)

    def _write_internal(self, node: Dict[str, object]) -> None:
        node["type"] = "internal"
        self._store_node(node, dirty=True)

    def _normalize_leaf_dict(self, node: Dict[str, object]) -> Dict[str, object]:
        records = node.get("records", [])
        for record in records:
            record["id"] = int(record.get("id", 0))
            record["math_score"] = int(record.get("math_score", 0))
            record["reading_score"] = int(record.get("reading_score", 0))
            record["writing_score"] = int(record.get("writing_score", 0))
        node["records"] = records
        node["keys"] = [record["id"] for record in records]
        node["next_leaf"] = int(node.get("next_leaf", 0))
        node.setdefault("id", int(node.get("id", 0)))
        return node

    def _normalize_internal_dict(self, node: Dict[str, object]) -> Dict[str, object]:
        node["keys"] = [int(key) for key in node.get("keys", [])]
        node["children"] = [int(child) for child in node.get("children", [])]
        node.setdefault("id", int(node.get("id", 0)))
        return node

    def _store_node(self, node: Dict[str, object], *, dirty: bool) -> None:
        node_id = int(node["id"])
        self._node_cache[node_id] = node
        if not dirty:
            return
        if self.flush_threshold == 0:
            self._persist_node(node)
        else:
            self._dirty_nodes.add(node_id)

    def _persist_node(self, node: Dict[str, object]) -> None:
        path = self.nodes_path / _NODE_FILENAME_TEMPLATE.format(node_id=node["id"])
        payload = self._node_payload(node)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)

    def _node_payload(self, node: Dict[str, object]) -> Dict[str, object]:
        if node.get("type") == "leaf":
            return {
                "type": "leaf",
                "id": node["id"],
                "records": node["records"],
                "next_leaf": node.get("next_leaf", 0),
            }
        return {
            "type": "internal",
            "id": node["id"],
            "keys": node["keys"],
            "children": node["children"],
        }

    def _after_write(self) -> None:
        if self.flush_threshold <= 0:
            return
        self._pending_ops += 1
        if self._pending_ops >= self.flush_threshold:
            self._flush_dirty_nodes()

    def _flush_dirty_nodes(self, force: bool = False) -> None:
        if self.flush_threshold == 0 and not force:
            return
        if not force and not self._dirty_nodes and not self._manifest_dirty:
            return
        for node_id in list(self._dirty_nodes):
            node = self._node_cache.get(node_id)
            if node is None:
                continue
            self._persist_node(node)
        self._dirty_nodes.clear()
        if self._manifest_dirty or force:
            self._persist_manifest()
            self._manifest_dirty = False
        self._pending_ops = 0

    def _disk_bytes(self) -> int:
        total = 0
        manifest_path = self.base_path / _MANIFEST_FILENAME
        if manifest_path.exists():
            try:
                total += manifest_path.stat().st_size
            except OSError:
                pass
        if self.nodes_path.exists():
            for entry in self.nodes_path.glob("*.json"):
                try:
                    stem = entry.stem
                    node_id = None
                    if stem.startswith("node_"):
                        try:
                            node_id = int(stem.split("_", 1)[1])
                        except ValueError:
                            node_id = None
                    if node_id is not None and node_id in self._dirty_nodes:
                        continue
                    total += entry.stat().st_size
                except OSError:
                    continue
        if self._manifest_dirty:
            payload = self._manifest_cache or {
                "order": self.order,
                "root_id": self.root_id,
                "leaf_head_id": self.leaf_head_id,
                "next_node_id": self.next_node_id,
            }
            total += len(json.dumps(payload))
        if self._dirty_nodes:
            for node_id in self._dirty_nodes:
                node = self._node_cache.get(node_id)
                if not node:
                    continue
                total += len(json.dumps(self._node_payload(node)))
        return total

    def _log(self, message: str) -> None:
        self.operation_log.append(message)


__all__ = ["DiskBPlusTreeStore"]
