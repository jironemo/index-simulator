from __future__ import annotations

import bisect
import time
from dataclasses import asdict
from typing import Dict, List, Optional, Tuple, Union

from .student import Student


class _BPlusTreeNode:
    __slots__ = ("keys", "is_leaf")

    def __init__(self, *, is_leaf: bool) -> None:
        self.keys: List[int] = []
        self.is_leaf = is_leaf


class _LeafNode(_BPlusTreeNode):
    __slots__ = ("values", "next_leaf")

    def __init__(self) -> None:
        super().__init__(is_leaf=True)
        self.values: List[Student] = []
        self.next_leaf: Optional[_LeafNode] = None

    def __getstate__(self) -> Dict[str, object]:
        return {"keys": self.keys, "values": self.values}

    def __setstate__(self, state: Dict[str, object]) -> None:
        self.__init__()
        self.keys = list(state.get("keys", []))
        self.values = list(state.get("values", []))
        self.next_leaf = None

    def to_state(self) -> Dict[str, object]:
        return {
            "type": "leaf",
            "keys": list(self.keys),
            "values": [asdict(student) for student in self.values],
        }

    @classmethod
    def from_state(cls, state: Dict[str, object]) -> "_LeafNode":
        node = cls()
        node.keys = list(state.get("keys", []))
        node.values = [Student(**value) for value in state.get("values", [])]
        node.next_leaf = None
        return node


class _InternalNode(_BPlusTreeNode):
    __slots__ = ("children",)

    def __init__(self) -> None:
        super().__init__(is_leaf=False)
        self.children: List[_BPlusTreeNode] = []

    def __getstate__(self) -> Dict[str, object]:
        return {"keys": self.keys, "children": self.children}

    def __setstate__(self, state: Dict[str, object]) -> None:
        self.__init__()
        self.keys = list(state.get("keys", []))
        self.children = list(state.get("children", []))

    def to_state(self) -> Dict[str, object]:
        return {
            "type": "internal",
            "keys": list(self.keys),
            "children": [self._child_to_state(child) for child in self.children],
        }

    @staticmethod
    def _child_to_state(child: _BPlusTreeNode) -> Dict[str, object]:
        if child.is_leaf:
            return child.to_state()  # type: ignore[return-value]
        return child.to_state()  # type: ignore[return-value]

    @classmethod
    def from_state(cls, state: Dict[str, object]) -> "_InternalNode":
        node = cls()
        node.keys = list(state.get("keys", []))
        children_state = state.get("children", [])
        node.children = [BPlusTreeModel._node_from_state(child_state) for child_state in children_state]
        return node


class BPlusTreeModel:
    """In-memory B+ tree supporting insert/update/delete/search with node splitting."""

    def __init__(self, order: int = 4) -> None:
        # ``order`` represents the maximum number of children for internal nodes.
        self.order = max(3, order)
        self.max_leaf_keys = self.order - 1
        self.max_internal_children = self.order
        self.min_leaf_keys = max(1, (self.max_leaf_keys + 1) // 2)
        self.min_internal_children = max(2, (self.max_internal_children + 1) // 2)

        self.root: _BPlusTreeNode = _LeafNode()
        self._first_leaf: _LeafNode = self.root  # type: ignore[assignment]
        self.operation_log: List[str] = []

    def __getstate__(self) -> Dict[str, object]:
        return {
            "order": self.order,
            "root": self.root,
            "operation_log": self.operation_log,
        }

    def __setstate__(self, state: Dict[str, object]) -> None:
        self.__init__(order=state.get("order", 4))
        self.root = state.get("root", self.root)
        self.operation_log = list(state.get("operation_log", []))
        self._first_leaf = self._locate_first_leaf()
        self._rebuild_leaf_links()

    def to_state(self) -> Dict[str, object]:
        return {
            "order": self.order,
            "root": self._node_to_state(self.root),
            "operation_log": list(self.operation_log),
        }

    @classmethod
    def from_state(cls, state: Dict[str, object]) -> "BPlusTreeModel":
        model = cls(order=int(state.get("order", 4)))
        model.operation_log = list(state.get("operation_log", []))
        root_state = state.get("root")
        if root_state:
            model.root = cls._node_from_state(root_state)
        model._first_leaf = model._locate_first_leaf()
        model._rebuild_leaf_links()
        return model

    # ------------------------------------------------------------------
    # Mutating operations
    # ------------------------------------------------------------------
    def insert(self, student: Student) -> Dict[str, int]:
        t0 = time.perf_counter_ns()
        action, promoted = self._insert(self.root, student)
        if promoted is not None:
            key, sibling = promoted
            new_root = _InternalNode()
            new_root.keys = [key]
            new_root.children = [self.root, sibling]
            self.root = new_root
        self._first_leaf = self._locate_first_leaf()
        t1 = time.perf_counter_ns()
        self._log(f"Student {student.id} {action}.")
        return {"in_memory_ns": t1 - t0}

    def update(self, student: Student) -> Dict[str, int]:
        return self.insert(student)

    def delete(self, student_id: int) -> Dict[str, int]:
        t0 = time.perf_counter_ns()
        leaf, path = self._find_leaf_with_path(student_id)
        idx = bisect.bisect_left(leaf.keys, student_id)
        if idx >= len(leaf.keys) or leaf.keys[idx] != student_id:
            self._log(f"Student {student_id} delete skipped (not found).")
            t1 = time.perf_counter_ns()
            return {"in_memory_ns": t1 - t0}

        leaf.keys.pop(idx)
        leaf.values.pop(idx)
        self._log(f"Student {student_id} deleted.")
        self._rebalance_after_delete(leaf, path)
        self._first_leaf = self._locate_first_leaf()
        t1 = time.perf_counter_ns()
        return {"in_memory_ns": t1 - t0}

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------
    def search(self, student_id: int) -> Optional[Student]:
        leaf, _ = self._find_leaf_with_path(student_id)
        idx = bisect.bisect_left(leaf.keys, student_id)
        if idx < len(leaf.keys) and leaf.keys[idx] == student_id:
            self._log(f"Student {student_id} located in leaf node.")
            return leaf.values[idx]
        self._log(f"Student {student_id} not present.")
        return None

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------
    def reset(self) -> None:
        self.root = _LeafNode()
        self._first_leaf = self.root  # type: ignore[assignment]
        self.operation_log.clear()
        self._rebuild_leaf_links()

    # ------------------------------------------------------------------
    # Snapshot utilities
    # ------------------------------------------------------------------
    def snapshot(self) -> Dict[str, object]:
        leaves: List[List[Dict[str, object]]] = []
        node: Optional[_LeafNode] = self._first_leaf
        while node:
            leaves.append([asdict(student) for student in node.values])
            node = node.next_leaf
        return {
            "order": self.order,
            "leaves": leaves,
            "record_count": sum(len(leaf) for leaf in leaves),
            "operation_log": list(self.operation_log[-10:]),
        }

    def export_records(self) -> List[Dict[str, object]]:
        records: List[Dict[str, object]] = []
        node: Optional[_LeafNode] = self._first_leaf
        while node:
            records.extend(asdict(student) for student in node.values)
            node = node.next_leaf
        return records

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _insert(
        self,
        node: _BPlusTreeNode,
        student: Student,
    ) -> Tuple[str, Optional[Tuple[int, _BPlusTreeNode]]]:
        if node.is_leaf:
            leaf = node  # type: ignore[assignment]
            idx = bisect.bisect_left(leaf.keys, student.id)
            if idx < len(leaf.keys) and leaf.keys[idx] == student.id:
                leaf.values[idx] = student
                action = "updated"
            else:
                leaf.keys.insert(idx, student.id)
                leaf.values.insert(idx, student)
                action = "inserted"
            if len(leaf.keys) > self.max_leaf_keys:
                promoted = self._split_leaf(leaf)
                return action, promoted
            return action, None

        internal = node  # type: ignore[assignment]
        child_index = bisect.bisect_right(internal.keys, student.id)
        child = internal.children[child_index]
        action, promoted = self._insert(child, student)
        if promoted is not None:
            promoted_key, new_child = promoted
            internal.keys.insert(child_index, promoted_key)
            internal.children.insert(child_index + 1, new_child)
            if len(internal.children) > self.max_internal_children:
                return action, self._split_internal(internal)
        return action, None

    def _split_leaf(self, leaf: _LeafNode) -> Tuple[int, _LeafNode]:
        split_index = len(leaf.keys) // 2
        sibling = _LeafNode()
        sibling.keys = leaf.keys[split_index:]
        sibling.values = leaf.values[split_index:]
        leaf.keys = leaf.keys[:split_index]
        leaf.values = leaf.values[:split_index]

        sibling.next_leaf = leaf.next_leaf
        leaf.next_leaf = sibling
        if self._first_leaf is leaf and leaf.keys:
            # unchanged
            pass
        elif self._first_leaf is leaf and not leaf.keys:
            self._first_leaf = sibling
        return sibling.keys[0], sibling

    def _split_internal(self, internal: _InternalNode) -> Tuple[int, _InternalNode]:
        mid_index = len(internal.keys) // 2
        promoted_key = internal.keys[mid_index]
        sibling = _InternalNode()
        sibling.keys = internal.keys[mid_index + 1 :]
        sibling.children = internal.children[mid_index + 1 :]

        internal.keys = internal.keys[:mid_index]
        internal.children = internal.children[: mid_index + 1]
        return promoted_key, sibling

    def _find_leaf_with_path(
        self,
        student_id: int,
    ) -> Tuple[_LeafNode, List[Tuple[_InternalNode, int]]]:
        node = self.root
        path: List[Tuple[_InternalNode, int]] = []
        while not node.is_leaf:
            internal = node  # type: ignore[assignment]
            child_index = bisect.bisect_right(internal.keys, student_id)
            path.append((internal, child_index))
            node = internal.children[child_index]
        return node, path  # type: ignore[return-value]

    def _rebalance_after_delete(self, node: _LeafNode, path: List[Tuple[_InternalNode, int]]) -> None:
        current: Union[_LeafNode, _InternalNode] = node
        while path:
            parent, index = path.pop()
            if isinstance(current, _LeafNode):
                merged = self._rebalance_leaf(parent, index, current)
                if not merged:
                    break
                current = parent
                continue
            merged = self._rebalance_internal(parent, index, current)
            if not merged:
                break
            current = parent

        if isinstance(self.root, _InternalNode) and len(self.root.children) == 1:
            self.root = self.root.children[0]

    def _rebalance_leaf(self, parent: _InternalNode, index: int, node: _LeafNode) -> bool:
        if len(node.keys) >= self.min_leaf_keys or parent is None:
            self._recompute_parent_keys(parent)
            return False

        merged = False
        left_sibling = parent.children[index - 1] if index > 0 else None
        right_sibling = parent.children[index + 1] if index + 1 < len(parent.children) else None

        if isinstance(left_sibling, _LeafNode) and len(left_sibling.keys) > self.min_leaf_keys:
            node.keys.insert(0, left_sibling.keys.pop())
            node.values.insert(0, left_sibling.values.pop())
            self._recompute_parent_keys(parent)
            return False

        if isinstance(right_sibling, _LeafNode) and len(right_sibling.keys) > self.min_leaf_keys:
            node.keys.append(right_sibling.keys.pop(0))
            node.values.append(right_sibling.values.pop(0))
            self._recompute_parent_keys(parent)
            return False

        if isinstance(left_sibling, _LeafNode):
            left_sibling.keys.extend(node.keys)
            left_sibling.values.extend(node.values)
            left_sibling.next_leaf = node.next_leaf
            parent.keys.pop(index - 1)
            parent.children.pop(index)
            merged = True
            node = left_sibling
        elif isinstance(right_sibling, _LeafNode):
            node.keys.extend(right_sibling.keys)
            node.values.extend(right_sibling.values)
            node.next_leaf = right_sibling.next_leaf
            parent.keys.pop(index)
            parent.children.pop(index + 1)
            merged = True

        self._recompute_parent_keys(parent)
        return merged

    def _rebalance_internal(self, parent: _InternalNode, index: int, node: _InternalNode) -> bool:
        if len(node.children) >= self.min_internal_children or parent is None:
            self._recompute_parent_keys(parent)
            return False

        merged = False
        left_sibling = parent.children[index - 1] if index > 0 else None
        right_sibling = parent.children[index + 1] if index + 1 < len(parent.children) else None

        if isinstance(left_sibling, _InternalNode) and len(left_sibling.children) > self.min_internal_children:
            borrow_key = parent.keys[index - 1]
            node.keys.insert(0, borrow_key)
            node.children.insert(0, left_sibling.children.pop())
            parent.keys[index - 1] = left_sibling.keys.pop()
            self._recompute_parent_keys(parent)
            return False

        if isinstance(right_sibling, _InternalNode) and len(right_sibling.children) > self.min_internal_children:
            borrow_key = parent.keys[index]
            node.keys.append(borrow_key)
            node.children.append(right_sibling.children.pop(0))
            parent.keys[index] = right_sibling.keys.pop(0)
            self._recompute_parent_keys(parent)
            return False

        if isinstance(left_sibling, _InternalNode):
            separator = parent.keys.pop(index - 1)
            left_sibling.keys.append(separator)
            left_sibling.keys.extend(node.keys)
            left_sibling.children.extend(node.children)
            parent.children.pop(index)
            merged = True
            node = left_sibling
        elif isinstance(right_sibling, _InternalNode):
            separator = parent.keys.pop(index)
            node.keys.append(separator)
            node.keys.extend(right_sibling.keys)
            node.children.extend(right_sibling.children)
            parent.children.pop(index + 1)
            merged = True

        self._recompute_parent_keys(parent)
        return merged

    def _recompute_parent_keys(self, parent: Optional[_InternalNode]) -> None:
        if parent is None:
            return
        for i in range(len(parent.keys)):
            # Parent separators should mirror the smallest key in the right subtree
            right_child = parent.children[i + 1]
            min_key = self._subtree_min_key(right_child)
            if min_key is not None:
                parent.keys[i] = min_key
                continue
            left_child = parent.children[i]
            max_key = self._subtree_max_key(left_child)
            if max_key is not None:
                parent.keys[i] = max_key

    def _subtree_min_key(self, node: _BPlusTreeNode) -> Optional[int]:
        current = node
        while not current.is_leaf and getattr(current, "children", None):
            current = current.children[0]  # type: ignore[index]
        return current.keys[0] if current.keys else None

    def _subtree_max_key(self, node: _BPlusTreeNode) -> Optional[int]:
        current = node
        while not current.is_leaf and getattr(current, "children", None):
            current = current.children[-1]  # type: ignore[index]
        return current.keys[-1] if current.keys else None

    def _locate_first_leaf(self) -> _LeafNode:
        node = self.root
        while not node.is_leaf:
            node = node.children[0]  # type: ignore[index]
        return node  # type: ignore[return-value]

    def _log(self, message: str) -> None:
        self.operation_log.append(message)

    def _rebuild_leaf_links(self) -> None:
        leaves: List[_LeafNode] = []

        def collect(node: _BPlusTreeNode) -> None:
            if node.is_leaf:
                leaves.append(node)  # type: ignore[arg-type]
                return
            internal = node  # type: ignore[assignment]
            for child in internal.children:
                collect(child)

        collect(self.root)
        if not leaves:
            self._first_leaf = self.root  # type: ignore[assignment]
            return
        for index, leaf in enumerate(leaves):
            next_leaf = leaves[index + 1] if index + 1 < len(leaves) else None
            leaf.next_leaf = next_leaf
        self._first_leaf = leaves[0]

    def _node_to_state(self, node: _BPlusTreeNode) -> Dict[str, object]:
        if node.is_leaf:
            return node.to_state()  # type: ignore[return-value]
        return node.to_state()  # type: ignore[return-value]

    @staticmethod
    def _node_from_state(state: Dict[str, object]) -> _BPlusTreeNode:
        node_type = state.get("type")
        if node_type == "leaf":
            return _LeafNode.from_state(state)
        if node_type == "internal":
            return _InternalNode.from_state(state)
        raise ValueError(f"Unknown node type: {node_type}")
