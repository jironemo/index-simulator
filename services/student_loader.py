from __future__ import annotations

import csv
import random
from dataclasses import replace
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

from models import Student


def load_students(
    csv_path: str | Path,
    *,
    limit: Optional[int] = None,
    start_id: int = 1,
    id_step: int = 1,
    randomize_ids: bool = False,
    rng: Optional[random.Random] = None,
) -> List[Student]:
    """Load student records from the canonical CSV dataset."""
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    students: List[Student] = []
    next_id = start_id
    used_ids: set[int] = set()
    if randomize_ids and rng is None:
        rng = random.Random()
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if randomize_ids and rng is not None:
                id_value = _generate_unique_id(rng, used_ids)
            else:
                id_value = next_id
                next_id += id_step
            students.append(Student.from_csv_row(row, id_value=id_value))
            if limit is not None and len(students) >= limit:
                break
    return students


def build_operations_from_students(
    students: Iterable[Student],
    *,
    include_updates: bool = False,
    include_deletes: bool = False,
) -> List[tuple[str, object]]:
    """Generate a basic workload from students iterable."""
    operations: List[tuple[str, object]] = []
    for student in students:
        operations.append(("insert", student))
        if include_updates:
            updated = replace(
                student,
                math_score=student.math_score + 1,
                reading_score=student.reading_score + 1,
                writing_score=student.writing_score + 1,
            )
            operations.append(("update", updated))
        if include_deletes:
            operations.append(("delete", student.id))
    return operations


def build_random_operations(
    students: Sequence[Student],
    *,
    update_count: int = 0,
    delete_count: int = 0,
    search_count: int = 0,
    rng: Optional[random.Random] = None,
) -> List[tuple[str, object]]:
    """Generate random operations targeting records within ``students``."""
    if not students:
        return []
    if rng is None:
        rng = random.Random()

    operations: List[tuple[str, object]] = []

    if update_count > 0:
        chosen = _choose_students(rng, students, update_count)
        for student in chosen:
            operations.append(("update", _with_adjusted_scores(rng, student)))

    if delete_count > 0:
        chosen = _choose_students(rng, students, delete_count)
        for student in chosen:
            operations.append(("delete", student.id))

    if search_count > 0:
        chosen = _choose_students(rng, students, search_count)
        for student in chosen:
            operations.append(("search", student.id))

    if operations:
        rng.shuffle(operations)

    return operations


def _generate_unique_id(rng: random.Random, used: set[int], lower: int = 10_000, upper: int = 9_999_999) -> int:
    while True:
        candidate = rng.randint(lower, upper)
        if candidate not in used:
            used.add(candidate)
            return candidate


def _choose_students(rng: random.Random, population: Sequence[Student], count: int) -> List[Student]:
    if count <= 0:
        return []
    if count <= len(population):
        # Prefer unique selections when possible to provide more coverage
        return list(rng.sample(population, count))
    return [rng.choice(population) for _ in range(count)]


def _with_adjusted_scores(rng: random.Random, student: Student) -> Student:
    delta = rng.randint(-3, 3)
    return replace(
        student,
        math_score=_clamp_score(student.math_score + delta),
        reading_score=_clamp_score(student.reading_score + delta),
        writing_score=_clamp_score(student.writing_score + delta),
    )


def _clamp_score(value: int) -> int:
    return max(0, min(100, value))
