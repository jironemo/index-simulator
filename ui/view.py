from __future__ import annotations

from collections import deque
from dataclasses import asdict, replace
import random
import tkinter as tk
from tkinter import messagebox, ttk
from typing import List, Optional, Union

from config import DATA_CSV_PATH, UI_HEIGHT, UI_REFRESH_INTERVAL_MS, UI_WIDTH
from models import Student
from services.simulation_controller import SimulationController
from services.student_loader import build_random_operations, load_students


GENDER_COLORS = {
    "female": "#ec4899",
    "male": "#38bdf8",
}


class LSMStudioView:
    """Tkinter front-end that polls SimulationController snapshots."""

    def __init__(
        self,
        controller: SimulationController,
        *,
        refresh_interval_ms: int = UI_REFRESH_INTERVAL_MS,
        width: int = UI_WIDTH,
        height: int = UI_HEIGHT,
        title: str = "LSM Studio (New Version)",
    ) -> None:
        self.controller = controller
        self.refresh_interval_ms = refresh_interval_ms
        self.root = tk.Tk()
        self.root.title(title)
        self.root.geometry(f"{width}x{height}")
        self._auto_step = tk.BooleanVar(value=True)
        self._status_var = tk.StringVar(value="Queue: 0 operations")
        self._backend_var = tk.StringVar(value=self._format_backend_status(0, 0))
        self.dataset_size_var = tk.StringVar(value="64")
        self.update_count_var = tk.StringVar(value="0")
        self.delete_count_var = tk.StringVar(value="0")
        self.search_count_var = tk.StringVar(value="0")
        self._suite_progress_var = tk.DoubleVar(value=0.0)
        self._suite_status_var = tk.StringVar(value="Simulation suite idle")
        self._suite_running = False
        self._suite_button: Optional[ttk.Button] = None
        self._suite_progress: Optional[ttk.Progressbar] = None
        self._suite_queue = deque()
        self._suite_results: list[str] = []
        self._suite_total_steps = 0
        self._suite_steps_completed = 0
        self._suite_previous_auto = self._auto_step.get()
        self._current_scenario_info: Optional[dict[str, int]] = None
        self._suite_total_scenarios = 0
        self._memtable_data: list[dict[str, object]] = []
        self._levels_data: list[list[list[dict[str, object]]]] = []
        self._levels_counts: tuple[int, int] = (0, 0)
        self._bptree_snapshot: dict[str, object] = {}
        self._lsm_canvas_dialogs: dict[str, tuple[str, str]] = {}
        self._build_layout()
        self.root.after(self.refresh_interval_ms, self._on_tick)

    # ------------------------------------------------------------------
    # Layout construction
    # ------------------------------------------------------------------
    def _build_layout(self) -> None:
        main = ttk.Frame(self.root, padding=12)
        main.pack(fill=tk.BOTH, expand=True)

        controls = ttk.Labelframe(main, text="Operations")
        controls.pack(fill=tk.X, pady=(0, 12))

        self.id_entry = self._make_labeled_entry(controls, "ID", 0)
        self.gender_entry = self._make_labeled_entry(controls, "Gender", 1)
        self.race_entry = self._make_labeled_entry(controls, "Race/Ethnicity", 2)
        self.math_entry = self._make_labeled_entry(controls, "Math", 3)
        self.reading_entry = self._make_labeled_entry(controls, "Reading", 4)
        self.writing_entry = self._make_labeled_entry(controls, "Writing", 5)

        button_frame = ttk.Frame(controls)
        button_frame.grid(row=0, column=6, rowspan=3, padx=(12, 0))

        ttk.Button(button_frame, text="Insert", command=self._on_insert).grid(row=0, column=0, sticky=tk.EW, pady=2)
        ttk.Button(button_frame, text="Update", command=self._on_update).grid(row=1, column=0, sticky=tk.EW, pady=2)
        ttk.Button(button_frame, text="Delete", command=self._on_delete).grid(row=2, column=0, sticky=tk.EW, pady=2)
        ttk.Button(button_frame, text="Search", command=self._on_search).grid(row=3, column=0, sticky=tk.EW, pady=2)
        ttk.Button(button_frame, text="Force Flush", command=self._on_force_flush).grid(row=4, column=0, sticky=tk.EW, pady=(8, 2))
        ttk.Button(button_frame, text="Force Compaction", command=self._on_force_compact).grid(row=5, column=0, sticky=tk.EW)

        queue_frame = ttk.Frame(controls)
        queue_frame.grid(row=3, column=0, columnspan=7, sticky=tk.W, pady=(8, 0))
        ttk.Label(queue_frame, text="Dataset size").grid(row=0, column=0, padx=(0, 4))
        ttk.Entry(queue_frame, textvariable=self.dataset_size_var, width=8).grid(row=0, column=1, padx=(0, 12))
        ttk.Button(queue_frame, text="Process Next", command=self._process_next).grid(row=0, column=2, padx=(0, 8))
        ttk.Button(queue_frame, text="Process All", command=self._process_all).grid(row=0, column=3, padx=(0, 8))
        ttk.Button(queue_frame, text="Load Dataset", command=self._load_dataset).grid(row=0, column=4, padx=(0, 8))
        ttk.Button(queue_frame, text="Reset", command=self._reset).grid(row=0, column=5, padx=(0, 8))
        ttk.Checkbutton(queue_frame, text="Auto step", variable=self._auto_step).grid(row=0, column=6, padx=(0, 8))

        ttk.Label(queue_frame, textvariable=self._status_var).grid(row=0, column=7, padx=(12, 0))
        ttk.Label(queue_frame, textvariable=self._backend_var).grid(row=0, column=8, padx=(12, 0), sticky=tk.W)

        ttk.Label(queue_frame, text="# Updates").grid(row=1, column=0, padx=(0, 4), pady=(6, 0), sticky=tk.W)
        ttk.Entry(queue_frame, textvariable=self.update_count_var, width=8).grid(row=1, column=1, padx=(0, 12), pady=(6, 0))
        ttk.Label(queue_frame, text="# Deletes").grid(row=1, column=2, padx=(0, 4), pady=(6, 0), sticky=tk.W)
        ttk.Entry(queue_frame, textvariable=self.delete_count_var, width=8).grid(row=1, column=3, padx=(0, 12), pady=(6, 0))
        ttk.Label(queue_frame, text="# Searches").grid(row=1, column=4, padx=(0, 4), pady=(6, 0), sticky=tk.W)
        ttk.Entry(queue_frame, textvariable=self.search_count_var, width=8).grid(row=1, column=5, padx=(0, 12), pady=(6, 0))

        suite_frame = ttk.Frame(controls)
        suite_frame.grid(row=4, column=0, columnspan=7, sticky=tk.EW, pady=(8, 0))
        suite_frame.columnconfigure(1, weight=1)
        self._suite_button = ttk.Button(suite_frame, text="Run Simulation Suite", command=self._run_simulation_suite)
        self._suite_button.grid(row=0, column=0, sticky=tk.W)
        self._suite_progress = ttk.Progressbar(suite_frame, variable=self._suite_progress_var, maximum=4, mode="determinate")
        self._suite_progress.grid(row=0, column=1, sticky=tk.EW, padx=(12, 12))
        ttk.Label(suite_frame, textvariable=self._suite_status_var).grid(row=0, column=2, sticky=tk.W)

        quick_frame = ttk.Frame(suite_frame)
        quick_frame.grid(row=1, column=0, columnspan=3, sticky=tk.W, pady=(8, 0))
        for idx, count in enumerate((100, 1000, 5000, 10000)):
            ttk.Button(
                quick_frame,
                text=f"Run {count}",
                command=lambda c=count: self._run_simulation_suite([c]),
            ).grid(row=0, column=idx, padx=(0 if idx == 0 else 6, 0))

        # Split main area into notebook tabs for visual sections
        notebook = ttk.Notebook(main)
        notebook.pack(fill=tk.BOTH, expand=True)

        lsm_frame = ttk.Frame(notebook, padding=8)
        self.lsm_canvas = tk.Canvas(lsm_frame, height=360, background="#020617")
        self.lsm_canvas.pack(fill=tk.BOTH, expand=True)
        notebook.add(lsm_frame, text="LSM Tree")

        bptree_frame = ttk.Frame(notebook, padding=8)
        self.bptree_canvas = tk.Canvas(bptree_frame, height=320, background="#030712")
        self.bptree_canvas.pack(fill=tk.BOTH, expand=True)
        notebook.add(bptree_frame, text="B+ Tree")

        metrics_frame = ttk.Frame(notebook, padding=8)
        self.metrics_tree = self._make_table_view(
            metrics_frame,
            columns=("structure", "operation", "count", "memory_ms", "memory_ns"),
            headings={
                "structure": "Structure",
                "operation": "Operation",
                "count": "Count",
                "memory_ms": "In-Memory (ms)",
                "memory_ns": "In-Memory (ns)",
            },
        )
        notebook.add(metrics_frame, text="Benchmarks")

        ops_frame = ttk.Frame(notebook, padding=8)
        self.operations_text = tk.Text(ops_frame, height=12, wrap=tk.WORD, state=tk.DISABLED, font=("Consolas", 10))
        self.operations_text.pack(fill=tk.BOTH, expand=True)
        notebook.add(ops_frame, text="Operations Log")

        log_frame = ttk.Frame(notebook, padding=8)
        self.log_text = tk.Text(log_frame, height=12, wrap=tk.WORD, state=tk.DISABLED, font=("Consolas", 10))
        self.log_text.pack(fill=tk.BOTH, expand=True)
        notebook.add(log_frame, text="B+ Tree Log")

    def _make_labeled_entry(self, parent: ttk.Frame, label: str, column: int) -> ttk.Entry:
        ttk.Label(parent, text=label).grid(row=column % 3, column=column // 3 * 2, sticky=tk.W, padx=(0, 4), pady=2)
        entry = ttk.Entry(parent, width=16)
        entry.grid(row=column % 3, column=column // 3 * 2 + 1, sticky=tk.W, pady=2)
        return entry

    def _make_table_view(self, parent: ttk.Frame, *, columns: tuple[str, ...], headings: dict[str, str]) -> ttk.Treeview:
        tree = ttk.Treeview(parent, columns=columns, show="headings", height=12)
        for column in columns:
            tree.heading(column, text=headings.get(column, column.capitalize()))
            width = 160 if column == "memory_ns" else 120
            tree.column(column, anchor=tk.CENTER, width=width)
        tree.pack(fill=tk.BOTH, expand=True)
        return tree

    # ------------------------------------------------------------------
    # Button handlers
    # ------------------------------------------------------------------
    def _on_insert(self) -> None:
        student = self._read_student_from_form()
        if student:
            self._execute_operation_dialog("insert", student, "Insert Operation")

    def _on_update(self) -> None:
        student = self._read_student_from_form()
        if student:
            self._execute_operation_dialog("update", student, "Update Operation")

    def _on_delete(self) -> None:
        student_id = self._read_student_id()
        if student_id is not None:
            self._execute_operation_dialog("delete", student_id, "Delete Operation")

    def _on_search(self) -> None:
        student_id = self._read_student_id()
        if student_id is not None:
            self._execute_operation_dialog("search", student_id, "Search Operation")

    def _on_force_flush(self) -> None:
        result = self.controller.force_flush()
        flush_metrics = result.get("flush_metrics", {}) or {}
        summary = self._build_flush_compact_summary("Memtable Flush", flush_metrics, result.get("lsm", {}))
        self._refresh_snapshot()
        self._show_copyable_dialog("Forced Memtable Flush", summary)

    def _on_force_compact(self) -> None:
        result = self.controller.force_compact()
        compact_metrics = result.get("compaction_metrics", {}) or {}
        summary = self._build_flush_compact_summary("Level Compaction", compact_metrics, result.get("lsm", {}))
        self._refresh_snapshot()
        self._show_copyable_dialog("Forced Compaction", summary)

    def _run_simulation_suite(self, scenario_counts: Optional[List[int]] = None) -> None:
        if self._suite_running:
            return
        self._suite_running = True
        self._suite_previous_auto = self._auto_step.get()
        self._auto_step.set(False)
        if self._suite_button is not None:
            self._suite_button.state(["disabled"])
        self._suite_progress_var.set(0.0)
        counts = scenario_counts or [100, 1000, 5000, 10000]
        self._suite_status_var.set("Preparing simulation suite...")
        self._suite_results.clear()
        self._suite_queue.clear()
        self._suite_steps_completed = 0
        try:
            self._prepare_simulation_suite(counts)
        except Exception as exc:  # noqa: BLE001 - surface suite failure to user
            self._show_error(f"Simulation suite failed: {exc}")
            self._finalize_simulation_suite(success=False)
            return

        if self._suite_total_steps <= 0:
            self._show_error("Simulation suite has no operations to run.")
            self._finalize_simulation_suite(success=False)
            return

        if self._suite_progress is not None:
            self._suite_progress.configure(maximum=float(self._suite_total_steps or 1))
        self._suite_status_var.set("Running simulation suite...")
        self.root.after(10, self._process_suite_queue)

    def _prepare_simulation_suite(self, scenarios: List[int]) -> None:
        queue = deque()
        total_steps = 0
        total_scenarios = len(scenarios)

        for index, count in enumerate(scenarios, start=1):
            rng = random.Random(count)
            students = load_students(
                DATA_CSV_PATH,
                limit=count,
                randomize_ids=True,
                rng=rng,
            )
            if len(students) < count:
                raise ValueError(f"Dataset provided only {len(students)} records for a {count} run.")

            half = max(1, count // 2)
            insert_ops = [("insert", student) for student in students]
            search_targets = self._choose_students(rng, students, half)
            search_ops = [("search", student.id) for student in search_targets]
            update_targets = self._choose_students(rng, students, half)
            update_ops = [("update", self._adjust_student_scores(student, rng)) for student in update_targets]
            delete_targets = self._choose_students(rng, students, half)
            delete_ops = [("delete", student.id) for student in delete_targets]

            scenario_total = len(insert_ops) + len(search_ops) + len(update_ops) + len(delete_ops)
            total_steps += scenario_total

            queue.append(("scenario_start", {
                "index": index,
                "count": count,
                "total_ops": scenario_total,
                "total_scenarios": total_scenarios,
            }))
            queue.append(("reset", None))
            for op in insert_ops:
                queue.append(("exec", op))
            for op in search_ops:
                queue.append(("exec", op))
            for op in update_ops:
                queue.append(("exec", op))
            for op in delete_ops:
                queue.append(("exec", op))
            queue.append(("collect", {"insert_count": count}))

        self._suite_queue = queue
        self._suite_total_steps = total_steps
        self._suite_steps_completed = 0
        self._suite_total_scenarios = total_scenarios
        self._current_scenario_info = None

    def _process_suite_queue(self) -> None:
        if not self._suite_running:
            return
        if not self._suite_queue:
            self._finalize_simulation_suite(success=True)
            return

        operations_per_tick = 25
        processed = 0

        try:
            while self._suite_queue and processed < operations_per_tick:
                action, payload = self._suite_queue.popleft()
                if action == "scenario_start":
                    info = dict(payload)
                    info["completed_ops"] = 0
                    self._current_scenario_info = info
                    self._suite_status_var.set(self._format_suite_status())
                elif action == "reset":
                    self.controller.reset()
                    self._refresh_snapshot()
                elif action == "exec":
                    operation, op_payload = payload  # type: ignore[misc]
                    self.controller.execute_now(operation, op_payload)
                    self._suite_steps_completed += 1
                    processed += 1
                    if self._current_scenario_info is not None:
                        completed = self._current_scenario_info.get("completed_ops", 0) + 1
                        self._current_scenario_info["completed_ops"] = completed
                        self._suite_status_var.set(self._format_suite_status())
                elif action == "collect":
                    insert_count = int(payload.get("insert_count", 0))  # type: ignore[union-attr]
                    metrics = self.controller.snapshot()["benchmarks"]
                    summary = self._format_simulation_summary(insert_count, metrics)
                    self._suite_results.append(summary)
                    self.controller.reset()
                    self._refresh_snapshot()
                    self._current_scenario_info = None
                else:
                    # Unknown payloads are ignored but should not occur
                    continue
        except Exception as exc:  # noqa: BLE001 - ensure graceful failure
            self._show_error(f"Simulation suite failed: {exc}")
            self._finalize_simulation_suite(success=False)
            return

        self._suite_progress_var.set(float(self._suite_steps_completed))
        self.root.update_idletasks()

        if self._suite_queue:
            self.root.after(20, self._process_suite_queue)
        else:
            self._finalize_simulation_suite(success=True)

    def _finalize_simulation_suite(self, *, success: bool) -> None:
        if not self._suite_running and success:
            return
        self._suite_running = False
        self._auto_step.set(self._suite_previous_auto)
        if self._suite_button is not None:
            self._suite_button.state(["!disabled"])
        status = "Simulation suite complete" if success else "Simulation suite failed"
        self._suite_status_var.set(status)
        if success and self._suite_results:
            summary_text = "\n\n".join(self._suite_results)
            self._show_copyable_dialog("Simulation Suite Results", summary_text)
        self._suite_progress_var.set(0.0)
        self._suite_total_steps = 0
        self._suite_steps_completed = 0
        self._suite_queue.clear()
        self._suite_results.clear()
        self._current_scenario_info = None
        self._suite_total_scenarios = 0
        self.controller.reset()
        self._refresh_snapshot()

    def _format_suite_status(self) -> str:
        info = self._current_scenario_info
        if not info:
            return "Running simulation suite..."
        index = int(info.get("index", 0))
        total_scenarios = int(info.get("total_scenarios", 0))
        count = int(info.get("count", 0))
        total_ops = int(info.get("total_ops", 0))
        completed = int(info.get("completed_ops", 0))
        return f"Scenario {index}/{total_scenarios} - {count} inserts ({completed}/{max(1, total_ops)} ops)"

    def _choose_students(
        self,
        rng: random.Random,
        population: list[Student],
        count: int,
    ) -> list[Student]:
        if count <= 0 or not population:
            return []
        if count >= len(population):
            return list(population)
        return rng.sample(population, count)

    def _adjust_student_scores(self, student: Student, rng: random.Random) -> Student:
        delta = rng.randint(-3, 3)
        return replace(
            student,
            math_score=self._clamp_score(student.math_score + delta),
            reading_score=self._clamp_score(student.reading_score + delta),
            writing_score=self._clamp_score(student.writing_score + delta),
        )

    @staticmethod
    def _clamp_score(value: int) -> int:
        return max(0, min(100, value))

    def _format_simulation_summary(
        self,
        insert_count: int,
        metrics: dict[str, dict[str, dict[str, float]]],
    ) -> str:
        half = max(1, insert_count // 2)
        lines = [
            f"Scenario - {insert_count} inserts, {half} updates/searches/deletes",
        ]
        for structure in ("lsm", "bptree"):
            structure_metrics = metrics.get(structure, {})
            lines.append(f"{structure.upper()}:")
            structure_total_ns = 0.0
            for operation in ("insert", "update", "search", "delete"):
                op_metrics = structure_metrics.get(
                    operation,
                    {
                        "count": 0,
                        "in_memory_ns": 0.0,
                        "file_io_ns": 0.0,
                        "total_in_memory_ns": 0.0,
                        "total_file_io_ns": 0.0,
                    },
                )
                count = int(op_metrics.get("count", 0))
                in_mem = op_metrics.get("in_memory_ns", 0.0)
                file_io = op_metrics.get("file_io_ns", 0.0)
                memory_bytes = op_metrics.get("memory_bytes", 0.0)
                disk_bytes = op_metrics.get("disk_bytes", 0.0)
                total_in_mem = op_metrics.get("total_in_memory_ns")
                if total_in_mem is None:
                    total_in_mem = in_mem * count
                total_file = op_metrics.get("total_file_io_ns")
                if total_file is None:
                    total_file = file_io * count
                structure_total_ns += float(total_in_mem) + float(total_file)
                entry = (
                    f"  {operation.title():7} count={count:<5} "
                    f"avg_mem={self._format_duration(in_mem)}"
                )
                if file_io:
                    entry += f" avg_io={self._format_duration(file_io)}"
                entry += f" mem_use={self._format_bytes(memory_bytes)}"
                if disk_bytes:
                    entry += f" disk_use={self._format_bytes(disk_bytes)}"
                lines.append(entry)
            if structure == "lsm":
                lines.append(
                    "  Total Time (ops + flush/compaction): "
                    f"{self._format_duration(structure_total_ns)}"
                )
            else:
                lines.append(f"  Total Time: {self._format_duration(structure_total_ns)}")
        return "\n".join(lines)

    @staticmethod
    def _format_duration(value: Union[int, float]) -> str:
        ns = float(value)
        if ns >= 1_000_000_000:
            return f"{ns / 1_000_000_000:.3f} s"
        if ns >= 1_000_000:
            return f"{ns / 1_000_000:.3f} ms"
        if ns >= 1_000:
            return f"{ns / 1_000:.3f} us"
        if ns <= 0:
            return "0 ns"
        return f"{ns:.0f} ns"

    @staticmethod
    def _format_bytes(value: Union[int, float]) -> str:
        size = float(value)
        if size <= 0:
            return "0 B"
        units = ["B", "KB", "MB", "GB", "TB"]
        index = 0
        while size >= 1024 and index < len(units) - 1:
            size /= 1024
            index += 1
        if index == 0:
            return f"{int(size)} {units[index]}"
        return f"{size:.2f} {units[index]}"

    def _execute_operation_dialog(self, operation: str, payload: object, dialog_title: str) -> None:
        try:
            result = self.controller.execute_now(operation, payload)
        except Exception as exc:  # noqa: BLE001 - surface error to user
            self._show_error(f"Failed to perform {operation}: {exc}")
            return
        self._append_operation_log(result)
        self._refresh_snapshot()
        dialog_body = self._build_operation_dialog_text(operation, payload, result)
        self._show_copyable_dialog(dialog_title, dialog_body)

    def _process_next(self) -> None:
        result = self.controller.step()
        if result:
            self._append_operation_log(result)
        self._refresh_snapshot()

    def _process_all(self) -> None:
        results = self.controller.drain_all()
        for result in results:
            self._append_operation_log(result)
        self._refresh_snapshot()

    def _load_dataset(self) -> None:
        limit = self._read_non_negative(self.dataset_size_var.get(), "Dataset size", empty_is_none=True)
        if limit is False:
            return
        limit_value = None if limit is None else limit
        if limit_value == 0:
            self._show_error("Dataset size must be greater than zero.")
            return

        update_count = self._read_non_negative(self.update_count_var.get(), "Update count")
        if update_count is False:
            return
        delete_count = self._read_non_negative(self.delete_count_var.get(), "Delete count")
        if delete_count is False:
            return
        search_count = self._read_non_negative(self.search_count_var.get(), "Search count")
        if search_count is False:
            return
        try:
            students = load_students(
                DATA_CSV_PATH,
                limit=limit_value,
                randomize_ids=True,
            )
        except Exception as exc:  # noqa: BLE001 - present error to user
            self._show_error(f"Failed to load dataset: {exc}")
            return
        for student in students:
            self.controller.enqueue("insert", student)
        random_ops = build_random_operations(
            students,
            update_count=update_count or 0,
            delete_count=delete_count or 0,
            search_count=search_count or 0,
        )
        if random_ops:
            self.controller.extend_queue(random_ops)
        self._status_var.set(
            "Queue: {depth} operations (inserted {inserted}, updates {updates}, deletes {deletes}, searches {searches})".format(
                depth=self.controller.queue_depth(),
                inserted=len(students),
                updates=update_count or 0,
                deletes=delete_count or 0,
                searches=search_count or 0,
            )
        )

    def _reset(self) -> None:
        self.controller.reset()
        self._status_var.set("Queue: 0 operations")
        self.dataset_size_var.set("64")
        self.update_count_var.set("0")
        self.delete_count_var.set("0")
        self.search_count_var.set("0")
        self._clear_canvas(self.lsm_canvas)
        self._clear_canvas(self.bptree_canvas)
        self._clear_tree(self.metrics_tree)
        self._set_text(self.operations_text, "")
        self._set_text(self.log_text, "")

    # ------------------------------------------------------------------
    # Form helpers
    # ------------------------------------------------------------------
    def _read_student_id(self) -> Optional[int]:
        raw = self.id_entry.get().strip()
        if not raw:
            self._show_error("ID is required for this action.")
            return None
        try:
            return int(raw)
        except ValueError:
            self._show_error("ID must be an integer.")
            return None

    def _read_student_from_form(self) -> Optional[Student]:
        student_id = self._read_student_id()
        if student_id is None:
            return None
        try:
            math = int(self.math_entry.get() or 0)
            reading = int(self.reading_entry.get() or 0)
            writing = int(self.writing_entry.get() or 0)
        except ValueError:
            self._show_error("Scores must be integers.")
            return None
        return Student(
            id=student_id,
            gender=self.gender_entry.get().strip(),
            race_ethnicity=self.race_entry.get().strip(),
            math_score=math,
            reading_score=reading,
            writing_score=writing,
        )

    # ------------------------------------------------------------------
    # Refresh loop
    # ------------------------------------------------------------------
    def _on_tick(self) -> None:
        if self._auto_step.get():
            result = self.controller.step()
            if result:
                self._append_operation_log(result)
        self._refresh_snapshot()
        self.root.after(self.refresh_interval_ms, self._on_tick)

    def _refresh_snapshot(self) -> None:
        snapshot = self.controller.snapshot()
        self._status_var.set(f"Queue: {snapshot['queue_depth']} operations")
        memory_bytes = int(snapshot.get("memory_bytes", 0) or 0)
        disk_bytes = int(snapshot.get("disk_bytes", 0) or 0)
        self._backend_var.set(self._format_backend_status(memory_bytes, disk_bytes))
        self._memtable_data = snapshot["lsm"]["memtable"]
        self._levels_data = snapshot["lsm"]["levels"]
        self._levels_counts = (
            snapshot["lsm"].get("flush_count", 0),
            snapshot["lsm"].get("compaction_count", 0),
        )
        self._bptree_snapshot = snapshot["bptree"]
        self._render_lsm()
        self._render_bptree()
        self._refresh_metrics(snapshot["benchmarks"])
        log = "\n".join(self._bptree_snapshot.get("operation_log", []))
        self._set_text(self.log_text, log)

    # ------------------------------------------------------------------
    # Canvas renderers
    # ------------------------------------------------------------------
    def _render_lsm(self) -> None:
        canvas = self.lsm_canvas
        self._clear_canvas(canvas)
        self._lsm_canvas_dialogs.clear()
        width = canvas.winfo_width() or canvas.winfo_reqwidth() or 800
        margin = 20
        y = margin

        # Draw memtable buckets
        memtable_tag = "memtable-dialog"
        canvas.create_text(
            margin,
            y,
            anchor=tk.NW,
            fill="#93c5fd",
            text=f"Memtable ({len(self._memtable_data)} records)",
            font=("Segoe UI", 12, "bold"),
            tags=(memtable_tag,),
        )
        y += 28
        rect_width = 70
        rect_height = 30
        spacing = 10
        x = margin

        for idx, record in enumerate(self._memtable_data[:60]):
            if x + rect_width > width - margin:
                x = margin
                y += rect_height + spacing
            color = GENDER_COLORS.get(str(record.get("gender", "")).lower(), "#22d3ee")
            canvas.create_rectangle(
                x,
                y,
                x + rect_width,
                y + rect_height,
                fill=color,
                outline="",
                tags=(memtable_tag, f"memtable-record-{idx}"),
            )
            canvas.create_text(
                x + rect_width / 2,
                y + rect_height / 2,
                text=str(record.get("id", "?")),
                fill="#0f172a",
                font=("Segoe UI", 10, "bold"),
                tags=(memtable_tag,),
            )
            x += rect_width + spacing

        if self._memtable_data:
            y += rect_height + spacing
            if len(self._memtable_data) > 60:
                canvas.create_text(
                    margin,
                    y,
                    anchor=tk.NW,
                    fill="#94a3b8",
                    text=f"… {len(self._memtable_data) - 60} more records",
                    font=("Segoe UI", 9),
                )
                y += spacing + 12
            else:
                y += spacing
        else:
            y += spacing

        memtable_area_bottom = y
        canvas.create_rectangle(
            margin - 10,
            margin - 6,
            width - margin + 10,
            memtable_area_bottom + 6,
            outline="",
            fill="",
            tags=(memtable_tag, "memtable-hitbox"),
        )
        canvas.tag_lower("memtable-hitbox")
        self._register_lsm_dialog(
            memtable_tag,
            "Memtable Records",
            self._format_student_records(self._memtable_data),
        )

        # Draw SSTable levels
        flushes, compactions = self._levels_counts
        canvas.create_text(
            margin,
            y,
            anchor=tk.NW,
            fill="#cbd5f5",
            text=f"LSM Tree Levels  •  Flushes: {flushes}  •  Compactions: {compactions}",
            font=("Segoe UI", 12, "bold"),
        )
        y += 32

        level_gap = 84
        table_gap = 12
        table_height = 38

        for level_index, level in enumerate(self._levels_data):
            canvas.create_text(
                margin,
                y,
                anchor=tk.W,
                fill="#67e8f9",
                text=f"Level {level_index}",
                font=("Segoe UI", 11, "bold"),
            )
            x_pos = margin + 110
            if not level:
                canvas.create_text(
                    x_pos,
                    y,
                    anchor=tk.W,
                    fill="#e2e8f0",
                    text="<empty>",
                    font=("Segoe UI", 10),
                )
                y += level_gap
                continue

            for table_idx, table in enumerate(level[:8]):
                table_tag = f"sstable-{level_index}-{table_idx}"
                table_width = max(90, min(240, len(table) * 14))
                if x_pos + table_width > width - margin:
                    x_pos = margin + 110
                    y += table_height + 18
                canvas.create_rectangle(
                    x_pos,
                    y - table_height / 2,
                    x_pos + table_width,
                    y + table_height / 2,
                    outline="#38bdf8",
                    width=2,
                    fill="#082f49",
                    tags=(table_tag,),
                )
                canvas.create_text(
                    x_pos + 6,
                    y,
                    anchor=tk.W,
                    fill="#e0f2fe",
                    text=f"{len(table)} records",
                    font=("Segoe UI", 10, "bold"),
                    tags=(table_tag,),
                )
                if table:
                    preview = ", ".join(f"{row.get('id')}" for row in table[:4])
                    canvas.create_text(
                        x_pos + table_width - 6,
                        y,
                        anchor=tk.E,
                        fill="#bae6fd",
                        text=preview + (" …" if len(table) > 4 else ""),
                        font=("Segoe UI", 9),
                        tags=(table_tag,),
                    )
                x_pos += table_width + table_gap
                self._register_lsm_dialog(
                    table_tag,
                    f"SSTable Level {level_index} • Table {table_idx + 1}",
                    self._format_student_records(table),
                )
            if len(level) > 8:
                canvas.create_text(
                    margin + 110,
                    y + table_height / 2 + 12,
                    anchor=tk.W,
                    fill="#94a3b8",
                    text=f"… {len(level) - 8} more tables",
                    font=("Segoe UI", 9),
                )

            y += level_gap

        bbox = canvas.bbox("all")
        if bbox:
            canvas.configure(scrollregion=bbox)

    def _render_bptree(self) -> None:
        canvas = self.bptree_canvas
        self._clear_canvas(canvas)
        width = canvas.winfo_width() or canvas.winfo_reqwidth() or 800
        margin = 20
        level_gap = 100
        node_width = 140
        node_height = 44

        order_raw = self._bptree_snapshot.get("order", 4)
        try:
            order = max(3, int(order_raw))
        except (TypeError, ValueError):
            order = 4
        record_count = self._bptree_snapshot.get("record_count", 0)
        leaves = self._bptree_snapshot.get("leaves", [])

        canvas.create_text(
            margin,
            margin - 10,
            anchor=tk.NW,
            fill="#c084fc",
            text=f"B+ Tree (order {order}, {record_count} records)",
            font=("Segoe UI", 11, "bold"),
        )

        levels = self._build_bptree_levels(order, leaves)
        if not levels:
            canvas.create_text(
                width / 2,
                margin + 40,
                text="<tree is empty>",
                fill="#e9d5ff",
                font=("Segoe UI", 11, "italic"),
            )
            return

        # Assign x positions starting from leaves and bubble up
        leaf_level = levels[-1]
        leaf_count = len(leaf_level)
        if leaf_count <= 1:
            spacing = 0
        else:
            spacing = (width - 2 * margin - node_width) / max(1, leaf_count - 1)
            spacing = max(40, spacing)
        start_x = margin + node_width / 2
        for idx, node in enumerate(leaf_level):
            node["cx"] = start_x + idx * spacing

        for level_index in range(len(levels) - 2, -1, -1):
            for node in levels[level_index]:
                children = node.get("children", [])
                if children:
                    node["cx"] = sum(child["cx"] for child in children) / len(children)
                else:
                    node["cx"] = width / 2

        # Connector lines first so nodes sit on top
        for level_index, level in enumerate(levels[:-1]):
            parent_y = margin + 40 + level_index * level_gap
            child_y = margin + 40 + (level_index + 1) * level_gap
            for node in level:
                x_center = node["cx"]
                for child in node.get("children", []):
                    canvas.create_line(
                        x_center,
                        parent_y + node_height / 2,
                        child["cx"],
                        child_y - node_height / 2,
                        fill="#10b981",
                    )

        # Draw nodes top-down
        for level_index, level in enumerate(levels):
            y_center = margin + 40 + level_index * level_gap
            is_leaf_level = level_index == len(levels) - 1
            label_text = (
                "Root"
                if level_index == 0
                else ("Leaves" if is_leaf_level else f"Level {level_index}")
            )
            canvas.create_text(
                margin,
                y_center - node_height / 2 - 14,
                anchor=tk.W,
                fill="#93c5fd",
                text=label_text,
                font=("Segoe UI", 10, "bold"),
            )
            for node in level:
                x_center = node["cx"]
                x0 = x_center - node_width / 2
                x1 = x_center + node_width / 2
                y0 = y_center - node_height / 2
                y1 = y_center + node_height / 2
                fill_color = "#3b0764" if level_index == 0 else ("#1e293b" if not is_leaf_level else "#042f2e")
                outline_color = "#a855f7" if level_index == 0 else ("#38bdf8" if not is_leaf_level else "#34d399")
                text_color = "#f3e8ff" if level_index == 0 else ("#cbd5f5" if not is_leaf_level else "#bbf7d0")
                canvas.create_rectangle(x0, y0, x1, y1, outline=outline_color, width=2, fill=fill_color)
                keys = node.get("keys", [])
                key_text = ", ".join(str(key) for key in keys) if keys else ("<leaf>" if is_leaf_level and len(levels) == 1 else "")
                if is_leaf_level and node.get("leaf_keys"):
                    key_text = node["leaf_keys"]
                display_text = key_text or "<empty>"
                canvas.create_text(
                    x_center,
                    y_center,
                    text=display_text,
                    fill=text_color,
                    font=("Segoe UI", 9 if len(display_text) < 18 else 8),
                )

        bbox = canvas.bbox("all")
        if bbox:
            canvas.configure(scrollregion=bbox)

    def _build_bptree_levels(self, order: int, leaves: list[list[dict[str, object]]]) -> list[list[dict[str, object]]]:
        if not leaves:
            return []

        leaf_nodes: list[dict[str, object]] = []
        for index, leaf in enumerate(leaves):
            keys = [item.get("id") for item in leaf]
            preview = ", ".join(str(item.get("id", "?")) for item in leaf[:6])
            if len(leaf) > 6:
                preview += " …"
            leaf_nodes.append({"keys": keys, "leaf_keys": preview, "children": []})

        levels: list[list[dict[str, object]]] = [leaf_nodes]
        current = leaf_nodes
        max_children = max(2, order)
        while len(current) > 1:
            parents: list[dict[str, object]] = []
            for i in range(0, len(current), max_children):
                group = current[i : i + max_children]
                separator_keys = []
                for child in group[1:]:
                    keys = child.get("keys", [])
                    if keys:
                        separator_keys.append(keys[0])
                parents.append({"keys": separator_keys, "children": group})
            levels.append(parents)
            current = parents

        return list(reversed(levels))

    def _read_non_negative(self, raw_value: str, field_name: str, *, empty_is_none: bool = False):
        value_str = raw_value.strip()
        if not value_str:
            return None if empty_is_none else 0
        try:
            value = int(value_str)
        except ValueError:
            self._show_error(f"{field_name} must be an integer.")
            return False
        if value < 0:
            self._show_error(f"{field_name} must be non-negative.")
            return False
        return value

    def _refresh_metrics(self, metrics: dict[str, dict[str, dict[str, Union[int, float]]]]) -> None:
        self._clear_tree(self.metrics_tree)
        for structure, per_op in metrics.items():
            for operation, values in per_op.items():
                count = int(values.get("count", 0))
                memory_ms = values.get("in_memory_ns", 0.0) / 1_000_000
                memory_ns = values.get("in_memory_ns", 0.0)
                self.metrics_tree.insert(
                    "",
                    tk.END,
                    values=(structure, operation, count, f"{memory_ms:.3f}", f"{memory_ns:.0f}"),
                )

    # ------------------------------------------------------------------
    # Dialog helpers
    # ------------------------------------------------------------------
    def _build_operation_dialog_text(
        self,
        operation: str,
        payload: object,
        result: dict[str, object],
    ) -> str:
        lines: list[str] = []
        operation_name = operation.upper()
        lines.append(f"Operation: {operation_name}")
        student_id = result.get("student_id")
        if student_id is not None:
            lines.append(f"Student ID: {student_id}")

        lines.append("")
        lines.append("Input Data:")
        if isinstance(payload, Student):
            for key, value in asdict(payload).items():
                lines.append(f"  {key}: {value}")
        else:
            lines.append(f"  student_id: {payload}")

        lines.append("")
        lines.append("Steps:")
        for step in self._operation_steps(operation, result):
            lines.append(f"  - {step}")

        lsm_metrics = result.get("lsm_metrics", {}) or {}
        bpt_metrics = result.get("bptree_metrics", {}) or {}

        lines.append("")
        lines.append("Timings (nanoseconds):")
        lines.extend(self._format_metric_lines("LSM Tree", lsm_metrics))
        lines.extend(self._format_metric_lines("B+ Tree", bpt_metrics))

        if operation == "search":
            lines.append("")
            lines.append("Results:")
            lsm_result = result.get("lsm_result") or "not found"
            bpt_result = result.get("bptree_result") or "not found"
            lines.append(f"  LSM Tree: {lsm_result}")
            lines.append(f"  B+ Tree: {bpt_result}")

        queue_depth = result.get("queued_operations")
        if queue_depth is not None:
            lines.append("")
            lines.append(f"Queue depth after operation: {queue_depth}")

        return "\n".join(lines)

    def _build_flush_compact_summary(
        self,
        action_label: str,
        metrics: dict[str, object],
        lsm_snapshot: dict[str, object],
    ) -> str:
        lines = [f"Action: {action_label}"]
        duration_ns = int(metrics.get("file_io_ns", 0) or 0)
        lines.append(f"Duration: {duration_ns:,} ns ({duration_ns / 1_000_000:.3f} ms)")

        memory_bytes = int(metrics.get("memory_bytes", 0) or 0)
        disk_bytes = int(metrics.get("disk_bytes", 0) or 0)
        lines.append(f"Memory footprint: {self._format_bytes(memory_bytes)}")
        lines.append(f"Disk footprint: {self._format_bytes(disk_bytes)}")

        memtable = lsm_snapshot.get("memtable", [])
        levels = lsm_snapshot.get("levels", [])
        flushes = lsm_snapshot.get("flush_count", 0)
        compactions = lsm_snapshot.get("compaction_count", 0)

        lines.append("")
        lines.append(f"Flush count: {flushes}")
        lines.append(f"Compaction count: {compactions}")
        lines.append(f"Memtable size: {len(memtable)} records")

        tombstones = sum(1 for record in memtable if record.get("tombstone"))
        lines.append(f"Memtable tombstones: {tombstones}")

        lines.append("")
        lines.append("Levels summary:")
        for index, level in enumerate(levels):
            level_size = sum(len(table) for table in level)
            tombstone_count = sum(
                1 for table in level for record in table if record.get("tombstone")
            )
            lines.append(
                f"  Level {index}: {level_size} records across {len(level)} tables (tombstones={tombstone_count})"
            )

        return "\n".join(lines)

    def _operation_steps(self, operation: str, result: dict[str, object]) -> list[str]:
        steps = ["Validated input data."]
        lsm_metrics = result.get("lsm_metrics", {}) or {}
        if operation in {"insert", "update"}:
            steps.append("Applied change to LSM memtable.")
            if lsm_metrics.get("file_io_ns", 0):
                steps.append("Flushed memtable into a new SSTable level.")
            steps.append("Applied change to B+ tree records.")
        elif operation == "delete":
            steps.append("Removed matching ID from LSM memtable if present.")
            steps.append("Removed matching ID from B+ tree if present.")
        elif operation == "search":
            steps.append("Scanned LSM memtable and SSTables for the ID.")
            steps.append("Probed B+ tree leaves for the ID.")
            found = "Found" if result.get("lsm_result") or result.get("bptree_result") else "No matches"
            steps.append(f"Captured search results ({found.lower()}).")
        steps.append("Recorded benchmark metrics.")
        return steps

    def _format_metric_lines(self, label: str, metrics: dict[str, object]) -> list[str]:
        lines: list[str] = [f"  {label}:"]
        in_memory_ns = int(metrics.get("in_memory_ns", 0) or 0)
        lines.append(
            f"    in-memory: {in_memory_ns:,} ns ({in_memory_ns / 1_000_000:.3f} ms)"
        )
        file_io_ns = int(metrics.get("file_io_ns", 0) or 0)
        if file_io_ns:
            lines.append(
                f"    flush/disk: {file_io_ns:,} ns ({file_io_ns / 1_000_000:.3f} ms)"
            )
        memory_bytes = int(metrics.get("memory_bytes", 0) or 0)
        disk_bytes = int(metrics.get("disk_bytes", 0) or 0)
        lines.append(f"    memory usage: {self._format_bytes(memory_bytes)}")
        if disk_bytes:
            lines.append(f"    disk usage: {self._format_bytes(disk_bytes)}")
        return lines

    def _format_backend_status(self, memory_bytes: int, disk_bytes: int) -> str:
        mode = self.controller.backend_mode
        mode_label = "On-disk" if mode == "disk" else "In-memory"
        usage_parts = [f"RAM {self._format_bytes(memory_bytes)}"]
        if disk_bytes:
            usage_parts.append(f"Disk {self._format_bytes(disk_bytes)}")
        suffix = ", ".join(usage_parts)
        base_path = self.controller.backend_base_path
        path_suffix = f" • {base_path}" if base_path and mode == "disk" else ""
        return f"Backend: {mode_label} ({suffix}){path_suffix}"

    def _format_student_records(self, records: list[dict[str, object]]) -> str:
        if not records:
            return "No records present."
        lines = [f"Total records: {len(records)}", ""]
        for index, record in enumerate(records, start=1):
            summary = (
                f"id={record.get('id')} gender={record.get('gender', '')} "
                f"race={record.get('race_ethnicity', '')} math={record.get('math_score', '')} "
                f"reading={record.get('reading_score', '')} writing={record.get('writing_score', '')}"
            )
            lines.append(f"{index:02d}. {summary.strip()}")
        return "\n".join(lines)

    def _register_lsm_dialog(self, tag: str, title: str, content: str) -> None:
        self._lsm_canvas_dialogs[tag] = (title, content)
        self.lsm_canvas.tag_bind(tag, "<Button-1>", lambda _event, key=tag: self._show_canvas_dialog(key))
        self.lsm_canvas.tag_bind(tag, "<Enter>", lambda _event: self.lsm_canvas.configure(cursor="hand2"))
        self.lsm_canvas.tag_bind(tag, "<Leave>", lambda _event: self.lsm_canvas.configure(cursor=""))

    def _show_canvas_dialog(self, tag: str) -> None:
        dialog_entry = self._lsm_canvas_dialogs.get(tag)
        if not dialog_entry:
            return
        title, content = dialog_entry
        self._show_copyable_dialog(title, content)

    def _show_copyable_dialog(self, title: str, body: str) -> None:
        dialog = tk.Toplevel(self.root)
        dialog.title(title)
        dialog.transient(self.root)
        dialog.grab_set()
        dialog.resizable(True, True)

        text_frame = ttk.Frame(dialog, padding=12)
        text_frame.pack(fill=tk.BOTH, expand=True)

        scrollbar = ttk.Scrollbar(text_frame, orient=tk.VERTICAL)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        text_widget = tk.Text(
            text_frame,
            wrap=tk.WORD,
            font=("Consolas", 10),
            height=18,
            width=64,
            yscrollcommand=scrollbar.set,
        )
        text_widget.insert(tk.END, body)
        text_widget.configure(state=tk.NORMAL)
        text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.configure(command=text_widget.yview)

        button_row = ttk.Frame(dialog, padding=(12, 0))
        button_row.pack(fill=tk.X, pady=(0, 12))

        ttk.Button(button_row, text="Copy", command=lambda: self._copy_to_clipboard(body)).pack(side=tk.LEFT)
        ttk.Button(button_row, text="Close", command=dialog.destroy).pack(side=tk.RIGHT)

        dialog.bind("<Escape>", lambda _event: dialog.destroy())
        dialog.protocol("WM_DELETE_WINDOW", dialog.destroy)
        dialog.update_idletasks()
        dialog.minsize(360, 240)
        text_widget.focus_set()

    def _copy_to_clipboard(self, value: str) -> None:
        self.root.clipboard_clear()
        self.root.clipboard_append(value)

    # ------------------------------------------------------------------
    # Utility methods
    # ------------------------------------------------------------------
    def _append_operation_log(self, result: dict[str, object]) -> None:
        message = (
            f"{result['operation']} id={result['student_id']} "
            f"lsm={result['lsm_metrics']} bptree={result['bptree_metrics']}"
        )
        current = self.operations_text.get("1.0", tk.END).strip()
        updated = (current + "\n" + message).strip()
        self._set_text(self.operations_text, updated)

    def _show_error(self, message: str) -> None:
        messagebox.showerror("LSM Studio", message)

    def _clear_tree(self, tree: ttk.Treeview) -> None:
        for row in tree.get_children():
            tree.delete(row)

    def _clear_canvas(self, canvas: tk.Canvas) -> None:
        canvas.delete("all")

    def _set_text(self, widget: tk.Text, value: str) -> None:
        widget.configure(state=tk.NORMAL)
        widget.delete("1.0", tk.END)
        widget.insert(tk.END, value)
        widget.configure(state=tk.DISABLED)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run(self) -> None:
        self.root.mainloop()
