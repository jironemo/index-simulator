from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import tkinter as tk
from tkinter import filedialog, messagebox, ttk


@dataclass(slots=True)
class BackendSelection:
    backend_type: str
    disk_path: Optional[Path]


def prompt_backend_selection(
    *,
    default_backend: str = "memory",
    default_disk_path: Optional[Path] = None,
) -> BackendSelection | None:
    """Show a modal dialog asking the user to choose the backend."""

    env_backend = os.getenv("LSM_BACKEND")
    if env_backend:
        backend = env_backend.strip().lower()
        disk_env = os.getenv("LSM_BACKEND_PATH")
        disk_path = Path(disk_env).expanduser().resolve() if disk_env else default_disk_path
        return BackendSelection(backend_type=backend, disk_path=disk_path)

    try:
        root = tk.Tk()
    except tk.TclError:
        # Fall back to default backend when a display is unavailable
        return BackendSelection(default_backend, default_disk_path)

    root.title("Select Simulation Backend")
    root.geometry("420x220")
    root.resizable(False, False)

    selection_var = tk.StringVar(value=default_backend)
    disk_path_var = tk.StringVar(
        value=str((default_disk_path or Path("./storage_data")).expanduser().resolve())
    )
    result: dict[str, object] = {"cancelled": False}

    def on_backend_change(*_args: object) -> None:
        state = tk.NORMAL if selection_var.get() == "disk" else tk.DISABLED
        disk_entry.configure(state=state)
        browse_button.configure(state=state)

    def on_browse() -> None:
        path = filedialog.askdirectory()
        if path:
            disk_path_var.set(path)

    def on_ok() -> None:
        result["backend"] = selection_var.get()
        if result["backend"] == "disk":
            result["disk_path"] = Path(disk_path_var.get()).expanduser().resolve()
        else:
            result["disk_path"] = None
        root.quit()

    def on_cancel() -> None:
        result["cancelled"] = True
        root.quit()

    main = ttk.Frame(root, padding=16)
    main.pack(fill=tk.BOTH, expand=True)

    ttk.Label(main, text="Choose simulation backend:", font=("Segoe UI", 11, "bold")).pack(anchor=tk.W)

    radio_frame = ttk.Frame(main)
    radio_frame.pack(fill=tk.X, pady=(12, 0))

    ttk.Radiobutton(
        radio_frame,
        text="In-memory (default)",
        variable=selection_var,
        value="memory",
        command=on_backend_change,
    ).pack(anchor=tk.W)

    ttk.Radiobutton(
        radio_frame,
        text="On-disk (persists state)",
        variable=selection_var,
        value="disk",
        command=on_backend_change,
    ).pack(anchor=tk.W)

    disk_frame = ttk.Frame(main)
    disk_frame.pack(fill=tk.X, pady=(12, 0))
    ttk.Label(disk_frame, text="Data directory:").grid(row=0, column=0, sticky=tk.W)

    disk_entry = ttk.Entry(disk_frame, textvariable=disk_path_var, width=32)
    disk_entry.grid(row=0, column=1, padx=(8, 8), sticky=tk.EW)
    browse_button = ttk.Button(disk_frame, text="Browse", command=on_browse)
    browse_button.grid(row=0, column=2)
    disk_frame.columnconfigure(1, weight=1)

    button_frame = ttk.Frame(main)
    button_frame.pack(fill=tk.X, pady=(24, 0))

    ttk.Button(button_frame, text="Cancel", command=on_cancel).pack(side=tk.RIGHT, padx=(8, 0))
    ttk.Button(button_frame, text="Start", command=on_ok).pack(side=tk.RIGHT)

    on_backend_change()
    root.protocol("WM_DELETE_WINDOW", on_cancel)
    root.mainloop()
    root.destroy()

    if result.get("cancelled"):
        return None

    backend = str(result.get("backend", default_backend)).lower()
    disk_path = result.get("disk_path")
    if backend == "disk" and disk_path is None:
        messagebox.showerror("Invalid Configuration", "Disk backend requires a data directory.")
        return prompt_backend_selection(default_backend=default_backend, default_disk_path=default_disk_path)
    return BackendSelection(backend_type=backend, disk_path=disk_path if isinstance(disk_path, Path) else None)
