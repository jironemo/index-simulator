from __future__ import annotations

from pathlib import Path

from config import BPTREE_ORDER, LSM_LEVELS, LSM_MEMTABLE_THRESHOLD
from services.simulation_controller import SimulationController
from ui.backend_selector import BackendSelection, prompt_backend_selection
from ui.view import LSMStudioView


def run() -> None:
    default_disk_path = Path("./storage_data").resolve()
    selection: BackendSelection | None = prompt_backend_selection(
        default_backend="memory",
        default_disk_path=default_disk_path,
    )
    if selection is None:
        return

    controller = SimulationController(
        memtable_threshold=LSM_MEMTABLE_THRESHOLD,
        num_levels=LSM_LEVELS,
        bptree_order=BPTREE_ORDER,
        backend_type=selection.backend_type,
        disk_path=selection.disk_path,
    )
    view = LSMStudioView(controller)
    view.run()


if __name__ == "__main__":
    run()
