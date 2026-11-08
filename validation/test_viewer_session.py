"""
test_viewer_session.py — Viewer session loads job twin without opening a window.
"""

from __future__ import annotations

from waam_twin.runtime import init_taichi
from waam_twin.viewer.session import create_session


def run() -> None:
    init_taichi(backend="cpu")
    session = create_session(job="jobs/examples/bead_on_plate.yaml")
    twin = session.twin
    g = twin.grid
    assert twin.enable_vof, "bead_on_plate job should enable VOF"
    assert session.torch_spd_m_s > 0
    session.advance_physics(5, is_welding=True)
    telem = twin.get_telemetry()
    assert telem["step"] == 5
    i, j, k = session.torch_cell()
    assert 0 <= i < g.nx and 0 <= j < g.ny and k >= twin.nz_solid - 1, (i, j, k)
    # HUD bead section: none until the torch has travelled the lag, then a
    # plane across the travel direction (bead_line.csv travels +Y → j-plane).
    assert session.section_plane(3.0) is None
    step0 = twin._step_n
    twin._step_n = int(0.010 / session.path_driver.speed / g.dt)  # 10 mm travelled
    plane = session.section_plane(3.0)
    twin._step_n = step0
    assert plane is not None and plane[0] == 1, plane
    print(
        f"[viewer_session] job={session.job_label} grid={g.nx}×{g.ny}×{g.nz} "
        f"pool_W={telem['pool_width_mm']:.2f}mm"
    )


if __name__ == "__main__":
    run()
    print("PASS")
