"""
test_viewer_session.py — Viewer session loads job twin without opening GGUI.
"""

from __future__ import annotations

from waam_twin.platform import init_taichi
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
    print(
        f"[viewer_session] job={session.job_label} grid={g.nx}×{g.ny}×{g.nz} "
        f"pool_W={telem['pool_width_mm']:.2f}mm"
    )


if __name__ == "__main__":
    run()
    print("PASS")
