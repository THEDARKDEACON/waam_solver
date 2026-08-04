"""
test_hardware_preserves_domain.py — --preset / preset_override keeps job geometry.

Hardware profiles may coarsen dx to fit vram_budget / max_cells, but must not
rewrite domain_mm or plate.size_mm. Presets.yaml no longer carries domain_mm.
"""

from __future__ import annotations

import sys

from waam_twin.platform import init_taichi, reset_taichi, load_presets
from waam_twin import WAAMTwin
from waam_twin.job import resolve_plate_and_domain


def run() -> None:
    presets = load_presets()
    for cfg in presets.values():
        assert "domain_mm" not in cfg.__dataclass_fields__

    reset_taichi()
    init_taichi(backend="cpu")

    twin = WAAMTwin.from_job(
        "jobs/examples/bead_on_plate.yaml",
        preset_override="minimal",
    )
    g = twin.grid
    dx_mm = g.dx * 1e3
    dom = (g.nx * dx_mm, g.ny * dx_mm, g.nz * dx_mm)

    # Job asks for domain 80×80×25 and plate 50×50.
    assert abs(dom[0] - 80.0) < 5.0, f"domain X rewritten: {dom}"
    assert abs(dom[1] - 80.0) < 5.0, f"domain Y rewritten: {dom}"
    assert twin.plate_size_mm == (50.0, 50.0)

    i0, i1, j0, j1 = twin.resolve_plate_ij()
    span_x = (i1 - i0) * dx_mm
    span_y = (j1 - j0) * dx_mm
    assert abs(span_x - 50.0) < dx_mm * 2.0, f"plate X span={span_x}"
    assert abs(span_y - 50.0) < dx_mm * 2.0, f"plate Y span={span_y}"
    assert abs(span_x / span_y - 1.0) < 0.05, "top view should be square"

    # Derive domain from plate when domain_mm omitted.
    derived = resolve_plate_and_domain(
        {"plate": {"size_mm": [50, 50], "thickness_mm": 10.0, "domain_margin_mm": 15.0}},
        0.4,
    )
    assert derived["domain_mm"] == (80.0, 80.0, 25.0), derived["domain_mm"]
    assert derived["plate_size_mm"] == (50.0, 50.0)

    try:
        resolve_plate_and_domain(
            {"simulation": {"domain_mm": [40, 20, 15]}, "plate": {"size_mm": [100, 100]}},
            0.5,
        )
        raise AssertionError("expected ValueError for plate > domain")
    except ValueError as exc:
        assert "exceeds" in str(exc).lower()

    print(
        f"[hardware_domain] OK  domain≈{dom[0]:.0f}×{dom[1]:.0f}×{dom[2]:.0f} mm  "
        f"plate=50×50  dx={dx_mm:.3f} mm  grid={g.nx}×{g.ny}×{g.nz}"
    )


if __name__ == "__main__":
    try:
        run()
        print("PASS")
    except Exception as exc:
        print(f"FAIL: {exc}")
        sys.exit(1)
