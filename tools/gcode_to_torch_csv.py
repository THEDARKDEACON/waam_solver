"""
gcode_to_torch_csv.py — G-code or KRL .src → waam_twin torch_path CSV.

Usage (from FYP22-01 with parent on PYTHONPATH for G-code):
    PYTHONPATH=. python3 -m waam_twin.tools.gcode_to_torch_csv part.gcode -o waam_twin/jobs/paths/part.csv
    PYTHONPATH=. python3 -m waam_twin.tools.gcode_to_torch_csv --krl-src part.src -o waam_twin/jobs/paths/part.csv
"""

from __future__ import annotations

import argparse
import pathlib
import sys


def _parent_fyp_on_path() -> None:
    """Allow importing gcode_pipeline from FYP22-01 parent when nested."""
    here = pathlib.Path(__file__).resolve()
    parent = here.parents[2]  # FYP22-01
    if parent.name == "waam_twin":
        parent = parent.parent
    pstr = str(parent)
    if pstr not in sys.path:
        sys.path.insert(0, pstr)


def gcode_to_segments(raw_gcode: str, program_name: str = "WAAM_PART") -> list:
    _parent_fyp_on_path()
    from gcode_pipeline import clean_and_transpile

    result = clean_and_transpile(raw_gcode, program_name=program_name)
    return result.get("segments", [])


def krl_src_to_segments(src_text: str) -> list:
    _parent_fyp_on_path()
    from visualize_toolpath import extract_toolpath_segments_from_text

    return extract_toolpath_segments_from_text(src_text)


def main() -> int:
    parser = argparse.ArgumentParser(description="Export torch path CSV for waam_twin jobs")
    parser.add_argument("input", nargs="?", help="G-code file (.gcode)")
    parser.add_argument("--krl-src", help="KRL .src file (skip G-code transpile)")
    parser.add_argument("-o", "--output", required=True, help="Output CSV path")
    parser.add_argument("--all-moves", action="store_true", help="Include travel (torch FALSE) moves")
    parser.add_argument("--program-name", default="WAAM_PART")
    args = parser.parse_args()

    from waam_twin.toolpath.export import segments_to_csv_file

    if args.krl_src:
        text = pathlib.Path(args.krl_src).read_text(encoding="utf-8", errors="replace")
        segments = krl_src_to_segments(text)
    elif args.input:
        text = pathlib.Path(args.input).read_text(encoding="utf-8", errors="replace")
        segments = gcode_to_segments(text, program_name=args.program_name)
    else:
        parser.error("Provide input.gcode or --krl-src")

    out = segments_to_csv_file(segments, args.output, weld_only=not args.all_moves)
    n_pts = sum(
        len(s.get("points", []))
        for s in segments
        if not args.all_moves or str(s.get("torch_state", "TRUE")).upper() == "TRUE"
        or args.all_moves
    )
    print(f"[gcode_to_torch_csv] Wrote {out}  ({len(segments)} segments, ~{n_pts} KRL points)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
