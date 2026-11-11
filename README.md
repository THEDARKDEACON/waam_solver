# waam_twin

GPU melt-pool digital twin for wire-arc additive manufacturing (WAAM / GMAW).
It predicts the bead cross-section (width, penetration, height), the temperature
field and the Marangoni-driven flow of a single deposited bead.

**Solver:** D3Q19 TRT lattice-Boltzmann flow with a free-surface LBM interface,
enthalpy–porosity solidification, a Goldak or surface arc heat source, droplet
transfer and surfactant-dependent surface tension (Fe–S–O Sahoo isotherm).
Kernels are compiled with [Quadrants](https://github.com/Genesis-Embodied-AI/quadrants)
(a Taichi fork; the source keeps the `ti` spelling).

## Status (Nov 2026)

Calibrated and partly validated against published beads. Each test was
pre-registered before running; material properties are never fitted.

| Dataset | Fitted | Held-out result |
|---|---|---|
| Park et al. 2019, P-GMAW, measured U(t)·I(t) | η = 0.87 | 6 of 8 outputs within tolerance; the 2 failures were predicted (vertical-down depth, Bruno width) — [PARK_MEASURED_PREREG](docs/PARK_MEASURED_PREREG.md) |
| Yanala, Andreucci & Lambiase 2026, 14 unseen beads | η = 0.82 | 5 of 7 checks pass: penetration 23 % MAE, all trend signs, physical η; height 11.8 % (limit 10 %) and contact length fail — [LAMBIASE_PREREG](docs/LAMBIASE_PREREG.md) |

Known limits:
- Bead width responds too weakly to arc power (under-resolved spreading).
- Penetration comes from a Goldak volume fixed to the torch; no arc crater forms.
- Not mesh-converged: beads get taller and narrower as dx shrinks, and no mesh
  study has been repeated since the surface-height race fix ([MESH_STUDY](docs/MESH_STUDY.md)).
- Shop resolution (dx 0.4 mm) hits the numerical speed cap on part of the steps.
- Runs are 100–1500× slower than real time on an RTX 3050.

See [CALIBRATION](docs/CALIBRATION.md) and [VERIFICATION](docs/VERIFICATION.md).

## Install

Python ≥ 3.10, Linux, NVIDIA (CUDA) or AMD (ROCm) GPU recommended; CPU works for small runs.

```bash
git clone https://github.com/THEDARKDEACON/waam_solver.git waam_twin
cd waam_twin
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[export]"          # export = PyVista viewer + VTK output
python -c "from waam_twin.compiler import ti; ti.init(arch=ti.cuda); print(ti.cfg.arch)"
```

Choose the backend with `WAAM_BACKEND=cuda|amdgpu|vulkan|cpu`.

## Use

```bash
# Interactive viewer (M = colour mode, X/Y/Z = section planes; keys in docs/REFERENCE.md)
python -m waam_twin.viewer --job jobs/examples/bead_park_measured.yaml --sparse

# Steady bead cross-section (W, D, H in a window behind the torch)
python -m waam_twin.tools.steady_bead --job jobs/examples/bead_park_measured.yaml \
    --time-s 2.5 --x0 22 --x1 26 --out runs/steady/my_run.jsonl

# Calibration: Latin hypercube → Gaussian-process surrogate → posterior → held-out
S=jobs/calibration/sweep_lambiase.yaml
for stage in plan run fit; do python -m waam_twin.tools.calib_sweep $stage $S; done
python -m waam_twin.tools.calib_sweep validate $S --draws 0

# Regression suite
WAAM_BACKEND=cpu WAAM_PRESET=minimal python -m waam_twin.validation.run_all
```

From Python:

```python
from waam_twin.runtime import init_taichi
from waam_twin import WAAMTwin

init_taichi()
twin = WAAMTwin.from_job("jobs/examples/bead_park_measured.yaml")
twin.reset()
twin.run_path("jobs/examples/bead_park_measured.yaml", n_steps=800)
print(twin.get_telemetry())
```

A run is described by a **job YAML** (plate, process, heat source, path,
reference measurements) and a **material YAML** (`materials/`). Electrical power
is `process.mean_power_W` (measured mean of U·I) or U·I; the solver deposits
η·P split between arc and droplets.

## Layout

| Path | Contents |
|---|---|
| `twin.py`, `job.py`, `materials.py` | Orchestrator, job and material loaders |
| `kernels/`, `solvers/`, `collision.py` | GPU kernels, time-step order, TRT collision |
| `physics/` | Host-side physics: arc, droplets, surfactant, forces, deposition |
| `viewer/`, `export/` | PyVista viewer, VTK / ParaView export |
| `tools/` | `steady_bead`, `calib_sweep`, `eval_lambiase`, diagnostics |
| `jobs/examples/`, `jobs/calibration/` | Bead jobs (Park, Bruno, Lambiase) and calibration specs |
| `docs/` | Physics, calibration, pre-registrations and validation write-ups |
| `validation/` | Regression tests and baselines |
| `runs/` | Outputs (gitignored apart from closed-out evidence) |

## Documentation

- [docs/VALIDATION_WRITEUP.md](docs/VALIDATION_WRITEUP.md) — calibration and validation summary
- [docs/REFERENCE.md](docs/REFERENCE.md) — job keys, viewer controls, environment variables, presets, telemetry
- [docs/HPC.md](docs/HPC.md) — Docker / SLURM production runs
- [docs/CALIBRATION.md](docs/CALIBRATION.md) — what is fitted and what is fixed
- [docs/PHYSICS_FORCE_CORRECTNESS_SPEC.md](docs/PHYSICS_FORCE_CORRECTNESS_SPEC.md), [docs/weld_pool_physics.md](docs/weld_pool_physics.md) — force models
- [docs/physics/LBM.md](docs/physics/LBM.md) — lattice units, TRT, forcing
- [docs/MATERIALS.md](docs/MATERIALS.md) — material cards
