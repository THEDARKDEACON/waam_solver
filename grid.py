"""
grid.py — GPU field allocation (strict SoA, waam_twin v2)
==========================================================
VRAM scales with grid size; use platform.auto_grid() to fit host memory.
Typical budget at 256×128×64: ~400–500 MB for LBM fields + tracers.

SoA rule: all fields are ti.field(shape=(nx, ny, nz)) or (Q, nx, ny, nz).
Never use ti.Struct / ti.StructField (AoS breaks GPU coalescing).
"""

import taichi as ti
from .materials import MaterialProps


# D3Q19 velocity set — 19 discrete velocity directions
Q = 19

# D3Q19 discrete velocities (ex, ey, ez) — used in kernels
_EX = ( 0, 1,-1, 0, 0, 0, 0, 1,-1, 1,-1, 1,-1, 1,-1, 0, 0, 0, 0)
_EY = ( 0, 0, 0, 1,-1, 0, 0, 1,-1,-1, 1, 0, 0, 0, 0, 1,-1, 1,-1)
_EZ = ( 0, 0, 0, 0, 0, 1,-1, 0, 0, 0, 0, 1,-1,-1, 1, 1,-1,-1, 1)

# D3Q19 equilibrium weights
_W = (
    1.0/3.0,                                        # q=0  (rest)
    1.0/18.0, 1.0/18.0, 1.0/18.0,                  # q=1..3
    1.0/18.0, 1.0/18.0, 1.0/18.0,                  # q=4..6
    1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0,        # q=7..10
    1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0,        # q=11..14
    1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0,        # q=15..18
)

# Opposite direction lookup (for bounce-back boundaries)
_OPP = (0, 2,1, 4,3, 6,5, 8,7, 10,9, 12,11, 14,13, 16,15, 18,17)


class WAAMGrid:
    """
    Container for all GPU-resident fields and physical scaling parameters.

    Attributes (all ti.field, SoA):
        f_a, f_b : D3Q19 distribution functions (ping-pong buffers)
        T        : Temperature field                     [K]
        T_max    : Maximum temperature reached (HAZ)     [K]
        H        : Enthalpy field (thermal energy store) [J/m³]
        f_l      : Liquid fraction  0=solid … 1=liquid   [-]
        phi      : VOF phase field  0=gas   … 1=liquid   [-]
        rho      : Macroscopic density                   [lu]
        ux,uy,uz : Macroscopic velocity components       [lu/ts]
        Fx,Fy,Fz : Total body force (Marangoni+buoyancy) [lu/ts²]
        flags    : Cell type bitmask (see FLAG_* constants)
        
        porosity_pos    : Position of tracer particles [m]
        porosity_active : Whether the tracer is active (1) or inactive (0)

    LBM Lattice Units (lu):
        Δx  = physical cell size in metres
        Δt  = physical timestep in seconds = Δx / c_s / sqrt(3)  (Ma~0.1)
        c_s = lattice speed of sound = 1/sqrt(3)
    """

    # Cell type flags (compatible with FluidX3D convention for future porting)
    FLAG_FLUID  = 0
    FLAG_SOLID  = 1   # static solid wall
    FLAG_GAS    = 2   # gas cell (above free surface)
    FLAG_IFACE  = 4   # VOF interface cell

    def __init__(
        self,
        nx: int,
        ny: int,
        nz: int,
        dx: float,                  # Physical cell size [m]
        mat: MaterialProps,
        max_tracers: int = 50_000,
    ):
        self.nx, self.ny, self.nz = nx, ny, nz
        self.dx = dx
        self.mat = mat
        self.max_tracers = max_tracers

        # ── LBM non-dimensionalisation ────────────────────────────────────
        # We target Ma ≈ 0.05 for stability with molten steel velocities.
        # Physical max velocity ≈ 0.5 m/s (Marangoni jets)
        # Ma = u_max_lu / c_s = 0.05  →  u_max_lu = 0.05 / sqrt(3) ≈ 0.029 lu/ts
        self.u_ref_phys = 0.5          # m/s  (physical reference velocity)
        self.u_ref_lu   = 0.05         # lu/ts (lattice reference velocity)
        self.dt         = dx * self.u_ref_lu / self.u_ref_phys   # [s]

        # Kinematic viscosity in lattice units
        nu_phys = mat.mu / mat.rho     # m²/s
        nu_lu   = nu_phys * self.dt / (dx ** 2)
        self.tau = 3.0 * nu_lu + 0.5  # SRT relaxation time (will use MRT/Cumulant)

        # Thermal diffusivity in lattice units
        alpha_lu = mat.alpha * self.dt / (dx ** 2)
        self.tau_T = 3.0 * alpha_lu + 0.5  # Thermal relaxation time

        print(f"[WAAMGrid] Domain: {nx}×{ny}×{nz}  |  dx={dx*1000:.3f}mm  "
              f"dt={self.dt*1e6:.3f}µs  τ={self.tau:.4f}  τ_T={self.tau_T:.4f}")

        # ── GPU field allocation (strict SoA) ────────────────────────────
        shape3 = (nx, ny, nz)

        # Distribution functions — two full sets for ping-pong streaming
        self.f_a = ti.field(dtype=ti.f32, shape=(Q, nx, ny, nz))
        self.f_b = ti.field(dtype=ti.f32, shape=(Q, nx, ny, nz))

        # Macroscopic scalars
        self.rho   = ti.field(dtype=ti.f32, shape=shape3)
        self.T     = ti.field(dtype=ti.f32, shape=shape3)   # Temperature [K]
        self.T_max = ti.field(dtype=ti.f32, shape=shape3)   # Peak Temperature [K] (HAZ)
        self.H     = ti.field(dtype=ti.f32, shape=shape3)   # Enthalpy [J/m³]
        self.f_l   = ti.field(dtype=ti.f32, shape=shape3)   # Liquid fraction
        self.phi   = ti.field(dtype=ti.f32, shape=shape3)   # VOF phase field
        self.cp_rho_field = ti.field(dtype=ti.f32, shape=shape3)   # ρ·cp(T) [J/(m³·K)]
        self.alpha_lu_field = ti.field(dtype=ti.f32, shape=shape3) # α_lu per cell
        self.dgamma_lu_field = ti.field(dtype=ti.f32, shape=shape3)  # dγ/dT in LBM units
        self.tau_field = ti.field(dtype=ti.f32, shape=shape3)        # per-cell SRT τ
        self.phi_tmp = ti.field(dtype=ti.f32, shape=shape3)    # VOF advection buffer
        self.dT_dt = ti.field(dtype=ti.f32, shape=shape3)   # Cooling rate [K/s]
        self.T_prev = ti.field(dtype=ti.f32, shape=shape3)  # Previous-step T
        # Time-at-temperature integrals [s] (HAZ research)
        self.time_above_800_s = ti.field(dtype=ti.f32, shape=shape3)
        self.time_above_1100_s = ti.field(dtype=ti.f32, shape=shape3)
        self.time_above_solidus_s = ti.field(dtype=ti.f32, shape=shape3)
        # Export / diagnostics buffers (filled on demand)
        self.kappa_field = ti.field(dtype=ti.f32, shape=shape3)
        self.vorticity_mag = ti.field(dtype=ti.f32, shape=shape3)
        # End-of-step body-force snapshot for VTK [lu/ts²]
        self.Fx_snap = ti.field(dtype=ti.f32, shape=shape3)
        self.Fy_snap = ti.field(dtype=ti.f32, shape=shape3)
        self.Fz_snap = ti.field(dtype=ti.f32, shape=shape3)
        self.surface_k_buf = ti.field(dtype=ti.f32, shape=())  # arc height query
        self.deposit_vol_buf = ti.field(dtype=ti.f32, shape=())  # last droplet volume [m³]

        # Tracer particles for porosity tracking
        self.porosity_pos    = ti.Vector.field(3, dtype=ti.f32, shape=self.max_tracers)
        self.porosity_active = ti.field(dtype=ti.i32, shape=self.max_tracers)
        # Atomic ring-buffer head — O(1) slot allocation without serial scan
        self.tracer_head     = ti.field(dtype=ti.i32, shape=())

        # Macroscopic velocity components (SoA: separate fields per axis)
        self.ux = ti.field(dtype=ti.f32, shape=shape3)
        self.uy = ti.field(dtype=ti.f32, shape=shape3)
        self.uz = ti.field(dtype=ti.f32, shape=shape3)

        # Body force components
        self.Fx = ti.field(dtype=ti.f32, shape=shape3)
        self.Fy = ti.field(dtype=ti.f32, shape=shape3)
        self.Fz = ti.field(dtype=ti.f32, shape=shape3)

        # Cell type flags
        self.flags = ti.field(dtype=ti.i32, shape=shape3)

        # Expose velocity set constants as Taichi fields for kernel access
        self.ex = ti.field(dtype=ti.i32, shape=(Q,))
        self.ey = ti.field(dtype=ti.i32, shape=(Q,))
        self.ez = ti.field(dtype=ti.i32, shape=(Q,))
        self.w  = ti.field(dtype=ti.f32, shape=(Q,))
        self.opp = ti.field(dtype=ti.i32, shape=(Q,))

        import numpy as np
        self.ex.from_numpy(np.array(_EX, dtype=np.int32))
        self.ey.from_numpy(np.array(_EY, dtype=np.int32))
        self.ez.from_numpy(np.array(_EZ, dtype=np.int32))
        self.w.from_numpy(np.array(_W,   dtype=np.float32))
        self.opp.from_numpy(np.array(_OPP, dtype=np.int32))

        # ── Pointer swap state ───────────────────────────────────────────
        # We swap src/dst each step to avoid copying distribution data.
        self._read_buf  = self.f_a
        self._write_buf = self.f_b

    # ──────────────────────────────────────────────────────────────────────
    #  Public helpers
    # ──────────────────────────────────────────────────────────────────────

    def swap_buffers(self):
        """O(1) ping-pong: swaps read/write distribution buffer pointers."""
        self._read_buf, self._write_buf = self._write_buf, self._read_buf

    @property
    def f_src(self) -> ti.Field:
        return self._read_buf

    @property
    def f_dst(self) -> ti.Field:
        return self._write_buf

    def estimated_vram_mb(self) -> float:
        """Rough VRAM estimate based on field shapes."""
        n = self.nx * self.ny * self.nz
        dist = 2 * Q * n * 4          # f_a, f_b
        scalar = 22 * n * 4            # rho..tau_field + HAZ time + export buffers
        vector = 6 * n * 4            # ux,uy,uz,Fx,Fy,Fz
        tracers = self.max_tracers * (3 * 4 + 4) # pos(vec3) + active(int)
        return (dist + scalar + vector + tracers) / (1024 ** 2)
