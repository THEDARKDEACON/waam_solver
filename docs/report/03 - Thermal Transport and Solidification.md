---
title: Thermal Transport and Solidification
step: 3
tags:
  - report/step
  - thermal
  - phase-change
up: "[[WAAM Digital Twin — Project Report]]"
prev: "[[02 - The Lattice Boltzmann Core]]"
next: "[[04 - The Arc Heat Source]]"
---

# 03 — Thermal Transport and Solidification

> [!summary] The step in one sentence
> I made **enthalpy** the conserved variable rather than temperature, which is what lets latent heat, melting and solidification all fall out of a single fixed-grid update with no explicit interface tracking.

---

## Enthalpy–porosity: the choice that structures everything

Temperature is a bad primary variable for a melting problem, because at the melting point a lot of energy goes in and the temperature does not move. So I carry volumetric enthalpy $H$ [J/m³] and derive temperature and liquid fraction from it.

With $H_\text{sol} = \rho c_p T_\text{solidus}$ and $H_\text{liq} = H_\text{sol} + \rho L_f$:

| Regime | Liquid fraction $f_l$ | Temperature |
|---|---|---|
| $H \le H_\text{sol}$ | $0$ | $H/(\rho c_p)$ |
| $H_\text{sol} < H < H_\text{liq}$ | $(H - H_\text{sol})/(\rho L_f)$ | $T_\text{sol} + f_l(T_\text{liq} - T_\text{sol})$ |
| $H \ge H_\text{liq}$ | $1$ | $T_\text{liq} + (H - H_\text{liq})/(\rho c_p)$ |

> [!cite] Source
> This is the enthalpy–porosity technique of **Voller & Prakash (1987)** and **Brent, Voller & Reid (1988)**. It is the standard fixed-grid approach for convection–diffusion phase change, and it is why I never have to track a melt front explicitly — the mushy zone is just cells with intermediate $f_l$.

> [!note] A subtlety worth flagging
> Notice that $H_\text{liq} = \rho c_p T_\text{solidus} + \rho L$, using the **solidus**, not the liquidus. Any clamp I apply to enthalpy must use the same convention. When my vapour ceiling did not, recovered temperature overshot the cap by roughly the solidus–liquidus interval (about 45 K) and the peak-temperature readout stuck at a suspicious constant value forever. I now recover the phase state **three times per timestep** — after the thermal step, after the evaporative sink, and after the final clamp — so telemetry can never report a temperature that the enthalpy field no longer supports.

---

## Advection and diffusion

I solve

$$\frac{\partial H}{\partial t} + \mathbf{u}\cdot\nabla H = \nabla\cdot(k\nabla T)$$

with a deliberately hybrid discretisation.

**Diffusion** is a straightforward seven-point central Laplacian on temperature.

**Advection** is a **minmod-limited second-order upwind** scheme, and crucially it acts on **enthalpy, not temperature**, so latent heat is carried along with the moving melt rather than being left behind. The limiter compares the first-order and second-order differences and takes the smaller when they agree in sign, zero otherwise:

$$g = \text{minmod}(d_1, d_2), \qquad \text{minmod}(a,b) = \begin{cases} a & \text{if } ab>0,\ |a|<|b| \\ b & \text{if } ab>0,\ |a|\ge|b| \\ 0 & \text{otherwise}\end{cases}$$

This needs a five-wide stencil per axis. It gives me second-order accuracy in smooth regions without oscillating across the sharp thermal gradient at the fusion boundary.

> [!bug] Two boundary bugs I fixed here
> **Gas neighbours** used to be treated as fixed-ambient-temperature conductors, which made the free surface an enormous unphysical heat sink. They are now adiabatic (zero-gradient). But **solid cells still conduct** — I deliberately left diffusion active in the substrate and only skipped advection, because the plate must be a real heat sink, not an insulator. Getting one of those two right and the other wrong is very easy.

> [!cite] Source
> The upwind-advection-plus-central-diffusion structure follows **Patankar (1980)**, *Numerical Heat Transfer and Fluid Flow*. Fourier conduction and the Stefan benchmark come from **Carslaw & Jaeger (1959)** and **Crank (1984)**.

---

## Mushy-zone drag: Carman–Kozeny

A partially solidified cell should resist flow, going smoothly from free liquid to immobile solid. I apply this inside the collision step rather than as a body force:

$$\mathbf{u}_\text{final} = \frac{\mathbf{u}^*}{1 + C_K\dfrac{(1-f_l)^2}{f_l^3 + \varepsilon}}$$

with $C_K = 1.6\times 10^5$.

> [!tip] Why the division matters
> Writing it as a division makes the treatment **semi-implicit** and therefore unconditionally stable for arbitrarily large $C_K$. An explicit subtraction would blow up as $f_l \to 0$, exactly where the drag is supposed to be strongest.

> [!cite] Source
> Carman–Kozeny permeability as applied to solidification, via **Voller & Prakash (1987)**, with permeability provenance from **Poirier (1987)**.

---

## Temperature-dependent properties

Every step, every cell looks up $c_p(T)$, $k(T)$, $\mu(T)$ and $d\gamma/dT(T)$ from GPU-resident tables and recomputes its local diffusivity and relaxation time:

$$\alpha^{lu} = \frac{k(T)}{\rho c_p(T)}\frac{\Delta t}{\Delta x^2}, \qquad \tau = 3\frac{\mu(T)}{\rho}\frac{\Delta t}{\Delta x^2} + \tfrac{1}{2}$$

The lookup is piecewise-linear with end clamping, and I unrolled the knot search over a fixed maximum of eight knots so it compiles to branchless GPU code. If a material YAML supplies more knots than that, I warn — and under strict mode I raise, rather than silently truncating a carefully measured property curve.

There is also a dual-alloy path that evaluates both the wire and plate tables and blends them by local composition, for studies where filler and substrate differ.

---

## Boundary heat losses

On every gas-exposed face:

$$q_\text{loss} = h_\text{conv}(T - T_\infty) + \varepsilon\sigma_{SB}(T^4 - T_\infty^4)$$

removing $q\,\Delta t/\Delta x$ of enthalpy per cell face.

> [!bug] An error of six orders of magnitude
> An earlier version multiplied this by $\rho c_p$ as well. A surface flux over a cell face removes $q\,\Delta t/\Delta x$ joules per cubic metre — no heat capacity belongs in that conversion. The bug overweighted boundary losses by roughly a million.

Defaults are $h_\text{conv} = 25$ W/m²K and $\varepsilon = 0.3$; my calibrated case uses 35 with both mechanisms active.

---

## Evaporative cooling: Hertz–Knudsen

Early on, my pool would heat until it pinned against a hard temperature ceiling and simply sit there. That ceiling was a numerical safety net masquerading as physics. The proper fix was a real evaporative enthalpy sink.

Saturation pressure from Clausius–Clapeyron, mass flux from Hertz–Knudsen:

$$P_\text{sat}(T) = P_\text{ref}\exp\!\left[\frac{L_v}{R_v}\left(\frac{1}{T_b} - \frac{1}{T}\right)\right], \qquad \dot m = \frac{C_\text{acc}P_\text{sat}}{\sqrt{2\pi R_v T}}, \qquad q = \dot m L_v$$

Three implementation details that mattered:

1. **Soft quadratic onset.** I ramp the sink in as $\left(\frac{T-T_\text{onset}}{T_b-T_\text{onset}}\right)^2$ below boiling, using the *same* schedule as vapour recoil so the two stay consistent.
2. **Cell selection includes hot bulk liquid**, not just the free surface. Because arc surface weighting deposits energy into liquid under the torch, a surface-only sink leaves the interior saturated at the cap.
3. **An energy-conservation cap** prevents the sink from undershooting below the onset temperature in a single step.

I accumulate the removed energy and report it in telemetry, so I can see how much of my thermal budget evaporation is eating.

> [!cite] Source
> Hertz–Knudsen / **Anisimov** mass flux, with accommodation coefficient from **Knight** ($C_\text{acc} \approx 0.54$). Recoil framing from **Matsunawa & Semak (1997)**.

---

## HAZ and cooling-rate tracking

- **`T_max`** is a per-cell running maximum. This is my heat-affected-zone field, and it exports directly to ParaView.
- **Cooling rate** carries a discrete-event filter: if the temperature jump in one step exceeds 25 K, I zero the rate for that step. Droplet injection, freeze clamps and gas-to-metal cell birth would otherwise pollute the signal with spurious values around $10^6$ K/s. Real GMAW HAZ cooling is $10^3$ to a few $\times 10^3$ K/s, and 25 K per 50 µs step is already $5\times10^5$ K/s, so the filter only removes things that were never physical.
- **Time-above-temperature integrals** at 800 °C, 1100 °C and the solidus, for metallurgical interpretation.

---

## How I verified it

| Test | Benchmark | Threshold |
|---|---|---|
| `test_thermal_diffusion` | Analytical Gaussian pulse decay | L2 < **1 %** |
| `test_stefan_solidification` | 1D Stefan front $s(t) = 2\lambda\sqrt{\alpha t}$ | slope error < **8 %** |
| `test_rosenthal_farfield` | Rosenthal moving point source, far field | < **55 %** |
| `test_thermocouple` | Virtual thermocouple at 3 mm | < **40 %** |
| `test_cooldown_energy` | Energy balance during cooling | pass |
| `test_evaporative_cooling` | Sink removes energy, respects onset | pass |

The Stefan benchmark solves the transcendental root $\sqrt{\pi}\lambda e^{\lambda^2}\text{erf}(\lambda) = \text{Ste}$ where $\text{Ste} = c_p\Delta T/L_f$, and compares the simulated front position slope.

> [!cite] Source
> **Rosenthal (1946)**, *The theory of moving sources of heat and its application to metal treatments*. I implemented the analytical solution in `validation/rosenthal.py` purely as a validation target — it is not part of the solver.

---

## Related

- [[04 - The Arc Heat Source]] — where the energy comes in
- [[05 - Free Surface and Wetting]] — freeze, remelt and trailing solidification
- [[02 - The Lattice Boltzmann Core]] — the flow field that advects this enthalpy
