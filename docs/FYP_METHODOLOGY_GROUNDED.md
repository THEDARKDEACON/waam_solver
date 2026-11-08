# FYP Methodology (Revised) — Grounded Soft Sensing and Supervisory Control

**Project:** In-situ monitoring and dynamic control of robotic MIG-based WAAM  
**Cell:** JKUAT robotic WAAM — Rustler EM250i  
**Status:** Revises §6 *General Research Methodology* of `FYP_Concept_Note.pdf` (earlier draft)  
**Scope of this file:** Methodology only (objectives / outcomes / partnerships unchanged unless noted)

---

## 0. What changed from the earlier concept note

| Earlier concept note | Revised methodology |
|----------------------|---------------------|
| Live first-principles melt-pool model runs **in parallel** with the bead and returns W/H/D each segment | High-fidelity twin is **too slow** for real-time inference; it becomes an **offline teacher** |
| Pyrometer melt-pool T must agree with the model before geometry is trusted | Pool peaks exceed available pyrometer range; T-gate is **retired** as the primary validator |
| Soft geometry comes **directly** from the CFD/LBM twin online | Soft geometry comes from a **distilled soft sensor** trained on grounded twin (and later sparse real labels) |
| Park / absolute macro as implicit success | **Tiered grounding:** code↔code, then sim↔shop; Park is context, not the sole gate |
| Single abstract chain: G-code → twin → MIG | Explicit **three abstraction layers** with a grounding gate between each |

The control *philosophy* (G-code geometry intent → supervisory correction of V / WFS / related setpoints) remains. The *sensing stack* and *validation discipline* are rewritten so the project does not hang on a real-time CFD soft sensor or an out-of-range pyrometer.

---

## 1. Why this path is worth exploring

1. **The product goal is still valid.** Holding bead width/height to a G-code intent on a conventional MIG (no industrial fieldbus welder) is the gap the cell still has.
2. **Literature already supports the control pattern** (virtual/soft sensors for bead size; MIMO / MPC / RNN supervisory geometry control). What is thinner — and therefore a useful FYP contribution — is **sim-trained soft sensing** after the full twin cannot run at control rates.
3. **We already own the hard part of the teacher:** a physics twin with Tier A Ansys agreement and an in-progress Fluent Tier B melt-pool compare. That investment is wasted if we coarsen the twin into a mach-limited “real-time” caricature.
4. **Abstraction without grounding would be indefensible.** Because the controller will sit on top of *estimates of estimates*, the method below **forces experimental and cross-code gates** before any layer may supervise the next. If a gate fails, we stop and fix that layer — we do not “hope Park appears.”

**Go / no-go for the FYP:** proceed if Layer A can be grounded (twin↔Fluent on W/D within band, residual documented). Soft-sensor and control work are then justified as engineering on a teacher that has been challenged, not as an untested stack of models.

---

## 2. Three abstraction layers (and why grounding is mandatory)

The revised method uses three layers of increasingly abstract data. Each layer may only feed the next after its gate passes.

```text
Layer A — High-fidelity teacher
  waam_twin (CFD / melt-pool)  +  Ansys Fluent Tier B (referee)
        │  GATE A: code ↔ code (W, D; H if Phase 2)
        ▼
Layer B — Distilled soft sensor / dynamics
  g_φ : (I, V, WFS, v, …) → (Ŵ, Ĥ, D̂)
  optional f_θ : dynamics for short-horizon planning
        │  GATE B: soft sensor ↔ twin labels; then sparse real W/H
        ▼
Layer C — Supervisory control on the physical cell
  y* from G-code  −  ŷ from g_φ  →  saturated Δu (V, WFS, …)
        │  GATE C: open-loop vs supervised beads (metrology)
        ▼
      Rustler EM250i + robot path
```

**Three layers of abstract data (stated plainly):**

| Layer | What it produces | Risk if ungrounded |
|-------|------------------|--------------------|
| **A** | Simulated W/H/D under process inputs | Wrong physics → wrong teacher forever |
| **B** | Fast Ŵ/Ĥ/D̂ from electricals / WFS | Fits the sim’s bias; tracks a fantasy bead |
| **C** | Corrective MIG setpoints | Moves the real process using Layer B’s fantasy |

Therefore: **no Layer B training until Gate A; no unsupervised Layer C until Gate B; no claim of success until Gate C.**

---

## 3. Grounding programme (do this before trusting control)

### Gate A — Teacher credibility (physics)

**A1. Twin ↔ Ansys Tier A (thermal)** — *done for the Park-lock Goldak question*  
- Same Q_net / path intent; fusion envelope W/D within ±15%.  
- Shared miss vs Park width is documented; **η/Goldak not retuned** to fake Park.

**A2. Twin CFD ↔ Fluent Tier B (VOF + melting, Phase 1)** — *in progress*  
- Same Goldak, path, material intent (mild/ER70S-6 family vs Park SS400/ER70S-6 — not SS316L).  
- Phase 1: **no wire**; score **W and D** only.  
- Win: Fluent vs twin CFD within **±15%** on W and D. Park is **informational**.  
- Phase 2 (wire) only after Phase 1 pool exists → then H if crown appears.

**A3. Residual card**  
- Publish a one-page record: twin W/D/H working point, Fluent W/D, Park macro, shop lock untouched.  
- Explicit statement: controller may track **this plant’s** \(y^*\), not “match Park paper.”

**Stop rule:** If Gate A2 fails badly (Fluent and twin disagree), **do not** distill a soft sensor from the twin. Diagnose mesh/UDF/phase setup first.

### Gate B — Soft sensor credibility

**B1. Train \(g_φ\) on Gate-A twin rollouts** over a bounded process box (I/V or power, WFS, \(v\), optional CTWD). Labels = twin (or Fluent-agreed) W/H/D.  
**B2. Hold out** twin cases never seen in training; report MAE / % on W and H.  
**B3. Sparse real calibration** — even a few etched or scanned beads: fit a residual \(y_{\mathrm{real}} \approx y_{\mathrm{soft}} + r(\cdot)\).  
**Stop rule:** If B2 is poor, enlarge data or simplify \(g_φ\); if B3 shows large bias, **do not** close the loop on soft estimates alone.

### Gate C — Control credibility

**C1.** One multi-layer (or multi-segment) part **open-loop** (fixed schedule).  
**C2.** Same part **supervised** (Layer C using \(g_φ\)).  
**C3.** Compare post-weld geometry (section/scan), arc loss, and electrical traces.  
**Success:** supervised deposit stays **closer to G-code \(y^*\)** than open-loop — not “closer to Park.”

---

## 4. Step-by-step methodology (revised §6)

### (a) Job preparation from the part file
- Slice CAD → G-code with a qualified profile; retain existing path transpiler (torch path, travel speed, arc state).  
- Extract **intended bead width \(W^*\)** and **layer height \(H^*\)** from the G-code / process profile.  
- These form the geometry reference \(y^*(s)\) along the path — the quantity the supervisor tracks.

### (b) Nominal parameter derivation
- From \(W^*\), \(H^*\), and wire cross-section, compute a **nominal** schedule (WFS, voltage / related setpoints, travel).  
- This is the open-loop baseline and the centre of the Soft-sensor training box — not yet closed-loop.

### (c) Command of the welding power source
- Establish computer → Rustler EM250i link (serial / panel path as implemented).  
- Issue voltage, inductance, and wire-feed speed; verify against measured arc voltage and current.  
- Record on a common time base: **I, V, WFS**, robot pose / travel, acoustics (stability aid).  
- **Pyrometer:** optional HAZ / cold-region monitor only; **not** used as melt-pool \(T_{\max}\) gate (range insufficient).

### (d) High-fidelity teacher (Layer A) — offline
- Run `waam_twin` jobs over the process box; export W/H/D (and liquidus-based fusion metrics as defined).  
- Complete Fluent Tier B Phase 1 (and Phase 2 if H is required); fill `fluent_metrics.json`; run `compare_tierB.py`.  
- **Do not** coarsen the twin to chase wall-clock RTF for control; mach-limit cascades invalidate labels.

### (e) Soft sensor and optional dynamics (Layer B)
- Train \(g_φ(I,V,\mathrm{WFS},v,\ldots) \rightarrow (\hat W,\hat H,\hat D)\) on Gate-A labels.  
- Optionally train \(f_θ\) for one-step / short-horizon prediction of \(y\).  
- Deploy \(g_φ\) at control rate (order 10–50 Hz is sufficient for segment supervision).  
- Pass Gate B before any automatic correction is enabled.

### (f) Supervisory correction (Layer C)
- Each segment (or low-rate window):  
  \(\hat y = g_φ(u_{\mathrm{meas}})\), \(e = y^* - \hat y\).  
- Apply a **saturated** update, e.g.  
  \(u \leftarrow \mathrm{sat}_{\mathcal{U}}(u + K e)\)  
  with \(\mathcal{U}\) = WPS limits on V, WFS, inductance, and rate limits / deadbands.  
- Prefer WFS ↔ height, travel/power ↔ width as first-order levers; keep the MIG’s inner arc regulation intact.  
- **Gated enable:** corrections off until Gate B passes; corrections pause on arc-loss / acoustic instability flags.

### (g) Experimental comparison (Gate C)
- Deposit the same qualified part twice: fixed schedule vs supervised.  
- Compare bead dimensions (section or scan), electrical logs, and process faults.  
- Report whether supervision improved tracking of **G-code \(y^*\)**.

---

## 5. What we deliberately do *not* do

- Run full LBM/VOF in the PLC / segment loop as the soft sensor.  
- Retune η/Goldak solely to match Park after Tier A showed a shared thermal miss.  
- Promote an ungrounded dx / \(C_{\mathrm{darcy}}\) “fast” twin into the teacher set.  
- Use out-of-range pool pyrometry as a pass/fail for geometry estimates.  
- Claim Park absolute macros as the control success metric.

---

## 6. Mapping to specific objectives (concept note §3)

| Concept-note objective | Revised method |
|------------------------|----------------|
| 1. Control V, inductance, WFS on Rustler; verify vs measured V/I | Unchanged — Layer C actuation + electrical verification |
| 2. Intended W and layer height from Prusa G-code; WFS from them | Unchanged — \(y^*\) and nominal schedule |
| 3. Create/validate melt-pool model vs pyrometer; predict W/H/D for correction | **Split:** (i) validate twin vs Fluent (+ document residual); (ii) distill soft sensor; (iii) correct from \(y^*-\hat y\) with sat — **not** live CFD + pool pyrometer gate |

---

## 7. Minimum evidence pack (FYP defence)

1. Gate A report: Tier A table + Fluent Phase 1 W/D vs twin CFD + residual card.  
2. Gate B: soft-sensor parity plots (train/hold-out) + any real residual fit.  
3. Gate C: open-loop vs supervised geometry table and electrical traces.  
4. Explicit statement of the three layers and which gate unlocked the next.

---

## 8. One-paragraph methodology abstract (for the concept note body)

The study retains the JKUAT robotic WAAM cell and the Rustler EM250i. Intended bead width and layer height are taken from G-code and used as the supervisory reference. A high-fidelity melt-pool twin, challenged against Ansys Tier A thermal fusion and Fluent Tier B VOF melting (and deposition when ready), generates labelled process→geometry data offline; it is not required to run at control rates. A distilled soft sensor maps live current, voltage, wire-feed speed and travel to estimated bead geometry. Saturated updates of welding setpoints then reduce the error between G-code intent and those estimates. Because the stack stacks simulation, learning and control, each layer is gated by cross-code or experimental grounding before it may supervise the next. Success is closer tracking of the intended bead on the physical cell, not reproduction of a published macrograph alone.
