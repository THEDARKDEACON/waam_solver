# Materials and Methods

**Project:** Implementation of In-Situ Monitoring and Dynamic Control of Robotic MIG-Based WAAM  
**Cell:** JKUAT robotic WAAM cell — Rustler EM250i welding power source  
**Purpose of this chapter:** Full proposal *Materials and Methods* (replaces the short §6 methodology of the earlier concept note)

---

## 4.1 Overview of the research design

The work is an experimental and computational study on the existing robotic wire-arc additive manufacturing (WAAM) cell at Jomo Kenyatta University of Agriculture and Technology (JKUAT). The robot path planner, the operator interface, and the monitoring channels already available on the cell are retained. The welding power source for this project is the Rustler EM250i. Earlier software written for a different inverter (AICO 200I) is not reused.

The overall aim is to hold deposited bead width and layer height close to the values intended in the part file, by changing welding setpoints during deposition. The method does **not** attempt to run a full melt-pool fluid simulation inside the control loop. Instead it proceeds in three linked stages:

1. **Offline process model.** A first-principles melt-pool model predicts bead width, height and penetration depth from welding parameters and torch motion. That model is checked against independent numerical results and, where available, against measured beads, before its predictions are trusted as training data.
2. **Fast geometry estimator (soft sensor).** Because the detailed model is too slow to run while the arc is on, a compact estimator is trained from the validated model (and later refined with a small number of real beads). In operation it maps live electrical and motion measurements to estimated bead geometry at a rate suitable for control.
3. **Supervisory correction on the cell.** The intended bead geometry is taken from the G-code. The estimator’s output is compared with that intent. Bounded corrections are then applied to wire-feed speed, voltage and related setpoints on the Rustler EM250i for the following weld segment.

Each stage is only allowed to feed the next after a clear check has been passed. If a check fails, that stage is corrected before closed-loop deposition is claimed. Success is defined as a supervised deposit that stays closer to the bead specified in the G-code than an otherwise identical open-loop deposit — not as reproduction of a published coupon from another laboratory.

---

## 4.2 Materials, equipment and software

### 4.2.1 Deposition cell

| Item | Role in this study |
|------|--------------------|
| Industrial robot (KUKA cell already installed at JKUAT) | Follows the planned torch path; pose and travel speed are recorded |
| Rustler EM250i MIG inverter | Supplies the arc; voltage, inductance and wire-feed speed are commanded from the cell computer |
| Filler wire and shielding gas as used on the qualified cell process | Feedstock for bead-on-plate and multi-layer trials |
| Substrate plate consistent with the qualified mild-steel process window | Build plate for calibration and comparison prints |

Exact wire grade, plate grade, gas composition and flow will be fixed in the process profile used for slicing and will be stated in the experimental log for every trial.

### 4.2.2 Sensing and data acquisition

The following quantities will be recorded on a **common time base** during deposition:

- Arc voltage and welding current (measured, not only commanded)
- Wire-feed related signals as instrumented on the cell
- Robot pose / travel speed
- Acoustic emissions from the process (stability aid; not the primary geometry sensor)
- Temperature where the available pyrometer or thermocouples can read reliably (for example heat-affected or cooler regions)

**Note on melt-pool temperature.** Peak melt-pool temperatures on this process commonly exceed the usable range of the cell’s existing pyrometer. Melt-pool temperature agreement will therefore **not** be used as the main pass/fail test before a geometry estimate may influence the weld. Temperature channels remain useful for documentation and for monitoring cooler regions of the build.

### 4.2.3 Computing and software

- Existing CAD → slicer → Prusa-style G-code workflow already used on the cell
- Existing path transpiler that extracts torch path, travel speed and arc on/off from G-code
- Cell computer software for serial communication with the Rustler EM250i control panel path
- First-principles melt-pool simulation software developed for this cell (`waam_twin`), run offline
- An independent commercial or established numerical solver used as a cross-check of the melt-pool model (same process intent, comparable geometry metrics)
- Training and evaluation scripts for the soft sensor and for logging supervisory corrections

---

## 4.3 Specific objectives addressed by this method

The main objective remains: to implement in-situ process observation and closed-loop control of robotic MIG-based WAAM on the Rustler EM250i, so that welding parameters can be changed during deposition using live measurements and a process model that has been validated offline.

**Specific objectives (tuned for this proposal):**

1. To implement control of welding voltage, inductance and wire-feed speed on the Rustler EM250i, and to verify each commanded value against measured arc voltage and current.
2. To extract intended bead width and layer height from the existing Prusa G-code (and process profile), and to derive a nominal wire-feed speed from those geometric targets and the wire cross-section.
3. To develop and validate an offline first-principles melt-pool model that predicts bead width, height and penetration depth; to distill from it a fast soft sensor that estimates the same quantities from live process measurements; and to use the soft-sensor estimates as supervisory feedback for corrective signals during deposition.

Objective 3 is deliberately split into **validate offline → distill a fast estimator → correct on the cell**, so that the proposal does not depend on running the full melt-pool model in real time or on out-of-range pool pyrometry.

---

## 4.4 Methodological framework

The method follows the part file through to a verified correction on a later weld segment. Figure 4.1 summarises the information flow.

```text
Part file (CAD)
      │
      ▼
G-code + process profile ──► intended bead width W* and layer height H*
      │                      + nominal welding schedule
      ▼
Robot path + Rustler EM250i setpoints
      │
      ├── live measurements: current, voltage, WFS, travel, acoustics
      │
      ▼
Offline melt-pool model ──(after validation)──► labelled process → geometry data
      │
      ▼
Soft sensor (fast estimator) ──► estimated Ŵ, Ĥ, D̂ during the print
      │
      ▼
Compare with W*, H* ──► bounded corrections to next segment (V, WFS, …)
      │
      ▼
Open-loop vs supervised print of the same part (metrology)
```

**Working rule.** Predictions from the detailed model are not used to train the soft sensor until the model has been checked. Soft-sensor estimates are not used to change the weld automatically until the estimator has been checked on held-out cases (and, where possible, on a few real beads). Claims of improved deposition wait for the open-loop versus supervised comparison.

---

## 4.5 Preparation of the job from the part file

1. A computer-aided design model of the trial part is sliced to G-code using a **qualified process profile** already accepted on the cell (travel speed, layer settings, and related print parameters).
2. The existing transpiler continues to obtain, from that file:
   - torch path,
   - programmed travel speed,
   - arc on / arc off state along the path.
3. In addition, **intended bead width** \(W^*\) and **intended layer height** \(H^*\) are read from the G-code and/or the same process profile.
4. Along the path these two quantities form the geometry reference \(y^* = (W^*, H^*)\) that the supervisor will track.

No new CAM stack is introduced. The addition is only that bead intent becomes an explicit, machine-readable target rather than an informal expectation of the operator.

---

## 4.6 Derivation of nominal welding parameters

For each weld move (or for each segment of a longer bead):

1. The cross-sectional area of the filler wire is taken from the wire diameter in the process sheet.
2. A **nominal** wire-feed speed is calculated so that the volume of wire delivered per unit length is consistent with the intended bead cross-section formed by \(W^*\) and \(H^*\) (subject to the usual process assumptions of the cell’s qualified schedule).
3. Nominal voltage and inductance are taken from the qualified schedule for that wire-feed / travel combination, or from a characterisation already performed on the Rustler EM250i.
4. This nominal schedule is the **open-loop baseline**. It is also the centre of the parameter region later explored when generating training data for the soft sensor.

At this stage no closed-loop correction is applied; the aim is a repeatable starting point that matches the part file’s geometric intent as far as a fixed schedule can.

---

## 4.7 Command of the welding power source and in-situ measurement

### 4.7.1 Computer-to-welder link

A communication link will be established between the cell computer and the Rustler EM250i, beginning from the serial connection between the inverter and its control panel (as characterised for this machine). Through this link the computer will issue:

- welding voltage,
- inductance,
- wire-feed speed.

Commanded values will be checked against **measured** arc voltage and current so that actuation is verified, not only assumed.

### 4.7.2 Synchronised logging

During every trial the following will be logged with a shared time stamp:

- commanded and measured electrical quantities,
- wire-feed related measurements,
- robot position and travel speed,
- acoustic features used as a process-stability flag,
- any temperature channels within instrument range.

Loss of arc, abnormal acoustic behaviour, or failure of the communication link will be treated as conditions that **disable** automatic parameter updates until the process is stable again.

---

## 4.8 Offline melt-pool process model

### 4.8.1 Role of the model

A first-principles melt-pool model will be used to relate process inputs (current, voltage or heat input, wire-feed speed, travel speed, and torch path) to predicted:

- bead width,
- bead height (crown),
- penetration depth.

The model runs **offline**: it prepares labelled examples and supports design of the soft sensor. It is not required to finish a full simulation of each segment before the robot reaches the next waypoint.

### 4.8.2 Validation of the model (plain-language checks)

Before any model output is used as training truth for the soft sensor, the following checks will be applied:

1. **Internal consistency.** For a fixed process window, predicted trends must be physically sensible (for example, higher heat input tending to widen the fusion zone within the explored range; wire feed contributing to deposited height).
2. **Independent numerical cross-check.** For selected cases, the same process intent (path, heat source description, material family) will be run in an independent established solver. Predicted fusion width and depth will be compared. Large unexplained disagreement stops the method until the cause is found.
3. **Sparse experimental anchors.** Where sectioned or scanned beads from the cell (or from a documented reference coupon in the same process family) are available, predicted geometry will be compared with measured geometry. Absolute agreement with every published coupon is **not** required for the controller to be useful; what is required is that residual error is known and stated.

Only after these checks is the model allowed to generate the bulk of the soft-sensor training set.

### 4.8.3 What is deliberately avoided

- Coarsening the physics model solely to chase wall-clock speed for “live” CFD inside the control loop, in a way that would invalidate the labels.
- Using an out-of-range melt-pool pyrometer reading as the sole gate that unlocks geometry feedback.
- Retuning the model only to match a single external macrograph while ignoring disagreement with an independent numerical check.

---

## 4.9 Soft sensor for real-time geometry estimation

### 4.9.1 Purpose

The soft sensor is a compact mapping

\[
g:\ (I,\ V,\ \mathrm{WFS},\ v,\ \ldots)\ \longrightarrow\ (\hat{W},\ \hat{H},\ \hat{D})
\]

from quantities that can be measured while welding to estimates of bead width, height and penetration. It exists because the detailed melt-pool model cannot keep up with the process at control rates, yet the supervisor still needs a geometry estimate between segments.

### 4.9.2 Training and checking

1. Generate a set of offline simulations that cover a **bounded** box of process parameters around the nominal schedule (current/voltage or power, wire-feed speed, travel speed, and related quantities such as contact-tip-to-work distance if instrumented).
2. Train \(g\) so that its outputs match the validated model’s bead width, height and depth on those cases.
3. Hold out cases that were never used in training; report estimation error on width and height.
4. Where a small number of real beads can be measured (etch, scan, or calliper/section as available), fit a simple residual correction so that estimates move toward the physical cell, not only toward the simulation.

If hold-out error is large, the estimator is simplified or the training set is enlarged before any automatic correction is enabled. If real beads show a large systematic bias and no residual correction is available, closed-loop use of the soft sensor alone is deferred.

### 4.9.3 Use during deposition

During a print, the soft sensor is evaluated at a modest rate (on the order of tens of times per second is sufficient for segment-level supervision). Its outputs \(\hat{W}\) and \(\hat{H}\) are the geometry estimates compared with \(W^*\) and \(H^*\) from the G-code.

---

## 4.10 Supervisory closed-loop correction

### 4.10.1 Error and update

For each weld segment (or low-rate window along a long bead):

1. Form the geometry error  
   \(e = y^* - \hat{y}\),  
   where \(y^* = (W^*, H^*)\) and \(\hat{y} = (\hat{W}, \hat{H})\).
2. Apply a **bounded** update to the welding setpoints, for example  
   \(u \leftarrow \mathrm{sat}(u + K\, e)\),  
   where \(u\) may include wire-feed speed, voltage and inductance, \(\mathrm{sat}(\cdot)\) enforces machine limits and rate limits, and \(K\) is a simple gain or gain matrix chosen from the process characterisation.
3. Prefer, as a first-order guide:
   - wire-feed speed for height tracking,
   - travel speed and/or heat-related setpoints for width tracking,  
   while leaving the inverter’s inner arc regulation intact.

Corrections act on the **following** segment (or the next admissible window), not as high-bandwidth force control of the robot path. Path planning remains with the existing robot controller.

### 4.10.2 Safety and enable logic

Automatic correction is enabled only when:

- the soft sensor has passed its hold-out (and residual) checks,
- the computer–welder link is healthy,
- the arc is stable (no arc-loss flag; acoustic instability within agreed bounds),
- commanded setpoints remain inside the qualified process envelope.

If any of these conditions fail, the cell reverts to the last safe nominal schedule or pauses deposition according to the cell’s existing safety practice.

---

## 4.11 Experimental comparison and evaluation

### 4.11.1 Design

One qualified multi-layer (or multi-segment) part will be deposited **twice** under otherwise identical conditions:

| Trial | Welding schedule |
|-------|------------------|
| Open-loop | Fixed nominal parameters from §4.6 |
| Supervised | Same start schedule, with §4.10 corrections enabled |

### 4.11.2 Measurements

The two deposits will be compared using:

- bead width and layer height measured after welding (section, scan, or other metrology available on site),
- electrical traces (current, voltage, wire-feed) and any recorded arc-loss events,
- qualitative process notes (spatter, visible humping, overflow).

### 4.11.3 Success criterion

The supervised deposit is judged successful if it remains **closer to the intended bead width and layer height from the G-code** than the open-loop deposit, without a worse rate of arc loss or gross surface defects. Matching an external laboratory’s published bead dimensions is not the success metric.

---

## 4.12 Scope and limitations (methodological)

For proposal clarity, the following are **in scope**:

- supervisory correction of MIG setpoints on a conventional inverter without industrial fieldbus,
- use of an offline physics model and a distilled soft sensor for geometry feedback,
- comparison of open-loop and supervised builds on the JKUAT cell.

The following are **out of scope** for this project:

- replacing the robot motion planner with melt-pool-level path replanning,
- a full materials-science study of microstructure or mechanical properties,
- claiming hard real-time computational fluid dynamics inside the welding loop,
- relying on melt-pool pyrometry beyond the instrument’s usable range as a geometry validator.

---

## 4.13 One-paragraph abstract of the method

The study is conducted on the existing JKUAT robotic WAAM cell with the Rustler EM250i as the welding power source. Intended bead width and layer height are taken from the part’s G-code and used as the supervisory reference. A first-principles melt-pool model is run offline, checked against independent numerical results and sparse bead measurements, and then used to train a fast soft sensor that estimates bead geometry from live current, voltage, wire-feed speed and travel. Bounded updates of welding setpoints reduce the error between the G-code intent and those estimates. Because simulation, estimation and control are stacked, each stage is checked before it may drive the next. The decisive experiment deposits the same part once with fixed parameters and once under supervision, and asks which deposit stays closer to the intended bead.

---

## Mapping from the earlier concept note

| Earlier concept-note method step | This chapter |
|----------------------------------|--------------|
| (a) Job from part file | §4.5 |
| (b) Parameter derivation | §4.6 |
| (c) Command of power source + sensing | §4.7 |
| (d) Segment evaluation by process model (implied live) | §4.8 offline + §4.9 soft sensor |
| (e) Pyrometer temperature gate then geometry correction | Temperature gate retired; §4.8–4.10 |
| (f) Open-loop vs supervised comparison | §4.11 |
| Specific objective 3 (live model + pyrometer) | Rewritten in §4.3 |

Internal engineering notes (solver tiers, residual cards, repository validation gates) live in the project technical documentation and are **not** required wording for this proposal chapter.
