# Live Classical Digital Twin — Integration Plan

**Status:** Implementation plan (how to get there from what we already have)  
**Audience:** Project team building a supervisory WAAM digital twin  
**Companion docs:**
- [`WAAM_DT_OBJECTIVE_ARCHITECTURE.md`](WAAM_DT_OBJECTIVE_ARCHITECTURE.md) — target architecture (FigJam-aligned)
- [`WAAM_PARALLEL_PREDICTIVE_ARCHITECTURE.md`](WAAM_PARALLEL_PREDICTIVE_ARCHITECTURE.md) — predictive / defect framing
- [`WAAM_TWIN_V2_EXECUTION_PLAN.md`](WAAM_TWIN_V2_EXECUTION_PLAN.md) — physics engine workstreams

**Repos in scope (already on disk):**

| Repo / folder | Role today | Role in the live twin |
|---------------|------------|------------------------|
| `FYP22-01/` (+ `waam_twin/`) | Robot HMI, VarProxy, Socket.IO joint stream, melt-pool physics | Pose / torch_state publisher + **physics heart** |
| `WAAM_Sensor_Integration/` | Arduino → serial → Pi WebSocket `:9900` | Electrical / thermal **sensor broadcast** |
| `Acoustic Analysis/` | Live mic → RMS/kurtosis/STFT, CSV/NPZ, web spectrogram | Process-stability / **defect-risk aid** (parallel channel) |

This document answers: *how do we make `waam_twin` operate both as an offline job-YAML simulator and as a classical digital twin that consumes live streams?* It also lists the problems you will hit.

---

## 1. Goal (classical digital twin, our definition)

A classical digital twin for WAAM here means:

1. **Subscribe** to the same live broadcasts the cell already publishes (joints / TCP, sensors, optionally acoustics).
2. **Time-align** those streams into a single process packet `U(t)`.
3. **Advance** a physics (or hybrid) model on that packet fast enough to stay useful vs wall clock (**soft real-time** is acceptable).
4. **Infer risk** — melt-pool / bead metrics + acoustic / electrical anomalies → “defect likely in this window?”
5. **Optionally feedback** — bounded supervisory actions only (ΔI, ΔWFS, pause), never replace KRC4 motion planning.

It does **not** mean: hard closed-loop force control at Lattice-Boltzmann `dt` (~tens of µs). That is the wrong control layer for this engine.

```text
Physical cell                    Virtual twin
─────────────                    ────────────
KUKA + PSU + sensors  ──broadcasts──►  Connector → Physics → Risk → Feedback
        ▲                                                          │
        └──────────────── bounded WPS / KR commands ───────────────┘
```

---

## 2. What already exists (do not rebuild)

### 2.1 `FYP22-01` — robot + twin core

| Asset | Path / interface | Notes |
|-------|------------------|-------|
| VarProxy client | `kuka.py` → TCP `172.31.1.147:7000` | `$AXIS_ACT`, `$POS_ACT`, `$OUT[1]` |
| Joint / path broadcast | `main.py` → Socket.IO `:4900` | Events: `joint_angles`, `cartesian_coords`, `torch_state` (~50 ms) |
| Offline physics | `waam_twin/` job YAML → `WAAMTwin.from_job` / `run_path` | Goldak, forces, VOF, telemetry |
| Live TCP step (thin) | `waam_twin/kuka_adapter.py` → `step_from_tcp()` | Used from MockKUKA / `kuka.py` when env twin enabled |
| Telemetry API | `WAAMTwin.get_telemetry()` | Pool W/D, T_peak, bead height, forces, porosity proxy, CTWD, … |
| Frame map | `waam_twin/frame.py` + `WAAM_FRAME` | Robot mm → sim metres |

**Gap:** Live **I / V / WFS** are not yet injected into `twin.step(...)`. Pose can be live; process sheet is still mostly job YAML.

### 2.2 `WAAM_Sensor_Integration` — electrical / thermal broadcast

| Asset | Path / interface | Notes |
|-------|------------------|-------|
| Arduino front-end | `Sensors_Arduino/`, `thermocouplesNano/` | V, I, WFS-related counts, TCs |
| Serial → WebSocket | `Serial_data_to_UI.py` | Bind `0.0.0.0:9900`, JSON dict per line, ~200 ms sleep |
| OCR / UI helpers | `src/ocr/`, figures | Optional; not required for twin ingest |

**Typical payload fields** (parsed key:value lines): `Time(S)`, `V`, `C`/`I`, thermocouple channels (`T1`…), encoder/WFS counts as instrumented. Exact keys depend on the Arduino sketch — the connector must **normalize** them.

**Gap:** No shared schema version; relative `Time(S)` not wall-clock NTP; twin is not a subscriber yet.

### 2.3 `Acoustic Analysis` — stability / defect channel

| Asset | Path / interface | Notes |
|-------|------------------|-------|
| Feature core | `acoustic_core.py` | WAV load, PSD→dB, RMS, kurtosis, optional AE burst metrics |
| Live dashboard | `acoustic_dashboard.py` | Mic capture, live spectrogram, CSV feature log, `acoustic_live_3d_stream.npz` |
| Web view | `web_acoustic_server.py` | Flask playback / live NPZ snapshot |
| Research context | PDFs + `Paper_Synopsis.md` | RMS/kurtosis ↔ transfer stability; MFCC/ML ↔ geometric defects |

**Gap:** Acoustics are a **parallel monitor**, not fused into `U(t)` or twin risk logic. No WebSocket publisher for features yet (file/NPZ/CSV based). Literature shows acoustics excel at **instability / porosity / transfer anomalies**, while the physics twin excels at **thermal geometry (W/H/D)**. Fuse them; do not expect acoustics alone to replace melt-pool CFD.

### 2.4 Dual-mode vision (target product behaviour)

| Mode | Driver | Use |
|------|--------|-----|
| **A — Offline / job** | `torch_path` + process block in YAML | Calibration, held-outs, HPC batches, viewer demos |
| **B — Live twin** | Connector `U(t)` + optional YAML material/geometry | Cell shadowing, advisory risk, soft-RT prediction |

Same `WAAMTwin` instance API; different **input adapter**. Mode A already works. Mode B is mostly glue + sync + risk layer.

---

## 3. Target architecture (concrete)

```mermaid
flowchart LR
  subgraph Physical
    KRC[KRC4 VarProxy :7000]
    ARD[Arduino sensors]
    MIC[Microphone]
    PI[Pi serial→WS :9900]
  end

  subgraph Publishers
    MAIN[FYP22-01 main.py<br/>Socket.IO :4900]
    ACO[Acoustic feature publisher<br/>proposed :10000]
  end

  subgraph TwinHost
    CONN[Connector<br/>align + U(t)]
    PHYS[waam_twin physics<br/>soft-RT window]
    RISK[Risk fusion<br/>physics + acoustic + electrical]
    FB[Feedback supervisor<br/>WPS / KR bounds]
  end

  KRC --> MAIN
  ARD --> PI
  MAIN -->|cartesian_coords, torch_state| CONN
  PI -->|I,V,WFS,TC| CONN
  MIC --> ACO
  ACO -->|RMS, kurtosis, band energy| RISK
  CONN --> PHYS
  PHYS -->|get_telemetry S(t)| RISK
  RISK --> FB
  FB -.->|bounded Δ / pause| Physical
```

### 3.1 Live process packet `U(t)` (canonical)

```json
{
  "t_wall_s": 1710000000.123,
  "t_rel_s": 12.45,
  "pose_mm": {"x": 420.1, "y": -10.2, "z": 185.0},
  "pose_sim_m": {"x": 0.020, "y": 0.020, "z": 0.012},
  "speed_mm_s": 10.0,
  "is_welding": true,
  "current_A": 100.0,
  "voltage_V": 15.0,
  "wfs_m_min": 3.5,
  "ctwd_mm": 15.0,
  "T_tc_C": [45.0, 48.0, null, null],
  "material_id": "ER70S-6",
  "packet_age_ms": 35,
  "skew_pose_elec_ms": 18,
  "valid": true
}
```

Acoustics ride a **sidecar** packet `A(t)` (higher rate, not forced into every LBM step):

```json
{
  "t_wall_s": 1710000000.123,
  "rms": 0.042,
  "kurtosis": 4.1,
  "band_energy_rel": {"arc_1_5k": 0.62},
  "anomaly_score": 0.31
}
```

### 3.2 Synchronisation budgets (from objective architecture)

| Rule | Target | If violated |
|------|--------|-------------|
| Pose update | ~50 ms (today’s Socket.IO) | Extrapolate last pose; flag stale |
| I / V / WFS | ~50–200 ms (serial + WS sleep) | Hold last; widen uncertainty |
| Pose ↔ electrical skew | ≤ 20–50 ms after align | Warn; **do not auto-command** |
| Packet age before command | ≤ 100–200 ms | Advisory only |
| Physics lag vs wall clock | Soft-RT OK | Feedback uses **latest finished** `S(t)` |

---

## 4. How to go about it (phased plan)

### Phase 0 — Freeze interfaces (1–2 days)

1. **Document schemas** for sensor JSON keys (map `C`→`current_A`, etc.) and Socket.IO event payloads.
2. Add a versioned `schemas/live_packet_v1.json` in `waam_twin/` (or shared `WAAM/schemas/`).
3. Agree clock policy: prefer **host NTP wall clock** on twin PC; Pi and audio PC sync via NTP; relative `Time(S)` is secondary.
4. Confirm network: twin host can reach `ws://<Pi>:9900` and `http://<FYP-host>:4900`.

**Exit:** Sample captures of both streams logged to disk with wall timestamps.

### Phase 1 — Connector MVP (offline replay first)

Build `waam_twin/live/connector.py` (name flexible) that:

1. Subscribes to Socket.IO `:4900` and WebSocket `:9900`.
2. Buffers last N messages with `t_wall`.
3. Emits `U(t)` at a fixed rate (e.g. 20 Hz).
4. Writes a **session bag**: `runs/live/<session_id>/u_packets.jsonl` + raw streams.

Then add **replay mode**: feed recorded bags into the twin without the robot. This is mandatory for debugging sync and physics without occupying the cell.

**Exit:** Replay a short bead; twin torch tracks recorded TCP; printed `U(t)` looks sane.

### Phase 2 — Live pose → physics (extend existing bridge)

Today:

```python
# kuka_adapter.step_from_tcp — pose only
twin.step(x, y, is_welding=is_welding, torch_z_m=z)
```

Extend to:

```python
def step_from_live(twin, U: dict) -> dict:
    x, y, z = U["pose_sim_m"]["x"], U["pose_sim_m"]["y"], U["pose_sim_m"]["z"]
    apply_process_overrides(twin, current_A=U["current_A"], voltage_V=U["voltage_V"],
                            wfs_m_min=U["wfs_m_min"])  # new helper
    twin.step(x, y, is_welding=U["is_welding"], torch_z_m=z if twin.use_torch_z else None)
    return twin.get_telemetry()
```

Implementation notes:

- Add `apply_process_overrides` that updates heat-source power (`η·I·V`), mass deposition rate from WFS, and CTWD if measured — **without** rewriting the whole job each step.
- Keep material / domain / Goldak shape from the job YAML (Mode A geometry setup).
- Windowing: for soft-RT, run physics on a **torch-centred subdomain** or coarser `dx` for live Mode B; keep fine jobs for offline validation.

**Exit:** Live or replayed TCP + live I/V move the pool; telemetry W/D and T_peak respond to current steps.

### Phase 3 — Risk layer (advisory digital twin)

Do **not** start with closed-loop. Ship an operator-facing risk panel:

| Source | Indicators | Suggested use |
|--------|------------|---------------|
| Physics `S(t)` | Pool W/D vs target bands, remelt proxy, T_peak, porosity_pct, force Mach-cap warnings | Geometry / fusion risk |
| Electrical | I/V outside WPS, sudden dropouts, CTWD drift | Energy / stick-out risk |
| Acoustic `A(t)` | RMS/kurtosis spikes, arc-band energy shifts, anomaly score | Transfer instability / porosity precursor |

Fusion rule (v1, transparent):

```text
risk = max(
  physics_band_violation,
  electrical_wps_violation,
  acoustic_anomaly > thr
)
action_policy = ADVISORY   # Phase 3: UI alert only
```

Log every decision with `U`, `S`, `A` for post-build vs macrographs / scans.

**Exit:** During a known bad run (spatter / unstable transfer), acoustic + electrical flags fire; during a good calibrate bead, physics stays in band.

### Phase 4 — Acoustic publisher + fusion

1. In `Acoustic Analysis/`, add a thin publisher (WebSocket or Socket.IO) that emits `A(t)` at ~10–50 Hz from the same feature path as `acoustic_dashboard.py` / `acoustic_core.compute_frame_features`.
2. Twin connector subscribes; risk layer consumes without blocking LBM.
3. Optional: train a small online anomaly model on **good-only** lab welds (Lopez-style unsupervised idea) using your existing good/bad CSV/NPZ sets under `Acoustic Analysis/`.

**Exit:** Twin HUD shows RMS/kurtosis next to pool W/D; session logs contain both.

### Phase 5 — Supervisory feedback (Controller B)

Only after Phase 3 false-positive rate is acceptable:

| Channel | Allowed actions | Hard limits |
|---------|-----------------|-------------|
| WPS control | Bounded ΔI, ΔWFS, Δtravel request | Stay inside WPS; rate-limit; deadband |
| KR control | Pause / hold / resume via existing VarProxy / HMI flags | No trajectory rewrite inside twin |
| Safety | Pi thermocouple interlayer gates remain **authoritative** | Twin never bypasses |

If `valid=false` or skew/age violated → **warn only**.

**Exit:** Dry-run on MockKUKA + recorded sensors; then one supervised live trial with e-stop authority on the operator.

### Phase 6 — Operational hardening

- Health checks: stream heartbeats, GPU OOM guard, twin restart without killing HMI.
- Dual mode switch in UI: Offline job / Live twin / Replay bag.
- CI: connector unit tests with recorded fixtures; no robot required.
- Benchmark: report **real-time factor** = simulated process time / wall time for the live preset.

---

## 5. Software layout (proposed)

Keep packages separate; add a thin integration layer so nothing circular-imports.

```text
FYP22-01/waam_twin/
  live/
    connector.py          # subscribe, align, U(t)
    process_overrides.py  # I/V/WFS → twin BC
    risk.py               # fuse S(t)+A(t)+electrical
    replay.py             # bag → U(t)
    schemas/live_packet_v1.json
  kuka_adapter.py         # keep; call from live/

WAAM_Sensor_Integration/
  (unchanged publishers; optional schema doc + NTP note)

Acoustic Analysis/
  acoustic_feature_publisher.py   # NEW: emit A(t)
  acoustic_core.py                # reuse
```

Orchestration options:

- **In-process** on the twin GPU host: connector threads + Taichi main loop (simplest for FYP).
- **Multi-process**: connector writes shared memory / ZMQ; physics process reads (cleaner isolation).

Prefer in-process for the first working demo.

---

## 6. What “fast enough” means (honest compute)

| Quantity | Typical in this repo | Implication |
|----------|----------------------|-------------|
| `dt` | `0.1 · dx` (e.g. dx=0.4 mm → dt≈40 µs) | ~25k LBM steps per second of *process* time |
| Joint stream | ~20 Hz | You do **not** need one LBM step per Socket.IO tick; you need many LBM steps per process ms |
| Soft-RT | Architecture already allows lag | Feedback uses latest finished prediction |
| Full multilayer wall @ fine dx | Often RTF ≪ 1 | Use live **window** / coarser preset for Mode B |

Practical live strategy:

1. **Shadow window:** simulate only a box around the torch (or a short recent path length).
2. **Two-tier:** fast reduced model (thermal / empirical bead) for 10–20 Hz advisory; full LBM periodically or on demand.
3. **GPU host dedicated** to twin; HMI and acoustics can live on other machines.

Defect prediction at “that instant” should mean: *over the last 50–500 ms of process time*, not a single LBM collision.

---

## 7. Problems you will encounter

### 7.1 Time synchronisation

- Pi `Time(S)` is relative from process start; audio timestamps may be another clock; Socket.IO arrivals are host-local.
- **Skew** between pose and I/V of 100+ ms will invent false “hot spots” or miss arc-off.
- **Mitigation:** NTP everywhere; stamp on twin host arrival; enforce skew budget; replay bags with injected delay for tests.

### 7.2 Schema and unit drift

- Sensor keys (`C` vs `I`, counts vs m/min), voltage dividers, CTWD not measured.
- Acoustic levels depend on mic gain / distance; thresholds are not portable without calibration.
- **Mitigation:** versioned schema + calibration YAML per cell; never hardcode magic factors in three repos.

### 7.3 Soft-RT vs operator expectation

- Operators hear “digital twin” and expect 1:1 wall-clock + instant defect labels.
- Full physics will lag; acoustic flags may lead physics by hundreds of ms (good) or false-alarm on fan noise (bad).
- **Mitigation:** UI shows RTF, packet age, and “advisory / not controlling”; separate acoustic alerts from geometry predictions.

### 7.4 Frame / path registration

- Already burned once: path outside plate; `z_mm` TCP vs layer-height convention.
- Live `$POS_ACT` must map through `WeldFrame` correctly; CTWD vs bead top must stay consistent with `use_torch_z`.
- **Mitigation:** live HUD overlays torch on plate AABB; fail loud if pose leaves workpiece AABB.

### 7.5 Process override coupling

- Changing I/V every 50 ms without smoothing can numerically shock Goldak / VOF.
- WFS → mass source must match droplet / continuous deposition mode already in the twin.
- **Mitigation:** low-pass process overrides (e.g. 2–5 Hz); rate limits; freeze material properties.

### 7.6 Acoustic environment

- Shop noise, cooling fans, robot motion, contact tip changes dominate RMS.
- Literature needs high SR / careful mic placement; consumer mics miss >20 kHz AE content.
- **Mitigation:** treat acoustics as **relative** anomaly vs a good-run baseline for *this* cell; do not claim absolute defect class without labelled data.

### 7.7 False defect claims

- Physics porosity tracers and pool W/D are **proxies**, not NDT.
- Absolute W/D gates still need macrographs for credibility (existing validation doctrine).
- Closing the loop on a wrong proxy can **create** defects.
- **Mitigation:** Phase 3 advisory-only until labelled sessions exist; keep WPS hard bounds.

### 7.8 Network / ops fragility

- Hardcoded IPs (`192.168.0.100`, `172.31.1.147`), `debug=True` reloader on `:4900`, Arduino USB path by-id.
- Twin GPU driver resets kill the session mid-weld.
- **Mitigation:** env-config for endpoints; watchdog + safe state (advisory off) on disconnect; no auto-command if any stream dead.

### 7.9 Safety and authority

- Twin must not override Pi thermocouple interlayer stops or e-stop.
- Writing weld params to a real PSU without a WPS governor is a lab hazard.
- **Mitigation:** feedback outputs go through a **command governor** with allow-list and human confirm for Phase 5 first trials.

### 7.10 Software integration debt

- Three repos, three UIs, no shared package today.
- Risk of copying feature code instead of publishing `A(t)`.
- **Mitigation:** publisher/subscriber contracts only; keep physics inside `waam_twin`; keep DSP inside `Acoustic Analysis`.

### 7.11 Validation gap for live mode

- Offline jobs have gates (`bead_calibrate`, PIONEER wall). Live mode needs a parallel gate: *recorded bag → twin metrics within X% of offline job with same path*.
- **Mitigation:** “live parity” test — bag built from a job’s synthetic `U(t)` must reproduce offline telemetry within tolerance.

---

## 8. Dual-mode product behaviour (explicit)

```text
                    ┌─────────────────────┐
                    │   WAAMTwin engine   │
                    │  (Taichi LBM+VOF)   │
                    └─────────┬───────────┘
                              │
           ┌──────────────────┼──────────────────┐
           ▼                                     ▼
   Mode A: JobDriver                      Mode B: LiveDriver
   torch_path / CSV                       connector U(t)
   process from YAML                      process overrides from sensors
   n_steps / path length                  soft-RT until stream ends
           │                                     │
           └──────────► get_telemetry() ◄────────┘
                              │
                              ▼
                     Risk / export / viewer
```

CLI sketch:

```bash
# Offline (existing)
python -m waam_twin.viewer --job jobs/examples/wall_pioneer_m1.yaml

# Live advisory
python -m waam_twin.live.run --job jobs/examples/bead_calibrate.yaml \
  --joints http://127.0.0.1:4900 \
  --sensors ws://192.168.0.100:9900 \
  --acoustic ws://127.0.0.1:10000 \
  --mode advisory

# Replay
python -m waam_twin.live.run --bag runs/live/session_001 --mode advisory
```

---

## 9. Minimal first demo (recommend this order)

1. Record 30 s of `:4900` + `:9900` during a bead-on-plate (even if twin offline).
2. Replay bag → `step_from_live` with process overrides → plot T_peak and pool W vs time.
3. Add acoustic CSV aligned by wall clock → show RMS next to T_peak.
4. Only then: live subscribe without robot motion change (shadow mode).
5. Only then: advisory UI alerts.
6. Only much later: bounded ΔWFS / pause.

This sequence fails fast on sync/schema issues before touching the torch.

---

## 10. Success criteria

| Level | Criterion |
|-------|-----------|
| L1 Connector | Pose + I/V packets aligned; skew metrics logged |
| L2 Shadow | Live/replay TCP drives twin; RTF and telemetry visible |
| L3 Advisory | Risk flags correlate with at least one known bad session and stay quiet on a good calibrate bead |
| L4 Parity | Bag-from-job reproduces offline gate metrics within agreed % |
| L5 Supervisory | Bounded pause or Δ process under human supervision, WPS never violated |

---

## 11. Bottom line

You already have the **three pillars** of a classical WAAM digital twin:

- **Motion broadcast** (`FYP22-01` Socket.IO / VarProxy)
- **Process sensing** (`WAAM_Sensor_Integration` WebSocket)
- **Physics twin** (`waam_twin`) plus a **stability channel** (`Acoustic Analysis`)

What is missing is not another simulator — it is the **connector, process overrides, soft-RT live driver, risk fusion, and (later) a WPS-bounded feedback governor**.

Treat Mode A (job YAML) as the scientific / calibration spine. Treat Mode B (live streams) as an advisory twin first. Expect the hardest problems to be **time sync, schema/units, soft-RT expectations, and false defect confidence** — not the LBM kernels you already built.

---

## 12. Related file map (quick index)

| Concern | Where to look |
|---------|----------------|
| Objective DT architecture | `waam_twin/docs/WAAM_DT_OBJECTIVE_ARCHITECTURE.md` |
| TCP → sim step | `waam_twin/kuka_adapter.py`, `FYP22-01/kuka.py` |
| Joint publisher | `FYP22-01/main.py` (`stream_joint_states`) |
| Sensor publisher | `WAAM_Sensor_Integration/Serial_data_to_UI.py` |
| Telemetry fields | `waam_twin/twin.py` (`get_telemetry`) |
| Acoustic features | `Acoustic Analysis/acoustic_core.py`, `acoustic_dashboard.py` |
| Offline multilayer path caveats | Job `z_mm` = build height; see coupled `_resolve_arc_k` |
