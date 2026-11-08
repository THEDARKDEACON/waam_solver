# Tier B — Ansys Fluent 2024 R2 (VOF + deposition)

## Where the guide is

**Hands-on operator guide (use this):**

`D:\ansys_2024r2\fluent_tierB\README.md`

That guide assumes the **same Workbench project as Tier A**:

`D:\ansys_2024r2\Deposit_Study.wbpj`

Add a Fluent system beside Transient Thermal; keep Mechanical results.
Custom UDFs, params, journals, and compare scripts stay under `fluent_tierB\`.

---

## Win condition

- Fluent vs waam_twin CFD within +/-15% W/D (H if crown is scoreable).
- Park etch is **not** a gate; **no** Goldak/eta retune.
- Twin reference = dx 0.3 / C_darcy 10 CFD, **not** Tier A thermal (~3.4 mm).

Tier A (Mechanical ↔ twin thermal) is finished and separate.

---

## Same-project note

| System in Deposit_Study | Role |
|-------------------------|------|
| Transient Thermal (existing) | Tier A — do not delete |
| Fluid Flow / Fluent (new) | Tier B — own geometry with air gap for VOF |

Do not edit the Mechanical plate geometry in place to add the gas box; give Fluent its own Geometry cell (see README Step 2).

---

## Repo vs D: disk

| On D: | In this repo |
|-------|----------------|
| Full guide, UDFs, metrics | This short pointer + Tier A Mechanical guide |

Mechanical Tier A steps remain in [`SETUP_MECHANICAL.md`](SETUP_MECHANICAL.md).
