"""
materials.py — Material property loader (waam_twin v2)
========================================================
Loads thermophysical properties from YAML files under materials/.
Built-in values in materials/placeholders/ are NOT validated — check `status`.
"""

from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass, field

from .material_tables import MaterialTables, interp_linear, tables_from_yaml

from .paths import PROJECT_ROOT as _PROJECT_ROOT
from .paths import resolve_project_path
_MATERIALS_JSON = _PROJECT_ROOT / "materials.json"
_SCHEMA_PATH = _PROJECT_ROOT / "materials" / "schema.json"
_PLACEHOLDER_DIR = _PROJECT_ROOT / "materials" / "placeholders"


@dataclass
class MaterialProps:
    """Complete thermophysical property set for one alloy (SI units)."""
    name: str
    k: float
    rho: float
    cp: float
    alpha: float
    T_solidus: float
    T_liquidus: float
    L_fusion: float
    gamma_0: float
    dgamma_dT: float
    T_ref_gamma: float
    mu: float
    beta_T: float = 1.2e-4
    contact_angle_deg: float = 80.0
    rho_e_ohm_m: float = 1.5e-7
    eta_stick: float = 0.85
    status: str = "placeholder"
    source: str = ""
    high_sulphur: bool = False
    tables: MaterialTables = field(default_factory=MaterialTables)

    def cp_at(self, T: float) -> float:
        return interp_linear(T, self.tables.cp, self.cp)

    def k_at(self, T: float) -> float:
        return interp_linear(T, self.tables.k, self.k)

    def mu_at(self, T: float) -> float:
        return interp_linear(T, self.tables.mu, self.mu)

    def dgamma_dT_at(self, T: float) -> float:
        return interp_linear(T, self.tables.dgamma_dT, self.dgamma_dT)


def _props_from_constants(name: str, status: str, source: str, c: dict) -> MaterialProps:
    rho = float(c["rho"])
    cp = float(c["cp"])
    k = float(c["k"])
    alpha = float(c.get("alpha", k / (rho * cp)))
    return MaterialProps(
        name=name,
        k=k,
        rho=rho,
        cp=cp,
        alpha=alpha,
        T_solidus=float(c["T_solidus"]),
        T_liquidus=float(c["T_liquidus"]),
        L_fusion=float(c["L_fusion"]),
        gamma_0=float(c["gamma_0"]),
        dgamma_dT=float(c["dgamma_dT"]),
        T_ref_gamma=float(c.get("T_ref_gamma", c["T_liquidus"])),
        mu=float(c["mu"]),
        beta_T=float(c.get("beta_T", 1.2e-4)),
        status=status,
        source=source or "",
    )


def _surface_from_yaml(data: dict) -> tuple[float, float, float]:
    surface = data.get("surface", {}) or {}
    electrical = data.get("electrical", {}) or {}
    theta = float(surface.get("contact_angle_deg", 80.0))
    rho_e = float(electrical.get("rho_e_ohm_m", 1.5e-7))
    eta_stick = float(electrical.get("eta_stick", 0.85))
    return theta, rho_e, eta_stick


def validate_material_data(data: dict, path: pathlib.Path | None = None) -> None:
    """Validate loaded YAML against materials/schema.json when jsonschema is installed."""
    if not _SCHEMA_PATH.exists():
        return
    label = str(path or data.get("name", "material"))
    try:
        import json
        import jsonschema

        with open(_SCHEMA_PATH) as f:
            schema = json.load(f)
        jsonschema.validate(instance=data, schema=schema)
    except ImportError:
        required = {"name", "status", "constants"}
        missing = required - set(data.keys())
        if missing:
            raise ValueError(f"Material {label} missing keys: {sorted(missing)}")
        c = data["constants"]
        for key in ("rho", "cp", "k", "T_solidus", "T_liquidus", "L_fusion"):
            if key not in c:
                raise ValueError(f"Material {label} constants missing '{key}'")
    except Exception as exc:
        from jsonschema import ValidationError

        if isinstance(exc, ValidationError):
            raise ValueError(f"Material schema validation failed for {label}: {exc.message}") from exc
        raise


def _load_yaml_file(path: pathlib.Path) -> MaterialProps:
    try:
        import yaml
    except ImportError as exc:
        raise ImportError("PyYAML required: pip install pyyaml") from exc
    with open(path) as f:
        data = yaml.safe_load(f)
    validate_material_data(data, path)
    name = data.get("name", path.stem)
    props = _props_from_constants(
        name,
        data.get("status", "placeholder"),
        data.get("source", ""),
        data["constants"],
    )
    theta, rho_e, eta_stick = _surface_from_yaml(data)
    props.contact_angle_deg = theta
    props.rho_e_ohm_m = rho_e
    props.eta_stick = eta_stick
    props.tables = tables_from_yaml(data.get("tables"))
    return props


def _resolve_material_path(name: str) -> pathlib.Path | None:
    p = resolve_project_path(name)
    if p.suffix in (".yaml", ".yml") and p.exists():
        return p
    if p.exists():
        return p
    for base in (_PROJECT_ROOT / "materials" / "validated", _PLACEHOLDER_DIR):
        for ext in (".yaml", ".yml"):
            candidate = base / f"{name}{ext}"
            if candidate.exists():
                return candidate
            candidate = base / name
            if candidate.exists():
                return candidate
    return None


# Legacy fallback when YAML is missing
_LEGACY_LIBRARY: dict[str, MaterialProps] = {
    "ER70S-6": _props_from_constants(
        "ER70S-6", "placeholder", "legacy Python constant",
        dict(k=30.0, rho=7000.0, cp=680.0, alpha=6.3e-6,
             T_solidus=1748.0, T_liquidus=1793.0, L_fusion=272_000.0,
             gamma_0=1.8, dgamma_dT=-4.3e-4, T_ref_gamma=1793.0, mu=6.5e-3),
    ),
    "SS316L": _props_from_constants(
        "SS316L", "placeholder", "legacy Python constant",
        dict(k=16.3, rho=6900.0, cp=750.0, alpha=3.1e-6,
             T_solidus=1648.0, T_liquidus=1723.0, L_fusion=285_000.0,
             gamma_0=1.6, dgamma_dT=-3.5e-4, T_ref_gamma=1723.0, mu=5.5e-3),
    ),
    "AISI4043": _props_from_constants(
        "AISI4043", "placeholder", "legacy Python constant",
        dict(k=160.0, rho=2380.0, cp=900.0, alpha=74.0e-6,
             T_solidus=846.0, T_liquidus=904.0, L_fusion=397_000.0,
             gamma_0=0.87, dgamma_dT=-1.55e-4, T_ref_gamma=904.0, mu=1.3e-3,
             beta_T=2.5e-4),
    ),
}


def load_material(name: str, high_sulphur: bool = False) -> MaterialProps:
    """
    Load material from YAML path or materials/placeholders/<name>.yaml.
    Warns when status is placeholder.
    """
    env_path = pathlib.Path(name) if "/" in name or name.endswith((".yaml", ".yml")) else None
    path = _resolve_material_path(name) if env_path is None else resolve_project_path(name)

    if path is not None and path.exists():
        props = _load_yaml_file(path)
    else:
        key_match = next(
            (k for k in _LEGACY_LIBRARY if name.upper().startswith(k.upper())),
            None,
        )
        if key_match is None:
            if _MATERIALS_JSON.exists():
                with open(_MATERIALS_JSON) as f:
                    entries = json.load(f)
                for entry in entries:
                    if name.lower() in entry.get("name", "").lower():
                        raise KeyError(
                            f"Material '{name}' in materials.json has process params only. "
                            f"Add materials/placeholders/{name}.yaml"
                        )
            raise KeyError(
                f"Material '{name}' not found. Add YAML under materials/placeholders/ "
                f"or pass a file path."
            )
        props = _LEGACY_LIBRARY[key_match]

    if high_sulphur:
        props = MaterialProps(
            **{**props.__dict__, "dgamma_dT": abs(props.dgamma_dT), "high_sulphur": True}
        )

    if props.status == "placeholder":
        print(
            f"[waam_twin] WARNING: material '{props.name}' has status=placeholder. "
            f"Results are illustrative only. ({props.source})"
        )

    return props
