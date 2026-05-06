"""
Configuration loader for SIDIS calculations.

Legacy behavior: load ``sidis/config.yaml`` at import if the file exists.
If it is missing, seed from ``DEFAULT_PHYSICS`` (same content as the historical
YAML) so ``import qcdlib.config_loader`` never fails.

``TruthModel`` passes the **full** merged card (physics + fNP); ``apply_physics``
copies only the perturbative allowlist into ``qcdlib`` globals so ``ALPHAS`` /
``MODEL_TORCH`` match the YAML while ``get_grid_path`` keeps a physics-shaped dict.
"""

from __future__ import annotations

import copy
import os
from typing import Any, Dict, FrozenSet

import yaml

# Top-level keys that define collinear/TMD/Ogata settings for ``qcdlib`` and OPE paths.
# All other keys on the same YAML (``pdfs``, ``ffs``, ``evolution``, …) are ignored here.
_PHYSICS_KEYS: FrozenSet[str] = frozenset(
    {
        "default_dtype",
        "alphaS_order",
        "dglap_order",
        "idis_order",
        "tmd_order",
        "tmd_resummation_order",
        "Q20",
        "ope",
        "qToQcut",
        "bgrid",
    }
)

# Inlined default matching historical sidis/config.yaml (when no file on disk).
DEFAULT_PHYSICS: Dict[str, Any] = {
    "default_dtype": "float64",
    "alphaS_order": 2,
    "dglap_order": 1,
    "idis_order": 1,
    "tmd_order": 1,
    "tmd_resummation_order": 2,
    "Q20": 1.6384,
    "ope": {
        "grid_files": {
            "pdf": {
                "p": {
                    "u": "../../grids/grids/tmdpdf_u_Q_1.28.txt",
                    "d": "../../grids/grids/tmdpdf_d_Q_1.28.txt",
                    "s": "../../grids/grids/tmdpdf_s_Q_1.28.txt",
                    "c": "../../grids/grids/tmdpdf_c_Q_1.28.txt",
                    "cb": "../../grids/grids/tmdpdf_cb_Q_1.28.txt",
                    "sb": "../../grids/grids/tmdpdf_sb_Q_1.28.txt",
                    "db": "../../grids/grids/tmdpdf_db_Q_1.28.txt",
                    "ub": "../../grids/grids/tmdpdf_ub_Q_1.28.txt",
                }
            },
            "ff": {
                "pi_plus": {
                    "u": "../../grids/grids/tmdff_u_Q_1.28.txt",
                    "d": "../../grids/grids/tmdff_d_Q_1.28.txt",
                    "s": "../../grids/grids/tmdff_s_Q_1.28.txt",
                    "c": "../../grids/grids/tmdff_c_Q_1.28.txt",
                    "cb": "../../grids/grids/tmdff_cb_Q_1.28.txt",
                    "sb": "../../grids/grids/tmdff_sb_Q_1.28.txt",
                    "db": "../../grids/grids/tmdff_db_Q_1.28.txt",
                    "ub": "../../grids/grids/tmdff_ub_Q_1.28.txt",
                }
            },
        }
    },
    "qToQcut": 0.2,
    "bgrid": {"b_min": 1.0e-3, "Nb": 500},
}


def _load_initial_config() -> Dict[str, Any]:
    _config_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "config.yaml"
    )
    try:
        with open(_config_path, "r") as f:
            loaded = yaml.safe_load(f)
        assert isinstance(loaded, dict)
        return loaded
    except FileNotFoundError:
        return copy.deepcopy(DEFAULT_PHYSICS)
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing config.yaml: {e}") from e


_config = _load_initial_config()


def _publish_config(c: Dict[str, Any]) -> None:
    """Assign module-level exports from a full physics config dict."""
    global alphaS_order, dglap_order, idis_order, tmd_order, tmd_resummation_order
    global Q20, config
    required = (
        "alphaS_order",
        "dglap_order",
        "idis_order",
        "tmd_order",
        "tmd_resummation_order",
        "Q20",
        "ope",
    )
    for k in required:
        assert k in c, f"physics config missing required key {k!r}"
    alphaS_order = c["alphaS_order"]
    dglap_order = c["dglap_order"]
    idis_order = c["idis_order"]
    tmd_order = c["tmd_order"]
    tmd_resummation_order = c["tmd_resummation_order"]
    Q20 = c["Q20"]
    config = copy.deepcopy(c)


_publish_config(_config)


def apply_physics(full_card: Dict[str, Any]) -> None:
    """
    Overwrite global QCD settings from a **single** unified configuration dict.

    ``full_card`` is the full merged YAML (perturbative + fNP). Only keys in
    ``_PHYSICS_KEYS`` are extracted and merged onto ``DEFAULT_PHYSICS``; the rest
    is ignored so ``config`` stays a clean physics namespace for legacy helpers.
    """
    assert isinstance(full_card, dict)
    physics = {
        k: copy.deepcopy(full_card[k]) for k in _PHYSICS_KEYS if k in full_card
    }
    merged = copy.deepcopy(DEFAULT_PHYSICS)
    merged.update(physics)
    _publish_config(merged)


def get_grid_path(pdf_or_ff, hadron, flavor):
    """
    Helper function to get grid file paths from config.

    Args:
        pdf_or_ff: 'pdf' or 'ff'
        hadron: 'p' for proton (pdf) or 'pi_plus' for pion (ff)
        flavor: 'u', 'd', 's', 'c', 'ub', 'db', 'sb', 'cb'

    Returns:
        Path to grid file
    """
    return config["ope"]["grid_files"][pdf_or_ff][hadron][flavor]
