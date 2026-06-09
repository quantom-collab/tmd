"""
Unified LHAPDF input loaders for spin validation.

LHAPDF ``xfxQ(pid, x, Q)`` returns ``x * f(x, Q)`` for PDFs and ``z * D(z, Q)``
for fragmentation functions.  All loaders return ``f`` or ``D`` by dividing by
``x`` or ``z`` via :func:`~spin.validation.lhapdf_io.load_flavor_densities`.
"""

from __future__ import annotations

import warnings
from typing import Dict, Tuple

import numpy as np
import torch

from spin.qiu_sterman import FLAVORS
from spin.validation.lhapdf_io import load_flavor_densities, try_import_lhapdf

# Qiu–Sterman paper IC scale
QS_MU0_SQ = 1.9
QS_MU0_GEV = float(np.sqrt(QS_MU0_SQ))
QS_PDF_SET = "NNPDF40_nnlo_pch_as_01180"

# Transversity / Collins paper IC scale
TC_MU0_SQ = 2.4
TC_MU0_GEV = float(np.sqrt(TC_MU0_SQ))
TC_PDF_SET_F1 = "NNPDF40_nnlo_pch_as_01180"
TC_PDF_SET_G1 = "NNPDFpol20_nlo_as_01180_1000"
TC_FF_SET_PION = "JAM20-SIDIS_FF_pion_nlo"

PDF_MEMBER = 0
FF_MEMBER = 0
FF_FLAVORS = ("u", "d", "s")

PROTON_MASS_GEV = 0.9389


def collinear_sivers_M() -> float:
    """Proton mass for -2 M T_F validation plots (GeV)."""
    return PROTON_MASS_GEV


def _toy_f1(x: torch.Tensor) -> Dict[str, torch.Tensor]:
    return {
        "u": 2.0 * x**0.5 * (1.0 - x) ** 3,
        "d": 1.0 * x**0.7 * (1.0 - x) ** 4,
        "s": 0.2 * x**0.8 * (1.0 - x) ** 5,
        "ubar": 0.1 * x ** (-0.1) * (1.0 - x) ** 7,
        "dbar": 0.12 * x ** (-0.1) * (1.0 - x) ** 7,
        "sbar": 0.08 * x ** (-0.1) * (1.0 - x) ** 7,
    }


def _toy_g1(f1: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {
        "u": 0.6 * f1["u"],
        "d": -0.35 * f1["d"],
        "s": 0.0 * f1["s"],
        "ubar": 0.0 * f1["ubar"],
        "dbar": 0.0 * f1["dbar"],
        "sbar": 0.0 * f1["sbar"],
    }


def _toy_D_piplus(z: torch.Tensor) -> Dict[str, torch.Tensor]:
    return {
        "u": 1.5 * z**0.5 * (1.0 - z),
        "d": 0.6 * z**0.7 * (1.0 - z) ** 1.5,
        "s": 0.3 * z**0.8 * (1.0 - z) ** 1.8,
    }


def _load_pdf_set(
    x: torch.Tensor, Q2: float, pdf_set: str, flavors: Tuple[str, ...] = FLAVORS
) -> Dict[str, torch.Tensor]:
    lhapdf = try_import_lhapdf()
    if lhapdf is None:
        raise ImportError("lhapdf is not installed")
    pdf = lhapdf.mkPDF(pdf_set, PDF_MEMBER)
    return load_flavor_densities(pdf, flavors, x, Q2)


def load_proton_pdf_fq(
    x: torch.Tensor,
    Q2: float | None = None,
    *,
    dtype: torch.dtype = torch.float64,
    backend: str = "lhapdf",
) -> Dict[str, torch.Tensor]:
    """Unpolarized proton PDFs f_q(x, Q) for Qiu–Sterman IC (default Q² = 1.9 GeV²)."""
    q2 = QS_MU0_SQ if Q2 is None else float(Q2)
    if abs(q2 - QS_MU0_SQ) > 1e-9 * QS_MU0_SQ:
        raise ValueError(f"Qiu–Sterman paper IC uses Q2 = {QS_MU0_SQ}; got {q2}")
    x = x.to(dtype=dtype)
    if backend == "toy":
        return _toy_f1(x)
    if backend not in ("lhapdf", "nnpdf"):
        raise ValueError(f"pdf backend must be 'lhapdf' or 'toy'; got {backend!r}")
    try:
        return _load_pdf_set(x, q2, QS_PDF_SET)
    except ImportError as exc:
        warnings.warn(f"LHAPDF unavailable ({exc}); using toy PDFs.", stacklevel=2)
        return _toy_f1(x)


def load_unpolarized_f1(
    x: torch.Tensor,
    Q2: float | None = None,
    *,
    force_toy: bool = False,
    pdf_set: str = TC_PDF_SET_F1,
) -> Dict[str, torch.Tensor]:
    """Unpolarized f_1^q(x, Q) for transversity IC."""
    q2 = TC_MU0_SQ if Q2 is None else float(Q2)
    x = x.to(dtype=torch.float64)
    if force_toy:
        warnings.warn("Using toy unpolarized PDFs.", stacklevel=2)
        return _toy_f1(x)
    try:
        return _load_pdf_set(x, q2, pdf_set)
    except Exception as exc:
        warnings.warn(f"LHAPDF f1 unavailable ({exc}); using toy.", stacklevel=2)
        return _toy_f1(x)


def load_helicity_g1(
    x: torch.Tensor,
    Q2: float | None = None,
    *,
    force_toy: bool = False,
    helicity_set: str = TC_PDF_SET_G1,
    f1: Dict[str, torch.Tensor] | None = None,
) -> Dict[str, torch.Tensor]:
    """Helicity g_1^q(x, Q) for transversity IC."""
    q2 = TC_MU0_SQ if Q2 is None else float(Q2)
    x = x.to(dtype=torch.float64)
    if force_toy:
        f1_use = f1 if f1 is not None else _toy_f1(x)
        warnings.warn("Using toy helicity PDFs.", stacklevel=2)
        return _toy_g1(f1_use)
    try:
        return _load_pdf_set(x, q2, helicity_set)
    except Exception as exc:
        f1_use = f1 if f1 is not None else _toy_f1(x)
        warnings.warn(f"LHAPDF g1 unavailable ({exc}); using toy.", stacklevel=2)
        return _toy_g1(f1_use)


def load_transversity_inputs(
    x: torch.Tensor,
    Q2: float | None = None,
    *,
    force_toy: bool = False,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    q2 = TC_MU0_SQ if Q2 is None else float(Q2)
    f1 = load_unpolarized_f1(x, q2, force_toy=force_toy)
    g1 = load_helicity_g1(x, q2, force_toy=force_toy, f1=f1)
    return f1, g1


def load_pion_ff(
    z: torch.Tensor,
    Q2: float | None = None,
    *,
    force_toy: bool = False,
    ff_set: str = TC_FF_SET_PION,
) -> Dict[str, torch.Tensor]:
    """Pi+ fragmentation functions D_{pi+/q}(z, Q) for Collins IC."""
    q2 = TC_MU0_SQ if Q2 is None else float(Q2)
    z = z.to(dtype=torch.float64)
    if force_toy:
        warnings.warn("Using toy pi+ FFs.", stacklevel=2)
        return _toy_D_piplus(z)
    try:
        lhapdf = try_import_lhapdf()
        if lhapdf is None:
            raise ImportError("lhapdf is not installed")
        ff = lhapdf.mkPDF(ff_set, FF_MEMBER)
        return load_flavor_densities(ff, FF_FLAVORS, z, q2)
    except Exception as exc:
        warnings.warn(f"LHAPDF FF unavailable ({exc}); using toy.", stacklevel=2)
        return _toy_D_piplus(z)
