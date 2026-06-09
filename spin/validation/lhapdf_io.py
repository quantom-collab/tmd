"""
Shared LHAPDF access conventions for Spin validation loaders.

LHAPDF ``xfxQ(pid, x, Q)`` returns ``x * f(x, Q)`` for collinear PDFs and
``z * D(z, Q)`` for fragmentation functions.  Spin IC formulas need ``f`` and
``D``, so loaders divide by ``x`` or ``z`` after every ``xfxQ`` call.
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
import torch

from spin.flavors import NAME_TO_PDG  # noqa: F401 — used by loaders


def _safe_divide(xfx: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Convert LHAPDF ``xfxQ`` output to parton density ``f`` or FF ``D``."""
    return xfx / np.maximum(x, 1e-30)


def sample_xfxQ(
    pdf,
    pid: int,
    points: np.ndarray,
    Q: float,
) -> np.ndarray:
    """Evaluate ``xfxQ`` on a 1D grid of Bjorken ``x`` or fragmentation ``z``."""
    return np.array([pdf.xfxQ(pid, xi, Q) for xi in points], dtype=float)


def load_flavor_densities(
    pdf,
    flavors: Sequence[str],
    points: torch.Tensor,
    Q2: float,
) -> dict[str, torch.Tensor]:
    """
    Return ``f_fl(x, Q)`` or ``D_fl(z, Q)`` from an LHAPDF PDF/FF handle.

    ``points`` is Bjorken ``x`` for PDF sets or fragmentation ``z`` for FF sets.
    """
    Q = float(Q2) ** 0.5
    pts = points.detach().cpu().numpy().astype(float)
    out: dict[str, torch.Tensor] = {}
    for fl in flavors:
        pid = NAME_TO_PDG[fl]
        xfx = sample_xfxQ(pdf, pid, pts, Q)
        density = _safe_divide(xfx, pts)
        out[fl] = torch.tensor(density, dtype=points.dtype, device=points.device)
    return out


def try_import_lhapdf():
    try:
        import lhapdf  # noqa: F401

        return lhapdf
    except ImportError:
        return None


def lhapdf_xfxQ_at_point(
    mkpdf: Callable[[str, int], object],
    pdf_set: str,
    member: int,
    pid: int,
    point: float,
    Q2: float,
) -> float:
    """Raw ``xfxQ`` for loader regression tests."""
    pdf = mkpdf(pdf_set, member)
    return float(pdf.xfxQ(pid, point, float(Q2) ** 0.5))
