"""
Collinear quark transversity PDF h_1^q(x, mu).

Initial condition (paper ansatz at Q0):

    h_1^q(x, Q0) = N_q^h * norm * x^{a_q} (1-x)^{b_q} * 1/2 * [f_1^q + g_1^q]

Only u and d are nonzero by default; sea and strange are zero unless supplied.
There is no gluon transversity for a spin-1/2 nucleon; quarks evolve as a
non-singlet LO distribution with P_h1 (see :mod:`Spin2.kernels`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch

from Spin2.qiu_sterman import FLAVORS, beta_binomial_norm


@dataclass
class TransversityParams:
    """Paper table parameters for the transversity IC profile."""

    N_u: float = 0.85
    a_u: float = 0.69
    b_u: float = 0.05
    N_d: float = -1.0
    a_d: float = 1.79
    b_d: float = 7.00

    def N_amplitude(self, flavor: str) -> float:
        if flavor == "u":
            return self.N_u
        if flavor == "d":
            return self.N_d
        return 0.0

    def alpha(self, flavor: str) -> float:
        if flavor == "u":
            return self.a_u
        if flavor == "d":
            return self.a_d
        return 0.0

    def beta(self, flavor: str) -> float:
        if flavor == "u":
            return self.b_u
        if flavor == "d":
            return self.b_d
        return 0.0


def _profile(x: torch.Tensor, flavor: str, p: TransversityParams) -> torch.Tensor:
    a = p.alpha(flavor)
    b = p.beta(flavor)
    if p.N_amplitude(flavor) == 0.0:
        return torch.zeros_like(x)
    norm = beta_binomial_norm(a, b)
    return p.N_amplitude(flavor) * norm * x**a * (1.0 - x) ** b


def build_h1_initial(
    x: torch.Tensor,
    f1: Mapping[str, torch.Tensor],
    g1: Mapping[str, torch.Tensor],
    params: TransversityParams | None = None,
) -> torch.Tensor:
    """
    Build h_1^q(x, Q0) in FLAVORS order; shape ``(6, nx)``.

    Sea and strange default to zero when not in ``f1`` / ``g1``.
    """
    p = params or TransversityParams()
    out = []
    for fl in FLAVORS:
        prof = _profile(x, fl, p)
        if prof.abs().max() == 0:
            out.append(torch.zeros_like(x))
            continue
        f = f1.get(fl, torch.zeros_like(x))
        g = g1.get(fl, torch.zeros_like(x))
        out.append(0.5 * prof * (f + g))
    return torch.stack(out, dim=0)
