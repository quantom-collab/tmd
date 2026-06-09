"""
Qiu-Sterman (Sivers) collinear function T_F^q(x, x; mu).

Initial condition at mu0:

    T_F^q(x, x; mu0) = N_q(x) * f_q(x, mu0)

with the paper parameterization for N_q(x). The full T_F is then evolved by
:class:`Spin2.dglap.NonSingletDGLAP`, not N_q alone.

The parameter ``g1T`` (default 0.180 GeV^2) is the nonperturbative transverse-
momentum width of the b-space Sivers factor. It is recorded here for downstream
TMD or b-space models and is **not** used in collinear x-space DGLAP evolution.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch

from Spin2 import params

FLAVORS = ("u", "d", "s", "ubar", "dbar", "sbar")


@dataclass
class QiuStermanParams:
    """
    Paper parameterization of the collinear Sivers profile N_q(x).

    Used to build T_F^q(x, x; mu0) = N_q(x) f_q(x, mu0) before DGLAP evolution of
    T_F itself. PDF factors f_q must be supplied separately (e.g. from LHAPDF).

    Attributes
    ----------
    N_u, N_d, N_s, N_ubar, N_dbar, N_sbar : float
        Flavor amplitudes in N_q(x).
    alpha_u, alpha_d, alpha_sea, beta : float
        Powers in x^alpha (1-x)^beta; sea flavors share alpha_sea.
    g1T : float
        Nonperturbative Sivers width in GeV^2 for the **b-space** Gaussian (or
        related) transversity/Sivers factor. Stored for TMD pipelines only; collinear
        DGLAP in :mod:`Spin2.dglap` does not read this field.
    """

    N_u: float = 0.077
    N_d: float = -0.152
    N_s: float = 0.167
    N_ubar: float = -0.033
    N_dbar: float = -0.069
    N_sbar: float = -0.002

    alpha_u: float = 0.967
    alpha_d: float = 1.188
    alpha_sea: float = 0.936
    beta: float = 5.129

    g1T: float = params.g1T_default

    def N_amplitude(self, flavor: str) -> float:
        return {
            "u": self.N_u,
            "d": self.N_d,
            "s": self.N_s,
            "ubar": self.N_ubar,
            "dbar": self.N_dbar,
            "sbar": self.N_sbar,
        }[flavor]

    def alpha(self, flavor: str) -> float:
        if flavor == "u":
            return self.alpha_u
        if flavor == "d":
            return self.alpha_d
        return self.alpha_sea


def beta_binomial_norm(alpha: float, beta: float) -> float:
    """((alpha + beta)^(alpha+beta)) / (alpha^alpha * beta^beta)."""
    ab = alpha + beta
    return (ab**ab) / (alpha**alpha * beta**beta)


def N_q(
    x: torch.Tensor,
    flavor: str,
    qs_params: QiuStermanParams | None = None,
) -> torch.Tensor:
    """
    Nonperturbative Sivers profile N_q(x).

    N_q(x) = N_q * norm * x^alpha_q * (1-x)^beta
    """
    p = qs_params or QiuStermanParams()
    a = p.alpha(flavor)
    norm = beta_binomial_norm(a, p.beta)
    amp = p.N_amplitude(flavor)
    return amp * norm * x**a * (1.0 - x) ** p.beta


def build_TF_initial(
    x: torch.Tensor,
    fq: Mapping[str, torch.Tensor],
    qs_params: QiuStermanParams | None = None,
) -> torch.Tensor:
    """
    Build collinear Qiu-Sterman initial data T_F^q(x, x; mu0) = N_q(x) f_q(x, mu0).

    Uses :class:`QiuStermanParams` for N_q only; ``g1T`` is not involved. Evolve
    the returned tensor with :class:`Spin2.dglap.NonSingletDGLAP` / :class:`Spin2.evolution.QiuStermanEvolution`.

    Parameters
    ----------
    x : Tensor
        Shape (nx,).
    fq : mapping
        flavor -> f_q(x, mu0) with shape (nx,).

    Returns
    -------
    Tensor
        Shape (len(fq), nx) in the order of keys in ``fq``.
    """
    p = qs_params or QiuStermanParams()
    flavors = list(fq.keys())
    out = []
    for fl in flavors:
        out.append(N_q(x, fl, p) * fq[fl])
    return torch.stack(out, dim=0)


def default_flavor_pdf_placeholder(x: torch.Tensor) -> dict[str, torch.Tensor]:
    """
    Simple positive profiles for tests when no external PDF is available.

    f_q(x) ~ x^0.5 (1-x)^3 — not a physical PDF, only for sanity checks.
    """
    f = x**0.5 * (1.0 - x) ** 3
    return {fl: f for fl in FLAVORS}


def build_default_TF0(
    x: torch.Tensor, qs_params: QiuStermanParams | None = None
) -> torch.Tensor:
    """Paper N_q times placeholder PDF; shape (6, nx)."""
    fq = default_flavor_pdf_placeholder(x)
    return build_TF_initial(x, {fl: fq[fl] for fl in FLAVORS}, qs_params)
