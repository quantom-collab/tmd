"""
Homogeneous Collins twist-3 fragmentation functions Hhat^{(3)}_{h/q}(z, mu).

The evolved collinear object is **Hhat^{(3)}**, not the Trento Collins moment
H_1^{perp(1)}. Relation:

    Hhat^{(3)}_{h/q}(z) = -2 z M_h H_1^{perp(1)}_{h/q}(z)|_Trento

In the homogeneous approximation of the paper, Hhat^{(3)} evolves with the same
LO non-singlet kernel as transversity (P_h1). This is approximate NLL / NLL',
not full twist-3 Collins evolution.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch

from spin.qiu_sterman import beta_binomial_norm

COLLINS_CHANNELS = ("fav", "unf", "unf_s")

_MH_PION = 0.1349768  # GeV, pi+ mass (PDG)


@dataclass
class CollinsParams:
    """Paper parameters for Hhat^{(3)} favored / unfavored ICs."""

    N_fav: float = -0.262
    alpha_fav: float = 1.69
    beta_fav: float = 0.0
    N_unf: float = 0.195
    alpha_unf: float = 0.32
    beta_unf: float = 0.0


def hhat_from_trento_moment(
    z: torch.Tensor,
    H1perp1: torch.Tensor,
    Mh: float,
    *,
    z_eps: float = 1e-12,
) -> torch.Tensor:
    """Hhat^{(3)} = -2 z M_h H_1^{perp(1)} (Trento convention)."""
    z_safe = torch.where(z.abs() > z_eps, z, torch.full_like(z, z_eps))
    return -2.0 * z_safe * Mh * H1perp1


def trento_moment_from_hhat(
    z: torch.Tensor,
    Hhat: torch.Tensor,
    Mh: float,
    *,
    z_eps: float = 1e-12,
) -> torch.Tensor:
    """H_1^{perp(1)}|_Trento from Hhat^{(3)}."""
    z_safe = torch.where(z.abs() > z_eps, z, torch.full_like(z, z_eps))
    return -Hhat / (2.0 * z_safe * Mh)


def build_collins_hhat_initial(
    z: torch.Tensor,
    D_piplus: Mapping[str, torch.Tensor],
    params: CollinsParams | None = None,
) -> tuple[tuple[str, ...], torch.Tensor]:
    """
    Paper IC for pi+ Collins Hhat channels at Q0.

    Parameters
    ----------
    D_piplus : mapping
        Keys ``'u'``, ``'d'``, ``'s'`` -> D_{pi+/q}(z, Q0), shape ``(nz,)``.

    Returns
    -------
    channels, Hhat0
        ``channels = ('fav', 'unf', 'unf_s')``, ``Hhat0.shape = (3, nz)``.
    """
    p = params or CollinsParams()
    norm_f = beta_binomial_norm(p.alpha_fav, p.beta_fav)
    norm_u = beta_binomial_norm(p.alpha_unf, p.beta_unf)
    zf = z ** p.alpha_fav * (1.0 - z) ** p.beta_fav
    zu = z ** p.alpha_unf * (1.0 - z) ** p.beta_unf

    fav = p.N_fav * norm_f * zf * D_piplus["u"]
    unf = p.N_unf * norm_u * zu * D_piplus["d"]
    unf_s = p.N_unf * norm_u * zu * D_piplus["s"]
    Hhat0 = torch.stack([fav, unf, unf_s], dim=0)
    return COLLINS_CHANNELS, Hhat0


def toy_D_piplus(z: torch.Tensor) -> dict[str, torch.Tensor]:
    """Toy pi+ FFs for validation."""
    return {
        "u": z**0.5 * (1.0 - z),
        "d": 0.5 * z**0.8 * (1.0 - z) ** 1.5,
        "s": 0.3 * z**0.8 * (1.0 - z) ** 1.8,
    }


def pion_mass_Mh() -> float:
    """Pion mass for Trento <-> Hhat conversion (GeV)."""
    return _MH_PION
