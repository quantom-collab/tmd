"""
LHAPDF / PDG flavor indexing for collinear T_F^q.

Indices (quark line numbers):

    1 = d,  2 = u,  3 = s
   -1 = dbar, -2 = ubar, -3 = sbar
    0 = gluon (zero for Qiu-Sterman T_F)
"""

from __future__ import annotations

import torch

from Spin2.qiu_sterman import FLAVORS

# Stacked tensor order: slot k holds PDG index PDG_INDICES[k]
PDG_INDICES: tuple[int, ...] = (-3, -2, -1, 0, 1, 2, 3)
N_PDG_SLOTS = len(PDG_INDICES)

NAME_TO_PDG: dict[str, int] = {
    "d": 1,
    "u": 2,
    "s": 3,
    "dbar": -1,
    "ubar": -2,
    "sbar": -3,
}

PDG_TO_NAME: dict[int, str | None] = {
    1: "d",
    2: "u",
    3: "s",
    -1: "dbar",
    -2: "ubar",
    -3: "sbar",
    0: None,
}

QUARK_PDG_INDICES: tuple[int, ...] = (1, 2, 3, -1, -2, -3)


def pdg_to_slot(pdg: int) -> int:
    """Map PDG flavor id to position in the (7, ...) stacked tensor."""
    try:
        return PDG_INDICES.index(pdg)
    except ValueError as exc:
        raise KeyError(f"Unknown PDG flavor index {pdg}") from exc


def slot_to_pdg(slot: int) -> int:
    return PDG_INDICES[slot]


def is_quark_pdg(pdg: int) -> bool:
    return pdg in QUARK_PDG_INDICES


def pack_TF_by_pdg(TF: torch.Tensor) -> torch.Tensor:
    """
    Pack T_F from FLAVORS-ordered stack to PDG-ordered stack.

    Parameters
    ----------
    TF : Tensor
        Shape ``(6, nx)`` or ``(6, nx, nQ2)`` in :data:`~Spin2.qiu_sterman.FLAVORS` order.

    Returns
    -------
    Tensor
        Shape ``(7, nx)`` or ``(7, nx, nQ2)``; slot for PDG 0 (gluon) is zero.
    """
    if TF.ndim not in (2, 3):
        raise ValueError(f"Expected TF with 2 or 3 dims, got shape {tuple(TF.shape)}")
    if TF.shape[0] != len(FLAVORS):
        raise ValueError(
            f"First dimension must be len(FLAVORS)={len(FLAVORS)}, got {TF.shape[0]}"
        )

    out = torch.zeros(
        (N_PDG_SLOTS,) + TF.shape[1:],
        dtype=TF.dtype,
        device=TF.device,
    )
    for i, name in enumerate(FLAVORS):
        out[pdg_to_slot(NAME_TO_PDG[name])] = TF[i]
    return out


def unpack_TF_by_pdg(TF_pdg: torch.Tensor) -> torch.Tensor:
    """
    Unpack PDG-ordered T_F to FLAVORS-ordered stack (drops gluon slot).
    """
    if TF_pdg.ndim not in (2, 3):
        raise ValueError(
            f"Expected TF_pdg with 2 or 3 dims, got shape {tuple(TF_pdg.shape)}"
        )
    if TF_pdg.shape[0] != N_PDG_SLOTS:
        raise ValueError(
            f"First dimension must be {N_PDG_SLOTS}, got {TF_pdg.shape[0]}"
        )

    out = torch.empty(
        (len(FLAVORS),) + TF_pdg.shape[1:],
        dtype=TF_pdg.dtype,
        device=TF_pdg.device,
    )
    for i, name in enumerate(FLAVORS):
        out[i] = TF_pdg[pdg_to_slot(NAME_TO_PDG[name])]
    return out
