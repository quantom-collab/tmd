"""PyTorch wrappers for precomputed x-space LO non-singlet evolution matrices."""

from __future__ import annotations

import math
from typing import Iterator, Sequence

import torch
import torch.nn as nn

from Spin2.flavors import (
    N_PDG_SLOTS,
    PDG_INDICES,
    PDG_TO_NAME,
    pack_TF_by_pdg,
    pdg_to_slot,
)
from Spin2.tools import interpolate


def _interp_at_Q2(
    f_xQ2: torch.Tensor, Q2: torch.Tensor, Q2_target: float
) -> torch.Tensor:
    """Interpolate along Q^2 (linear in ln Q^2)."""
    logQ2 = torch.log(Q2)
    logt = torch.tensor([math.log(Q2_target)], dtype=Q2.dtype, device=Q2.device)
    L = interpolate(logQ2, logt, "linear", Q2.device, dtype=Q2.dtype)
    return torch.einsum("nq,q->n", f_xQ2, L[0])


class NonSingletEvolution(nn.Module):
    """
    Generic LO non-singlet evolution: f(..., nx, nQ2) = M @ f0(..., nx).

    Used for Qiu-Sterman T_F, transversity h_1, and Collins Hhat^{(3)} channels.
    """

    def __init__(
        self,
        matrix: torch.Tensor,
        x_grid: torch.Tensor,
        Q2_grid: torch.Tensor,
    ) -> None:
        super().__init__()
        self.register_buffer("matrix", matrix)
        self.register_buffer("x_grid", x_grid)
        self.register_buffer("Q2_grid", Q2_grid)

    def forward(self, f0: torch.Tensor) -> torch.Tensor:
        return torch.einsum("jik,...i->...jk", self.matrix, f0)

    @classmethod
    def from_dglap(cls, dglap) -> "NonSingletEvolution":
        return cls(dglap.M_TF, dglap.x, dglap.Q2)


class QiuStermanEvolution(NonSingletEvolution):
    """Evolve Qiu-Sterman T_F^q(x, x; Q)."""

    @classmethod
    def from_dglap(cls, dglap) -> "QiuStermanEvolution":
        return cls(dglap.M_TF, dglap.x, dglap.Q2)


class TransversityEvolution(NonSingletEvolution):
    """Evolve transversity h_1^q(x; Q) with P_h1."""

    @classmethod
    def from_dglap(cls, dglap) -> "TransversityEvolution":
        return cls(dglap.M_TF, dglap.x, dglap.Q2)


class CollinsHhatEvolution(NonSingletEvolution):
    """Evolve homogeneous Collins Hhat^{(3)}(z; Q) with P_h1."""

    @classmethod
    def from_dglap(cls, dglap) -> "CollinsHhatEvolution":
        return cls(dglap.M_TF, dglap.x, dglap.Q2)


def evolve_flavors(
    evolution: NonSingletEvolution, f0_batch: torch.Tensor
) -> torch.Tensor:
    """Evolve ``(n_flavors, nx)`` -> ``(n_flavors, nx, nQ2)``."""
    return evolution(f0_batch)


def evolve_channels(
    evolution: NonSingletEvolution, f0_batch: torch.Tensor
) -> torch.Tensor:
    """Evolve ``(n_channels, nz)`` -> ``(n_channels, nz, nQ2)``."""
    return evolution(f0_batch)


class EvolvedT_F:
    """
    Evolved Qiu-Sterman T_F^q(x, mu); PDG-indexed shape ``(7, nx, nQ2)``.
    Gluon slot (PDG 0) is always zero.
    """

    PDG_INDICES = PDG_INDICES

    def __init__(self, x: torch.Tensor, Q2: torch.Tensor, data: torch.Tensor) -> None:
        if data.shape[0] != N_PDG_SLOTS or data.ndim != 3:
            raise ValueError(f"expected data shape (7, nx, nQ2), got {tuple(data.shape)}")
        self.x = x
        self.Q2 = Q2
        self.data = data

    @property
    def nx(self) -> int:
        return int(self.x.shape[0])

    @property
    def nQ2(self) -> int:
        return int(self.Q2.shape[0])

    @classmethod
    def from_flavor_stack(cls, x: torch.Tensor, Q2: torch.Tensor, TF: torch.Tensor):
        return cls(x, Q2, pack_TF_by_pdg(TF))

    def __len__(self) -> int:
        return N_PDG_SLOTS

    def __iter__(self) -> Iterator[int]:
        return iter(self.PDG_INDICES)

    def __getitem__(self, pdg: int) -> torch.Tensor:
        if pdg == 0:
            return torch.zeros(self.nx, self.nQ2, dtype=self.data.dtype, device=self.data.device)
        if pdg not in PDG_TO_NAME or PDG_TO_NAME[pdg] is None:
            raise KeyError(f"Unknown PDG flavor index {pdg}")
        return self.data[pdg_to_slot(pdg)]

    def at_mu(self, mu: float) -> torch.Tensor:
        q2 = float(mu) ** 2
        out = torch.zeros((N_PDG_SLOTS, self.nx), dtype=self.data.dtype, device=self.data.device)
        for slot, pdg in enumerate(self.PDG_INDICES):
            if pdg == 0:
                continue
            out[slot] = _interp_at_Q2(self.data[slot], self.Q2, q2)
        return out

    def pdg_at_mu(self, pdg: int, mu: float) -> torch.Tensor:
        if pdg == 0:
            return torch.zeros(self.nx, dtype=self.data.dtype, device=self.data.device)
        return self.at_mu(mu)[pdg_to_slot(pdg)]

    def as_dict_at_mu(self, mu: float) -> dict[int, torch.Tensor]:
        stacked = self.at_mu(mu)
        return {pdg: stacked[pdg_to_slot(pdg)] for pdg in self.PDG_INDICES}


class EvolvedTransversity:
    """
    Evolved transversity h_1^q(x, mu); same PDG layout as :class:`EvolvedT_F`.

    There is no gluon transversity for a spin-1/2 nucleon. Sea h_1 may be zero
    or user-supplied; evolution does not enforce sea = 0.
    """

    PDG_INDICES = PDG_INDICES

    def __init__(self, x: torch.Tensor, Q2: torch.Tensor, data: torch.Tensor) -> None:
        if data.shape[0] != N_PDG_SLOTS or data.ndim != 3:
            raise ValueError(f"expected data shape (7, nx, nQ2), got {tuple(data.shape)}")
        self.x = x
        self.Q2 = Q2
        self.data = data

    @property
    def nx(self) -> int:
        return int(self.x.shape[0])

    @property
    def nQ2(self) -> int:
        return int(self.Q2.shape[0])

    @classmethod
    def from_flavor_stack(cls, x: torch.Tensor, Q2: torch.Tensor, h1: torch.Tensor):
        packed = pack_TF_by_pdg(h1)
        packed[pdg_to_slot(0)] = 0.0
        return cls(x, Q2, packed)

    def __getitem__(self, pdg: int) -> torch.Tensor:
        if pdg == 0:
            return torch.zeros(self.nx, self.nQ2, dtype=self.data.dtype, device=self.data.device)
        if pdg not in PDG_TO_NAME or PDG_TO_NAME[pdg] is None:
            raise KeyError(f"Unknown PDG flavor index {pdg}")
        return self.data[pdg_to_slot(pdg)]

    def at_mu(self, mu: float) -> torch.Tensor:
        q2 = float(mu) ** 2
        out = torch.zeros((N_PDG_SLOTS, self.nx), dtype=self.data.dtype, device=self.data.device)
        for slot, pdg in enumerate(self.PDG_INDICES):
            if pdg == 0:
                continue
            out[slot] = _interp_at_Q2(self.data[slot], self.Q2, q2)
        return out

    def pdg_at_mu(self, pdg: int, mu: float) -> torch.Tensor:
        if pdg == 0:
            return torch.zeros(self.nx, dtype=self.data.dtype, device=self.data.device)
        return self.at_mu(mu)[pdg_to_slot(pdg)]


class EvolvedCollinsHhat:
    """
    Evolved homogeneous Collins twist-3 FFs Hhat^{(3)}_{h/q}(z, mu).

    Independent variable is fragmentation momentum fraction ``z`` (internally
    the same log grid machinery as x-space DGLAP).
    """

    def __init__(
        self,
        z: torch.Tensor,
        Q2: torch.Tensor,
        values: torch.Tensor,
        channels: Sequence[str],
    ) -> None:
        if values.ndim != 3:
            raise ValueError(f"values must be (n_channels, nz, nQ2), got {tuple(values.shape)}")
        if values.shape[0] != len(channels):
            raise ValueError("values first dim must match len(channels)")
        self.z = z
        self.x = z
        self.Q2 = Q2
        self.values = values
        self.channels = tuple(channels)
        self._ch_index = {c: i for i, c in enumerate(self.channels)}

    @property
    def nz(self) -> int:
        return int(self.z.shape[0])

    @property
    def nQ2(self) -> int:
        return int(self.Q2.shape[0])

    def __getitem__(self, name: str) -> torch.Tensor:
        return self.channel(name)

    def channel(self, name: str) -> torch.Tensor:
        """Return ``(nz, nQ2)`` for one fragmentation channel."""
        return self.values[self._ch_index[name]]

    def at_mu(self, mu: float) -> torch.Tensor:
        """All channels at mu; shape ``(n_channels, nz)``."""
        q2 = float(mu) ** 2
        out = []
        for i in range(len(self.channels)):
            out.append(_interp_at_Q2(self.values[i], self.Q2, q2))
        return torch.stack(out, dim=0)

    def channel_at_mu(self, name: str, mu: float) -> torch.Tensor:
        """Single channel at mu; shape ``(nz,)``."""
        return self.at_mu(mu)[self._ch_index[name]]


def evolve_TF(evolution: QiuStermanEvolution, TF0: torch.Tensor) -> EvolvedT_F:
    TF = evolve_flavors(evolution, TF0)
    return EvolvedT_F.from_flavor_stack(evolution.x_grid, evolution.Q2_grid, TF)


def evolve_transversity(
    evolution: TransversityEvolution, h1_0: torch.Tensor
) -> EvolvedTransversity:
    h1 = evolve_flavors(evolution, h1_0)
    return EvolvedTransversity.from_flavor_stack(evolution.x_grid, evolution.Q2_grid, h1)


def evolve_collins_hhat(
    evolution: CollinsHhatEvolution,
    Hhat0: torch.Tensor,
    channels: Sequence[str],
) -> EvolvedCollinsHhat:
    if Hhat0.shape[0] != len(channels):
        raise ValueError("Hhat0 first dim must match len(channels)")
    out = evolve_channels(evolution, Hhat0)
    return EvolvedCollinsHhat(evolution.x_grid, evolution.Q2_grid, out, channels)
