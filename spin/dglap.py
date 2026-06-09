"""
x-space non-singlet DGLAP matrix builder.

Builds M[nx, nx, nQ2] using Gaussian quadrature and RK stepping (jamx convention).
"""

from __future__ import annotations

import logging
import math
import os
import time

import torch

from Spin2 import params
from Spin2.base import EvolutionGrid
from Spin2.kernels import NonSingletKernels
from Spin2.tools import checkdir, interpolate

logger = logging.getLogger(__name__)

# Bump when the discrete shift-operator convention changes (invalidates cached .pt grids).
_CACHE_VERSION = "v3"


def cache_filename(
    kernel_type: str,
    order: int,
    nx: int,
    xmin: float,
    xmax: float,
    nQ2: int,
    Q20: float,
    LQ2max: float,
    eta: float = 0.0,
) -> str:
    base = (
        f"spin2_nonsinglet_kernel_{_CACHE_VERSION}_{kernel_type}_order{order}_nx{nx}_"
        f"xmin{xmin}_xmax{xmax}_nQ2{nQ2}_Q20{Q20}_LQ2max{LQ2max}"
    )
    if kernel_type == "qiu_sterman":
        return f"{base}_eta{eta}.pt"
    return f"{base}.pt"


class NonSingletDGLAP(EvolutionGrid):
    """
    Precompute evolution matrix M[nx, nx, nQ2] for LO non-singlet DGLAP.

    ``kernel_type='qiu_sterman'`` uses P_qq - eta delta(1-x) (eta defaults to N_C).
    ``kernel_type='transversity'`` uses P_h1 (transversity / homogeneous Collins).
    ``kernel_type='unpolarized'`` uses ordinary LO P_qq.
    """

    def __init__(
        self,
        dev: str = "cpu",
        dtype: torch.dtype = torch.float64,
        order: int = 0,
        kernel_type: str = "qiu_sterman",
        eta: float | None = None,
        nx: int = 32,
        xmin: float = 1e-4,
        xmax: float = 0.9999,
        nQ2: int = 20,
        Q20: float = params.mc2,
        LQ2max: float = 4.5,
        loadgrid: bool = True,
        grid_dir: str = "grids",
        steps: int = 5,
        ng: int = 50,
    ) -> None:
        self.order = int(order)
        self.kernel_type = kernel_type
        if kernel_type == "qiu_sterman":
            self.eta = float(params.NC if eta is None else eta)
        else:
            if eta is not None and float(eta) != 0.0:
                raise ValueError(
                    f"eta is only valid for kernel_type='qiu_sterman'; got eta={eta}"
                )
            self.eta = 0.0
        self.steps = int(steps)
        self.ng = int(ng)
        self.grid_dir = grid_dir

        super().__init__(dev, dtype, nx, xmin, xmax, nQ2, Q20, LQ2max)

        self.kernels = NonSingletKernels(
            dev, dtype, kernel_type=kernel_type, eta=self.eta, order=self.order
        )
        K = self.kernels.K

        import numpy as np

        yg, wg = np.polynomial.legendre.leggauss(self.ng)
        self.yg = torch.tensor(yg, dtype=self.dtype, device=self.dev)
        self.wg = torch.tensor(wg, dtype=self.dtype, device=self.dev)

        self.jac = torch.zeros((self.ng, self.nx), dtype=self.dtype, device=self.dev)
        self.Lg = torch.zeros(
            (self.ng, self.nx, self.nx), dtype=self.dtype, device=self.dev
        )
        self.xig_2D = torch.zeros((self.ng, self.nx), dtype=self.dtype, device=self.dev)
        self.xg_2D = torch.zeros((self.ng, self.nx), dtype=self.dtype, device=self.dev)

        xsmin = torch.log(self.x)
        xsmax = torch.log(torch.tensor(self.xmax, dtype=self.dtype, device=self.dev))
        self.xig_2D = torch.exp(
            0.5 * torch.einsum("j,g->gj", (xsmax - xsmin), self.yg)
            + 0.5 * (xsmax + xsmin)
        )
        self.jac = 0.5 * torch.einsum("j,gj->gj", (xsmax - xsmin), self.xig_2D)
        self.Lg = interpolate(self.xi, self.xig_2D, "cubic", self.dev, dtype=self.dtype)
        self.xg_2D = self.x.unsqueeze(0).repeat(self.ng, 1)

        checkdir(grid_dir)
        fname = cache_filename(
            self.kernel_type,
            self.order,
            nx,
            xmin,
            xmax,
            self.nQ2,
            Q20,
            LQ2max,
            eta=self.eta,
        )
        fpath = os.path.join(grid_dir, fname)

        if loadgrid and os.path.isfile(fpath):
            logger.info("Loading evolution matrix %s", fname)
            self.M_TF = torch.load(
                fpath, map_location=torch.device("cpu"), weights_only=True
            ).to(self.dev, dtype=self.dtype)
        else:
            t0 = time.time()
            self.M_TF = self.setup_TF_matrix(K)
            torch.save(self.M_TF.cpu(), fpath)
            logger.info("Built and saved %s in %.2f s", fname, time.time() - t0)

    def setup_TF_matrix(self, K: list) -> torch.Tensor:
        """Build M[nx, nx, nQ2] from Q20 through the Q^2 grid."""
        mu2ini = self.Q20
        mu2fin = self.Q2

        M_TF = torch.zeros(
            (self.nx, self.nx, self.nQ2), dtype=self.dtype, device=self.dev
        )

        def shift(mu2: float) -> torch.Tensor:
            return self.get_shift(mu2, K)

        for level in range(len(mu2fin)):
            if level == 0:
                mu2i, mu2f = float(mu2ini), float(mu2fin[level])
            else:
                mu2i, mu2f = float(mu2fin[level - 1]), float(mu2fin[level])
            idx = int(torch.searchsorted(self.Q2, mu2f))

            dt = math.log(mu2f / mu2i) / self.steps
            t = 0.0
            M = None
            for k in range(self.steps):
                mu20 = math.exp(t) * mu2i
                mu21 = math.exp(t + dt / 2) * mu2i
                mu22 = math.exp(t + dt) * mu2i
                shift0 = shift(mu20)
                shift1 = shift(mu21)
                shift2 = shift(mu22)
                T1 = torch.ones(shift0.shape, dtype=self.dtype, device=self.dev)
                T1[self.xi_2D != self.x_2D] = 0.0
                T2 = dt / 6 * (shift0 + 4 * shift1 + shift2)
                T3 = dt**2 / 6 * torch.einsum("ik,kj->ij", shift1, shift0 + shift1)
                T3 += dt**2 / 6 * torch.einsum("ik,kj->ij", shift2, shift1)
                T4 = dt**3 / 12 * torch.einsum("ik,kl,lj->ij", shift1, shift1, shift0)
                T4 += dt**3 / 12 * torch.einsum("ik,kl,lj->ij", shift2, shift1, shift1)
                T5 = dt**4 / 24 * torch.einsum(
                    "ik,kl,lm,mj->ij", shift2, shift1, shift1, shift0
                )
                step = T1 + T2 + T3 + T4 + T5
                if k == 0:
                    if level == 0:
                        M = step
                    else:
                        M = torch.einsum("ik,kj->ij", step, M_TF[:, :, idx - 1])
                else:
                    M = torch.einsum("ik,kj->ij", step, M)
                t += dt

            M_TF[:, :, idx] = M

        return M_TF

    def get_shift(self, mu2: float, K: list) -> torch.Tensor:
        nx = self.nx
        x_2D, xi_2D = self.x_2D, self.xi_2D

        Kxi = K[0](self.xg_2D, self.xig_2D, mu2)
        Kx = K[1](self.xg_2D, self.xig_2D, mu2)

        # Raw quadrature block; jamx convolution uses Axi.T (see jamx ``convolution``).
        Axi = torch.einsum("g,gj,gj,gji->ij", self.wg, self.jac, Kxi, self.Lg)

        Ac = torch.zeros((nx, nx), dtype=self.dtype, device=self.dev)
        Ac[:] = torch.einsum("g,gj,gj->j", self.wg, self.jac, Kx)

        Kc = K[2](x_2D, xi_2D, mu2)
        Ac[:] += Kc
        Ac[xi_2D != self.x_2D] = 0.0

        return Axi.T + Ac

    def evolve(self, f0: torch.Tensor) -> torch.Tensor:
        """Apply precomputed matrix: f(Q2) = M @ f0."""
        return torch.einsum("jik,...i->...jk", self.M_TF, f0)
