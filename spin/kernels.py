"""
LO non-singlet splitting kernels for x-space DGLAP matrix construction.

Kernel types
--------------
``qiu_sterman`` (default)
    P_qq^T(x) = P_qq(x) - eta * delta(1-x), eta = N_C by default.

``unpolarized``
    LO P_qq(x) = C_F [ (1+x^2)/(1-x)_+ + 3/2 delta(1-x) ].

``transversity``
    LO P_h1(x) = C_F [ 2x/(1-x)_+ + 3/2 delta(1-x) ].
    Same kernel as homogeneous Collins twist-3 FF evolution in the paper.
    No eta modification.
"""

from __future__ import annotations

import numpy as np
import torch

from spin import alphaS, params

KERNEL_TYPES = ("qiu_sterman", "unpolarized", "transversity")


class NonSingletKernels:
    """
    LO kernel triplet K = [K_xi, K_x, K_c] for one non-singlet evolution channel.
    """

    def __init__(
        self,
        dev: str,
        dtype: torch.dtype = torch.float64,
        kernel_type: str = "qiu_sterman",
        eta: float = params.NC,
        order: int = 0,
    ) -> None:
        if order != 0:
            raise ValueError("spin evolution supports LO kernels only (order=0).")
        if kernel_type not in KERNEL_TYPES:
            raise ValueError(
                f"kernel_type must be one of {KERNEL_TYPES}, got {kernel_type!r}"
            )
        if kernel_type != "qiu_sterman" and eta != 0.0:
            raise ValueError(
                f"eta is only meaningful for kernel_type='qiu_sterman'; got eta={eta}"
            )

        self.dev = dev
        self.dtype = dtype
        self.kernel_type = kernel_type
        self.eta = float(eta) if kernel_type == "qiu_sterman" else 0.0
        self.order = 0
        self.K: list = [None, None, None]
        self._build()

    def _build(self) -> None:
        aS = alphaS.get_alphaS

        def A(q2: float) -> float:
            return aS(q2) / (2.0 * np.pi)

        def _den(x: torch.Tensor, xi: torch.Tensor) -> torch.Tensor:
            den = 1.0 / (1.0 - x / xi)
            den[xi <= x] = 0.0
            return den

        def K_1_unpol(x: torch.Tensor, xi: torch.Tensor) -> torch.Tensor:
            den = _den(x, xi)
            kern = (1.0 + (x / xi) ** 2) / xi
            kern[xi <= x] = 0.0
            return kern * den

        def K_2_unpol(x: torch.Tensor, xi: torch.Tensor) -> torch.Tensor:
            den = _den(x, xi)
            kern = 2.0 * x / xi**2
            kern[xi <= x] = 0.0
            return kern * den

        def K_1_trans(x: torch.Tensor, xi: torch.Tensor) -> torch.Tensor:
            """Regular part of C_F * 2x/(1-x)_+ (jamx K_1 slot)."""
            den = _den(x, xi)
            kern = 2.0 * x / xi**2
            kern[xi <= x] = 0.0
            return kern * den

        def K_2_trans(x: torch.Tensor, xi: torch.Tensor) -> torch.Tensor:
            """
            Subtraction for ``2x/(1-x)_+`` plus prescription.

            Same LO jamx slot as unpolarized ``K_2_unpol``; implements
            ``P_h1 = P_qq - C_F(1-x)`` relative to the unpolarized kernel.
            """
            den = _den(x, xi)
            kern = 2.0 * x / xi**2
            kern[xi <= x] = 0.0
            return kern * den

        def K_3(x: torch.Tensor, xi: torch.Tensor) -> torch.Tensor:
            return (2.0 * torch.log(1.0 - x) + 1.5)[0, :]

        if self.kernel_type in ("qiu_sterman", "unpolarized"):
            K_1, K_2 = K_1_unpol, K_2_unpol
        else:
            K_1, K_2 = K_1_trans, K_2_trans

        def K_c(x: torch.Tensor, xi: torch.Tensor, q2: float) -> torch.Tensor:
            base = A(q2) * params.CF * K_3(x, xi)
            if self.kernel_type == "qiu_sterman":
                return base - A(q2) * self.eta
            return base

        self.K[0] = lambda x, xi, q2: A(q2) * params.CF * K_1(x, xi)
        self.K[1] = lambda x, xi, q2: -A(q2) * params.CF * K_2(x, xi)
        self.K[2] = K_c
