"""x and Q^2 grids for x-space evolution matrix construction."""

from __future__ import annotations

import numpy as np
import torch

from Spin2 import params


class EvolutionGrid:
    """
    Log-spaced x grid and piecewise log-spaced Q^2 grid.

    Q^2 is split between Q20 -> mb^2 (one third of points) and mb^2 -> 10^LQ2max
    (two thirds), matching the quantom-stats jamx convention.
    """

    def __init__(
        self,
        dev: str,
        dtype: torch.dtype = torch.float64,
        nx: int = 32,
        xmin: float = 1e-4,
        xmax: float = 0.9999,
        nQ2: int = 20,
        Q20: float = params.mc2,
        LQ2max: float = 4.5,
    ) -> None:
        if Q20 < params.mc2 - 0.03 or Q20 > params.mb2:
            raise ValueError(
                f"Q20={Q20} must lie between mc2={params.mc2} and mb2={params.mb2}"
            )

        self.dev = dev
        self.dtype = dtype
        self.nx = int(nx)
        self.xmin = float(xmin)
        self.xmax = float(xmax)
        self.nQ2 = int(nQ2)
        self.Q20 = float(Q20)
        self.LQ2max = float(LQ2max)

        self._setup_xgrid()
        self._setup_Q2grid()

    def _setup_xgrid(self) -> None:
        self.x = 10 ** torch.linspace(
            np.log10(self.xmin),
            np.log10(self.xmax),
            self.nx,
            dtype=self.dtype,
            device=self.dev,
        )
        self.xi = self.x
        self.xi_2D, self.x_2D = torch.meshgrid(self.xi, self.x, indexing="ij")

    def _setup_Q2grid(self) -> None:
        self.Q2_1 = 10 ** torch.linspace(
            np.log10(self.Q20),
            np.log10(params.mb2),
            round(self.nQ2 / 3),
            dtype=self.dtype,
            device=self.dev,
        )
        self.Q2_2 = (
            10
            ** torch.linspace(
                np.log10(params.mb2),
                self.LQ2max,
                round(self.nQ2 * 2 / 3) + 1,
                dtype=self.dtype,
                device=self.dev,
            )[1:]
        )
        self.Q2 = torch.cat((self.Q2_1, self.Q2_2))
        self.nQ2 = int(self.Q2.shape[0])
        self.nQ2_1 = int(self.Q2_1.shape[0])
