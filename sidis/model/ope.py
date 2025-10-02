import torch
import numpy as np
import pathlib
import sys

# import from utils module
from .utils import get_akima_derivatives_2d, interp_2d, interp_2d_multiple_events


class OPE(torch.nn.Module):
    def __init__(self, grid_file: str):
        super().__init__()

        # --this is an example of the OPE grids. Right now, only for u quark at Q=1 GeV.
        # --we'll update this for other quarks.

        # Get grid file path
        opegrids = np.genfromtxt(grid_file, skip_header=1)

        self.opegrids = torch.tensor(opegrids)  # shape: (25000,3)
        self.setup_interpolation()

    def setup_interpolation(self):

        self.xvals = self.opegrids[:, 0].unique()  # shape: (500,)
        self.bTvals = self.opegrids[:, 1].unique()  # shape: (500,)

        # NOTE: for now, we're only using the u quark, so it's (500, 500, 1).
        # Adding another flavors would make it self.opevals (500, 500, 2)
        self.opevals = self.opegrids[:, 2].reshape(
            len(self.xvals), len(self.bTvals)
        )  # shape: (500,500)

        self.d_x, self.d_bT, self.d_x_bT = get_akima_derivatives_2d(
            self.xvals, self.bTvals, self.opevals
        )

    def interp_ope(
        self, x: torch.Tensor, bT: torch.Tensor, type: str = "cubic"
    ) -> torch.Tensor:
        """
        Interpolate the OPE over the saved grids.
        """
        return interp_2d_multiple_events(
            x,
            bT,
            self.xvals,
            self.bTvals,
            self.opevals,
            self.d_x,
            self.d_bT,
            self.d_x_bT,
            type,
        )
        # return interp_2d(x, bT, self.xvals, self.bTvals, self.opevals, self.d_x, self.d_bT, self.d_x_bT, type)

    def forward(self, x: torch.Tensor, bT: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the OPE.
        This calls the interpolator over the saved OPE grids.
        """
        # interpolate over the saved OPE grids
        # ope = self.interpolate(x, bT)
        # return ope

        # --here x has shape (Nevents,) and bT has shape (Nevents, Nb)
        # opes = []
        # for i in range(len(x)):
        #     ope = self.interp_ope(torch.tensor(x[i,None]), torch.tensor(bT[i]))
        #     opes.append(ope)
        # ope = torch.stack(opes, axis=0)
        ope = self.interp_ope(x, bT)
        return ope

        # return self.model(x)


if __name__ == "__main__":
    import pathlib

    rootdir = pathlib.Path(__file__).resolve().parent

    torch.set_default_dtype(torch.float64)

    ope = OPE()
    x = torch.tensor([0.1, 0.2, 0.3])
    bT = torch.tensor([0.1, 0.2, 0.3])
    print(ope(x, bT))
