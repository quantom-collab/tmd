import torch
import numpy as np
import pathlib
from omegaconf import OmegaConf

# Handle both direct execution and module import
try:
    from .utils import get_akima_derivatives_2d, interp_2d
except ImportError:
    from utils import get_akima_derivatives_2d, interp_2d

class OPE(torch.nn.Module):
    def __init__(self):
        super().__init__()

        #--this is an example of the OPE grids. Right now, only for u quark at Q=1 GeV.
        #--we'll update this for other quarks.

        # Load config
        current_dir = pathlib.Path(__file__).parent
        config_file = current_dir / "../config.yaml"
        conf = OmegaConf.load(config_file)
        
        # Get grid file path
        grid_file = current_dir / conf.ope.grid_file
        opegrids = np.genfromtxt(grid_file, skip_header=1)
        
        opegrids = np.genfromtxt(grid_file, skip_header=1)
        self.opegrids = torch.tensor(opegrids) # shape: (25000,3)
        self.setup_interpolation()

    def setup_interpolation(self):

        self.xvals = self.opegrids[:,0].unique() # shape: (500,)
        self.bTvals = self.opegrids[:,1].unique() # shape: (500,)
        self.opevals = self.opegrids[:,2].reshape(len(self.xvals),len(self.bTvals)) # shape: (500,500)

        self.d_x, self.d_bT, self.d_x_bT = get_akima_derivatives_2d(self.xvals, self.bTvals, self.opevals)
        # self.interpolate = interpolate

    # def locate_1d(self, x: torch.Tensor, Xs: torch.Tensor) -> tuple[torch.tensor, torch.tensor]:
    #     """
    #     x is a tensor of points to interpolate to
    #     Xs is the grid

    #     We should add a failsafe here for x outside the range
    #     """
    #     idx = torch.searchsorted(Xs,x) - 1
    #     return idx,idx+1
    
    # def interp_ope_temporary(self, x: torch.Tensor, bT: torch.Tensor, type:str="cubic") -> torch.Tensor:
    #     """
    #     Interpolate the OPE over the saved grids.
    #     """
    #     x = torch.clamp(x, self.xvals[0], self.xvals[-1])
    #     bT = torch.clamp(bT, self.bTvals[0], self.bTvals[-1])

    #     idx_x1, idx_x2 = self.locate_1d(x, self.xvals)
    #     idx_bT1, idx_bT2 = self.locate_1d(bT, self.bTvals)

    #     h_x = self.xvals[idx_x2] - self.xvals[idx_x1]
    #     h_bT = self.bTvals[idx_bT2] - self.bTvals[idx_bT1]
    #     h_x_bT = h_x * h_bT

    #     theta_x = (x - self.xvals[idx_x1][:,0]) / h_x[:,0]
    #     theta_bT = (bT - self.bTvals[idx_bT1]) / h_bT
    #     #--may need nan to num here on theta_bT

    #     if type.lower() == "bilinear":
    #         # Corner values
    #         z00 = self.opevals[idx_x1,   idx_bT1   ]   # (Nx,Ny)
    #         z01 = self.opevals[idx_x1,   idx_bT2 ]   # (Nx,Ny)
    #         z10 = self.opevals[idx_x2,   idx_bT1   ]   # (Nx,Ny)
    #         z11 = self.opevals[idx_x2,   idx_bT2 ]   # (Nx,Ny)

    #         tx = theta_x[:, None]   # (Nx,1) broadcast over y
    #         tbT = theta_bT[None, :]   # (1,Ny) broadcast over x

    #         # Standard bilinear blend
    #         out = (1 - tx) * (1 - tbT) * z00 \
    #             + (1 - tx) * tbT       * z01 \
    #             + tx       * (1 - tbT) * z10 \
    #             + tx       * tbT       * z11
    #         return out

    #     elif type.lower() == "cubic":
    #         #--cubic interpolation
    #         base_mat = torch.tensor([ [1.,0.,-3.,2.], [0.,0.,3.,-2.], [0.,1.,-2.,1.], [0.,0.,-1.,1.] ])
    #         coeff_mat = torch.stack([torch.stack([self.opevals[idx_x1,idx_bT1], self.opevals[idx_x1,idx_bT2], h_bT * self.d_bT[idx_x1,idx_bT1], h_bT * self.d_bT[idx_x1,idx_bT2]],axis=-1),
    #                                 torch.stack([self.opevals[idx_x2,idx_bT1], self.opevals[idx_x2,idx_bT2], h_bT * self.d_bT[idx_x2,idx_bT1], h_bT * self.d_bT[idx_x2,idx_bT2]],axis=-1),
    #                                 torch.stack([h_x * self.d_x[idx_x1,idx_bT1], h_x * self.d_x[idx_x1,idx_bT2], h_x_bT * self.d_x_bT[idx_x1,idx_bT1], h_x_bT * self.d_x_bT[idx_x1,idx_bT2]],axis=-1),
    #                                 torch.stack([h_x * self.d_x[idx_x2,idx_bT1], h_x * self.d_x[idx_x2,idx_bT2], h_x_bT * self.d_x_bT[idx_x2,idx_bT1], h_x_bT * self.d_x_bT[idx_x2,idx_bT2]],axis=-1)
    #                                 ], axis=-2)
            
    #         theta_x_vec = torch.stack([torch.ones(theta_x.shape), theta_x, theta_x**2, theta_x**3],axis=-1)
    #         theta_y_vec = torch.stack([torch.ones(theta_y.shape), theta_y, theta_y**2, theta_y**3],axis=-1)

    #         M1 = base_mat.T @ coeff_mat @ base_mat

    #         return (theta_x_vec[:,None,None,:] @ M1 @theta_y_vec[None,:,:,None]).squeeze()
    #     else:
    #         raise ValueError(f"Invalid interpolation type: {mode}")



    #     ope = (3 * theta_x**2 - 2 * theta_x**3) * self.opevals[idx_x2, idx_bT2] + (1 - 3 * theta_x**2 + 2 * theta_x**3) * self.opevals[idx_x1, idx_bT2] + (theta_x**2 * h_x * (theta_x - 1)) * self.d_x[idx_x2, idx_bT2] + (theta_x**2 * h_x * (theta_x - 1)) * self.d_x[idx_x1, idx_bT2]

    #     # ope = self.get_closest_grid_values(idx_x1, idx_x2, idx_y1, idx_y2, grid_type="ope")
    #     return interpolate(x, bT, self.)

    # """
    # These next two functions gets the closest grid values for the given indices as integers.
    # This is intended for point-by-point interpolation.
    # """
    # def locate_1d(self, x: torch.Tensor, Xs: torch.Tensor) -> tuple[int,int]:
    #     """
    #     x is a tensor of points to interpolate to
    #     Xs is the grid

    #     We should add a failsafe here for x outside the range
    #     """
    #     idx = torch.searchsorted(Xs,x) - 1
    #     return idx,idx+1

    # def get_closest_grid_values(self, idx_x1:int,idx_x2:int,idx_y1:int,idx_y2:int, grid_type:str="ope") -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    #     """
    #     Get the closest grid values for the given indices.
    #     """
    #     if grid_type == "ope":
    #         return self.opevals[idx_x1:idx_x2,idx_y1:idx_y2]
    #     elif grid_type == "d_x":
    #         return self.d_x[idx_x1:idx_x2,idx_y1:idx_y2]
    #     elif grid_type == "d_bT":
    #         return self.d_bT[idx_x1:idx_x2,idx_y1:idx_y2]
    #     elif grid_type == "d_x_bT":
    #         return self.d_x_bT[idx_x1:idx_x2,idx_y1:idx_y2]
    #     else:
    #         raise ValueError(f"Invalid grid type: {grid_type}")

    def interp_ope(self, x: torch.Tensor, bT: torch.Tensor, type:str="cubic") -> torch.Tensor:
        """
        Interpolate the OPE over the saved grids.
        """
        return interp_2d(x, bT, self.xvals, self.bTvals, self.opevals, self.d_x, self.d_bT, self.d_x_bT, type)

    def forward(self, x: torch.Tensor, bT: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the OPE.
        This calls the interpolator over the saved OPE grids.
        """
        # interpolate over the saved OPE grids
        # ope = self.interpolate(x, bT)
        # return ope

        ope = self.interp_ope(x, bT)

        return ope

        # return self.model(x)


if __name__ == "__main__":
    import pathlib
    rootdir = pathlib.Path(__file__).resolve().parent

    torch.set_default_dtype(torch.float64)


    ope = OPE()
    x = torch.tensor([0.1,0.2,0.3])
    bT = torch.tensor([0.1,0.2,0.3])
    print(ope(x, bT))