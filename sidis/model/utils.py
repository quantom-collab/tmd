"""
This module contains the utilities for the model.
We'll do interpolation here with Akima.
We also will do Ogata.
Some other utilities will be here.
"""

import torch

def get_akima_derivatives_2d(x_grid: torch.Tensor, y_grid: torch.Tensor, z_grid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Get the Akima derivatives for a 2D grid.
    x_grid, y_grid are the grid points, z_grid is the function values.
    """
    #--this is the Akima delta_x for the x-direction
    delta_x = torch.empty((z_grid.shape[0]+3,z_grid.shape[1]))
    delta_x[2:-2] = (z_grid[1:] - z_grid[:-1]) / (x_grid[1:] - x_grid[:-1])[:,None]
    delta_x[1] = 2 * delta_x[2] - delta_x[3]
    delta_x[0] = 2 * delta_x[1] - delta_x[2]
    delta_x[-2] = 2 * delta_x[-3] - delta_x[-4]
    delta_x[-1] = 2 * delta_x[-2] - delta_x[-3]

    #--this is the Akima delta_y for the y-direction
    delta_y = torch.empty((z_grid.shape[0],z_grid.shape[1]+3))
    delta_y[:,2:-2] = (z_grid[:,1:] - z_grid[:,:-1]) / (y_grid[1:] - y_grid[:-1])
    delta_y[:,1] = 2 * delta_y[:,2] - delta_y[:,3]
    delta_y[:,0] = 2 * delta_y[:,1] - delta_y[:,2]
    delta_y[:,-2] = 2 * delta_y[:,-3] - delta_y[:,-4]
    delta_y[:,-1] = 2 * delta_y[:,-2] - delta_y[:,-3]

    #--this is the Akima delta_xy for the xy-direction
    delta_xy = torch.empty((delta_x.shape[0], delta_y.shape[1]))
    delta_xy[:,2:-2] = (delta_x[:,1:] - delta_x[:,:-1]) / (y_grid[1:] - y_grid[:-1])
    delta_xy[:,1] = 2 * delta_xy[:,2] - delta_xy[:,3]
    delta_xy[:,0] = 2 * delta_xy[:,1] - delta_xy[:,2]
    delta_xy[:,-2] = 2 * delta_xy[:,-3] - delta_xy[:,-4]
    delta_xy[:,-1] = 2 * delta_xy[:,-2] - delta_xy[:,-3]

    #--this is the weights for the x-direction and true derivative calculation in x-direction
    w_x = torch.abs(delta_x[1:]-delta_x[:-1]) + torch.abs(delta_x[1:] + delta_x[:-1]) / 2
    w1_x = w_x[:-2]
    w2_x = w_x[2:]

    w1_x[torch.where(torch.all(torch.stack([w1_x==0,w2_x==0]),axis=0))] = 1
    w2_x[torch.where(torch.all(torch.stack([w1_x==0,w2_x==0]),axis=0))] = 1
    d_x = w2_x/(w1_x + w2_x) * delta_x[1:-2] + w1_x/(w1_x + w2_x) * delta_x[2:-1]
    
    #--this is the weights for the y-direction and true derivative calculation in y-direction
    w_y = torch.abs(delta_y[:,1:] - delta_y[:,:-1]) + torch.abs(delta_y[:,1:] + delta_y[:,:-1]) / 2
    w1_y = w_y[:,:-2]
    w2_y = w_y[:,2:]
    w1_y[torch.where(torch.all(torch.stack([w1_y==0,w2_y==0]),axis=0))] = 1
    w2_y[torch.where(torch.all(torch.stack([w1_y==0,w2_y==0]),axis=0))] = 1

    d_y = (w2_y * delta_y[:,1:-2] + w1_y * delta_y[:,2:-1]) / (w1_y + w2_y)

    #--this is the Akima d_xy for the xy-direction
    d_xy = ( (w2_x * (w2_y * delta_xy[1:-2,1:-2] + w1_y * delta_xy[1:-2,2:-1]) + 
              w1_x * (w2_y * delta_xy[2:-1,1:-2] + w1_y * delta_xy[2:-1,2:-1])) / 
              ((w1_x + w2_x) * (w1_y + w2_y)) )

    ### Modified Akima sets NaN values (from cases where data is constant for more than two nodes) to 0.
    ### This change prevents overshoot in that range
    d_x = torch.nan_to_num(d_x)
    d_y = torch.nan_to_num(d_y)

    return d_x, d_y, d_xy

def locate_1d(x: torch.Tensor, Xs: torch.Tensor) -> tuple[torch.tensor, torch.tensor]:
    """
    x is a tensor of points to interpolate to
    Xs is the grid

    We should add a failsafe here for x outside the range
    """
    idx = torch.searchsorted(Xs,x) - 1
    return idx,idx+1

def interp_2d(x:torch.Tensor, y: torch.Tensor, x_grid: torch.Tensor, y_grid: torch.Tensor, z_grid: torch.Tensor, dx_grid: torch.Tensor, dy_grid: torch.Tensor, dx_dy_grid: torch.Tensor, type:str="cubic") -> torch.Tensor:
        """
        Interpolate the OPE over the saved grids.
        """
        x = torch.clamp(x, x_grid[0], x_grid[-1])
        y = torch.clamp(y, y_grid[0], y_grid[-1])

        idx_x1, idx_x2 = locate_1d(x, x_grid)
        idx_x1 = idx_x1[:,None]
        idx_x2 = idx_x2[:,None]
        idx_y1, idx_y2 = locate_1d(y, y_grid)

        h_x = x_grid[idx_x2] - x_grid[idx_x1]
        h_y = y_grid[idx_y2] - y_grid[idx_y1]
        h_x_y = h_x * h_y

        theta_x = (x - x_grid[idx_x1][:,0]) / h_x[:,0]
        theta_y = (y - y_grid[idx_y1]) / h_y
        #--may need nan to num here on theta_bT

        if type.lower() == "bilinear":
            # Corner values
            z00 = z_grid[idx_x1,   idx_y1   ]   # (Nx,Ny)
            z01 = z_grid[idx_x1,   idx_y2 ]   # (Nx,Ny)
            z10 = z_grid[idx_x2,   idx_y1   ]   # (Nx,Ny)
            z11 = z_grid[idx_x2,   idx_y2 ]   # (Nx,Ny)

            tx = theta_x[:, None]   # (Nx,1) broadcast over y
            ty = theta_y[None, :]   # (1,Ny) broadcast over x

            # Standard bilinear blend
            out = (1 - tx) * (1 - ty) * z00 \
                + (1 - tx) * ty       * z01 \
                + tx       * (1 - ty) * z10 \
                + tx       * ty       * z11
            return out

        elif type.lower() == "cubic":
            #--cubic interpolation
            base_mat = torch.tensor([ [1.,0.,-3.,2.], [0.,0.,3.,-2.], [0.,1.,-2.,1.], [0.,0.,-1.,1.] ])
            coeff_mat = torch.stack([torch.stack([z_grid[idx_x1,idx_y1], z_grid[idx_x1,idx_y2], h_y * dy_grid[idx_x1,idx_y1], h_y * dy_grid[idx_x1,idx_y2]],axis=-1),
                                    torch.stack([z_grid[idx_x2,idx_y1], z_grid[idx_x2,idx_y2], h_y * dy_grid[idx_x2,idx_y1], h_y * dy_grid[idx_x2,idx_y2]],axis=-1),
                                    torch.stack([h_x * dx_grid[idx_x1,idx_y1], h_x * dx_grid[idx_x1,idx_y2], h_x_y * dx_dy_grid[idx_x1,idx_y1], h_x_y * dx_dy_grid[idx_x1,idx_y2]],axis=-1),
                                    torch.stack([h_x * dx_grid[idx_x2,idx_y1], h_x * dx_grid[idx_x2,idx_y2], h_x_y * dx_dy_grid[idx_x2,idx_y1], h_x_y * dx_dy_grid[idx_x2,idx_y2]],axis=-1)
                                    ], axis=-2)
            
            theta_x_vec = torch.stack([torch.ones(theta_x.shape), theta_x, theta_x**2, theta_x**3],axis=-1)
            theta_y_vec = torch.stack([torch.ones(theta_y.shape), theta_y, theta_y**2, theta_y**3],axis=-1)

            M1 = base_mat.T @ coeff_mat @ base_mat

            return (theta_x_vec[:,None,None,:] @ M1 @theta_y_vec[None,:,:,None]).squeeze()
        else:
            raise ValueError(f"Invalid interpolation type: {type}")


# def interpolate(x: torch.Tensor, bT: torch.Tensor) -> torch.Tensor:
#     """
#     Interpolate the OPE over the saved grids.
#     """
    