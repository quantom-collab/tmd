import torch
import numpy as np
import pathlib
import sys
from omegaconf import OmegaConf

from scipy.special import jn_zeros

# Add the sidis directory to Python path for imports
current_dir = pathlib.Path(__file__).resolve().parent
sidis_dir = current_dir.parent
if str(sidis_dir) not in sys.path:
    sys.path.insert(0, str(sidis_dir))

class OGATA(torch.nn.Module):

    def __init__(self):
        super().__init__()

        # self.Nmax = 200
        # Load config
        current_dir = pathlib.Path(__file__).parent
        config_file = current_dir / "../config.yaml"
        conf = OmegaConf.load(config_file)

        self.besselj0 = torch.tensor(jn_zeros(0,conf.bgrid.Nb)).unsqueeze(0) #-- (1, Nb)
        self.bTmin = 1e-3
        self.xi = self.besselj0 / torch.pi #-- (1, Nb)

        self.ogata_weights = 2 / (torch.pi**2 * self.xi * torch.square(torch.special.bessel_j1(torch.pi * self.xi))) #-- (1, Nb)


    def get_psi(self, t: torch.Tensor) -> torch.Tensor:
        return t * (torch.tanh(torch.pi / 2 * torch.sinh(t)))

    def get_psi_prime(self, t: torch.Tensor) -> torch.Tensor:
        return torch.tanh(torch.pi/2 * torch.sinh(t)) + torch.pi / 2 * t * (1 - torch.tanh(torch.pi * torch.sinh(t) / 2)**2) * torch.cosh(t)

    def ogata_weights(self, xi: torch.Tensor) -> torch.Tensor:
        return 2/(torch.pi**2 * xi * torch.square(torch.special.bessel_j1(torch.pi * xi)))

    def get_h_fixed(self) -> torch.Tensor:
        return 0.0001

    def get_h_dynamic(self, qT: torch.Tensor) -> torch.Tensor:
        """
        We use a dynamic h based on the qT values.
        We want to have enough coverage of the bT space.
        We want to use the fewest number of points with enough coverage.
        """
        return torch.arcsinh(2 / torch.pi * torch.arctanh(self.bTmin * qT / self.besselj0[:,0]))/(self.besselj0[:,0] / torch.pi)

    def get_bTs(self, qT: torch.Tensor) -> torch.Tensor:
        """
        We make the bTs from the qTs.
        Each qT value has an array of bTs.
        """
        qT = qT.unsqueeze(1) #--here qT has shape (Nevents, 1)

        #--h is hyperparameters
        h = self.get_h_dynamic(qT) #-- (Nevents, 1)
        
        bTs = torch.pi / h * self.get_psi(h * self.xi) / qT #-- this is of shape (Nevents, Nb)
        return bTs

    def eval_ogata_func_var_h(self, integrand: torch.Tensor, bT: torch.Tensor, qT: torch.Tensor) -> torch.Tensor:

        qT = qT.unsqueeze(1) #--here qT has shape (Nevents, 1)
        #print('qT.shape', qT.shape)

        h_val = self.get_h_dynamic(qT) #-- (Nevents, 1)
        #print('h_val.shape', h_val.shape)
        #h_val = self.get_h_fixed()

        bT_qT = bT * qT #-- (Nevents, Nb)
        #print('bT_qT.shape', bT_qT.shape)

        J_vals = torch.special.bessel_j0(bT_qT) #-- (Nevents, Nb)
        psi_prime_vals = self.get_psi_prime(h_val * self.xi) #-- (Nevents, Nb)
        #print('ogata_weights.shape', self.ogata_weights.shape)
        #print('integrand.shape', integrand.shape)
        #print('J_vals.shape', J_vals.shape)
        #print('psi_prime_vals.shape', psi_prime_vals.shape)

        sum_integrand = self.ogata_weights * integrand * J_vals * psi_prime_vals
        #print('sum_integrand.shape', sum_integrand.shape)
        #print('qT.shape', qT.shape)
        sum = torch.sum(sum_integrand, axis=1)
        #print('sum.shape', sum.shape)

        return torch.pi / qT.squeeze() * sum

        # self.ogata_weights = self.ogata_weights(self.xi) #-- (Nb,)

        
        # psi_vals = get_psi(h[:,None]*xi_vals[None,:]) ### nqT x Nmax
        # psip_vals = get_psi_prime(h[:,None]*xi_vals[None,:]) ### nqT x Nmax
        # func_eval_points = torch.pi/h[:,None] *psi_vals / qT[:, None]
        # func_vals = func_in((torch.pi/h[:,None]) *psi_vals /qT[:,None])#,**kwargs)
        # j_vals = torch.special.bessel_j0((torch.pi/h[:,None]) *psi_vals) #psi_vals[None,:]*qT[:,None])
        
        # weights = ogata_weights(xi_vals)
        # return (torch.pi/qT * torch.sum(weights*func_vals*j_vals*psip_vals,axis=1), func_eval_points)