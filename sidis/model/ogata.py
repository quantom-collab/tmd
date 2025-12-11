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

    def __init__(self, nu: int = 0):
        super().__init__()

        # self.Nmax = 200
        # Load config
        current_dir = pathlib.Path(__file__).parent
        config_file = current_dir / "../config.yaml"
        conf = OmegaConf.load(config_file)

        self.nu = nu
        self.bTmin = 1e-3

        def get_h_dynamic_func(first_zero: torch.Tensor, qT: torch.Tensor) -> torch.Tensor:
            """
            We use a dynamic h based on the qT values.
            We want to have enough coverage of the bT space.
            We want to use the fewest number of points with enough coverage.
            This formula is derived from the fact that at the first zero of the Bessel function,
            we evaluate the integrand at bTmin.
            """
            return torch.arcsinh(
                2 / torch.pi * torch.arctanh(self.bTmin * qT / first_zero)
            ) / (first_zero / torch.pi)

        """
        Calculate the Ogata weights and define the dynamic h function and the Bessel function for the Hankel transform.
        For nu = 0, we use J0.
        For nu = 1, we use J1.
        """
        if self.nu == 0:
            self.besselj0 = torch.tensor(jn_zeros(0,conf.bgrid.Nb), dtype=torch.get_default_dtype()).unsqueeze(0) #-- (1, Nb)
            self.xi = self.besselj0 / torch.pi #-- (1, Nb)
            x = torch.pi * self.xi
            J1 = torch.special.bessel_j1(x)
            self.ogata_weights = 2.0 / (torch.pi**2 * self.xi * J1**2)  # (1, Nb)

            self.get_h_dynamic = lambda qT: get_h_dynamic_func(self.besselj0[:, 0], qT)
            self.BesselJ_for_Hankel = lambda x: torch.special.bessel_j0(x)
        elif self.nu == 1:
            self.besselj1 = torch.tensor(jn_zeros(1,conf.bgrid.Nb), dtype=torch.get_default_dtype()).unsqueeze(0) #-- (1, Nb)
            self.xi = self.besselj1 / torch.pi #-- (1, Nb)
            # ---- Compute J2 using recurrence (Torch-native) ----
            # J2(x) = 2 J1(x) / x - J0(x)
            x = torch.pi * self.xi
            J0 = torch.special.bessel_j0(x)
            J1 = torch.special.bessel_j1(x)
            J2 = 2 * J1 / x - J0
            self.ogata_weights = 2.0 / (torch.pi**2 * self.xi * J2**2)  # (1, Nb)

            self.get_h_dynamic = lambda qT: get_h_dynamic_func(self.besselj1[:, 0], qT)
            self.BesselJ_for_Hankel = lambda x: torch.special.bessel_j1(x)
        else:
            raise ValueError(f"Invalid nu: {self.nu}")

    def get_psi(self, t: torch.Tensor) -> torch.Tensor:
        return t * (torch.tanh(torch.pi / 2 * torch.sinh(t)))

    def get_psi_prime(self, t: torch.Tensor) -> torch.Tensor:
        tanh_term = torch.tanh(torch.pi / 2 * torch.sinh(t))
        sech2_factor = 1 - tanh_term**2
        return tanh_term + (torch.pi / 2) * t * sech2_factor * torch.cosh(t)

    def get_h_fixed(self) -> torch.Tensor:
        return 0.0001

    def get_bTs(self, qT: torch.Tensor) -> torch.Tensor:
        """
        We make the bTs from the qTs.
        Each qT value has an array of bTs.
        """
        qT = qT.unsqueeze(1) #--here qT has shape (Nevents, 1)

        #--h is hyperparameter. We choose to use a dynamic h based on the qT values.
        h = self.get_h_dynamic(qT) #-- (Nevents, 1)
        
        bTs = torch.pi / h * self.get_psi(h * self.xi) / qT #-- this is of shape (Nevents, Nb)
        return bTs

    def eval_ogata_func_var_h(self, integrand: torch.Tensor, bT: torch.Tensor, qT: torch.Tensor) -> torch.Tensor:
        """
        It should be understood that the integrand here is defined according to the Ogata quadrature. 
        Namely, Ogata should integrate \int_0^\infty dy f(y) J_v(y).
        "integrand" here is f(y).
        So, the integrand is already multiplied by factors of bT^(v+1) because that is what is needed for the Hankel transform.
        So, we must not multiply by bT^(v+1) here.
        """

        qT = qT.unsqueeze(1) #--here qT has shape (Nevents, 1)

        h_val = self.get_h_dynamic(qT) #-- (Nevents, 1); defined in the initialization of the class.

        bT_qT = bT * qT #-- (Nevents, Nb)

        #J_vals = torch.special.bessel_j0(bT_qT) #-- (Nevents, Nb)
        J_vals = self.BesselJ_for_Hankel(bT_qT) #-- (Nevents, Nb)
        psi_prime_vals = self.get_psi_prime(h_val * self.xi) #-- (Nevents, Nb); defined in the initialization of the class.

        sum_integrand = self.ogata_weights * integrand * J_vals * psi_prime_vals
        sum = torch.sum(sum_integrand, axis=1)

        return torch.pi / qT.squeeze() * sum
