import numpy as np
import pathlib
from omegaconf import OmegaConf
import mpmath
import torch
from scipy.special import spence

import sys
import os

# Add the sidis directory to Python path for imports
current_dir = pathlib.Path(__file__).resolve().parent
sidis_dir = current_dir.parent
if str(sidis_dir) not in sys.path:
    sys.path.insert(0, str(sidis_dir))

# Now we can import directly
import qcdlib.params as params
from qcdlib.alphaS import ALPHAS
from qcdlib.tmdmodel import MODEL_TORCH
from qcdlib.evolution_precalcs import r_Gamma, r_gamma, rbar, rbar0prime, delta

alphaS = ALPHAS()
tmdmodel = MODEL_TORCH()


class PERTURBATIVE_EVOLUTION(torch.nn.Module):
    """
    The final function is slightly different than Ebert, 2110.11360.
    """

    def __init__(self, order: int = 1):
        super().__init__()
        self.order = order
        self.aux = params

        # --Pre-computed coefficients (these are computed once)
        self.r_Gamma = r_Gamma[self.order]
        self.r_gamma = r_gamma[self.order]
        self.rbar = rbar[self.order]
        self.rbar0prime = rbar0prime  # --note that this is a tensor of shape (7,)
        self.delta = delta[self.order]

        self.alphaS = alphaS
        self.tmdmodel = tmdmodel

        # --threshold values
        self.mc2 = params.mc2
        self.mb2 = params.mb2

        # --Pre-compute alpha_S at thresholds (cached)
        self.alphaS_mc = self.alphaS.get_alphaS(self.mc2)
        self.alphaS_mb = self.alphaS.get_alphaS(self.mb2)

    # def Li2_torch(self, z):
    #     """
    #     True dilogarithm Li₂(z) using mpmath.polylog(2, z)
    #     """
    #     if isinstance(z, torch.Tensor):
    #         z_np = z.detach().cpu().numpy()

    #         # Handle both scalar and array cases
    #         if np.isscalar(z_np):
    #             result = complex(mpmath.polylog(2, complex(z_np)))
    #         else:
    #             result = np.array([complex(mpmath.polylog(2, complex(zi)))
    #                             for zi in z_np.flat]).reshape(z_np.shape)

    #         return torch.tensor(result, dtype=z.dtype, device=z.device)
    #     else:
    #         return torch.tensor(complex(mpmath.polylog(2, complex(z))))

    def Li2_torch(self, z):
        """Most efficient for CPU computation"""
        if isinstance(z, torch.Tensor):
            # Convert: Li2(z) = spence(1-z)
            w = 1 - z

            # Efficient conversion for CPU tensors
            if z.is_cuda:
                w_np = w.detach().cpu().numpy()  # GPU → CPU → NumPy
            else:
                w_np = w.detach().numpy()  # CPU → NumPy (faster)

            result_np = spence(w_np)  # Fast C implementation

            # Convert back efficiently
            result_tensor = torch.from_numpy(result_np).to(
                device=z.device, dtype=torch.complex128
            )
            return result_tensor
        else:
            return torch.tensor(spence(1 - z), dtype=torch.complex128)

    def get_eta_Gamma(
        self, alphaS_f: torch.Tensor, alphaS_i: torch.Tensor, Nf: int
    ) -> torch.Tensor:
        """
        Get the integral of the cusp anomalous dimension over the range of alphaS.
        alphaS_f: shape (nQ2,)
        alphaS_i: scalar
        delta: shape (4,)
        Result: shape (nQ2,) - same as alphaS_f
        """
        r_Gamma = self.r_Gamma.T[Nf]  # Shape: (4,)
        delta = self.delta.T[Nf]  # Shape: (4,)

        # --Handle broadcasting: alphaS_f can be (nQ2,), delta is (4,)
        if alphaS_f.dim() > 0:
            alphaS_f_exp = alphaS_f.unsqueeze(-1)  # (nQ2,) -> (nQ2, 1)
            delta_exp = delta.unsqueeze(0)  # (4,) -> (1, 4)
        else:
            alphaS_f_exp = alphaS_f
            delta_exp = delta

        # Now broadcasting works: (nQ2, 1) - (1, 4) = (nQ2, 4)
        numerator = alphaS_f_exp - 4 * torch.pi * delta_exp  # (nQ2, 4)
        denominator = alphaS_i - 4 * torch.pi * delta_exp  # (1, 4) -> (nQ2, 4)

        log_ratio = torch.log(numerator / denominator)  # (nQ2, 4)

        # Sum over the delta dimension (dim=1), keeping alphaS_f dimension
        return -0.5 * torch.sum(r_Gamma * log_ratio, dim=-1)  # (nQ2,)

    def get_K_gamma(
        self, alphaS_f: torch.Tensor, alphaS_i: torch.Tensor, Nf: int
    ) -> torch.Tensor:
        r_gamma = self.r_gamma.T[Nf]
        delta = self.delta.T[Nf]

        if alphaS_f.dim() > 0:
            alphaS_f_exp = alphaS_f.unsqueeze(-1)  # (nQ2, 1)
            delta_exp = delta.unsqueeze(0)  # (1, 4)
        else:
            alphaS_f_exp = alphaS_f
            delta_exp = delta

        log_ratio = torch.log(
            (alphaS_f_exp - 4 * torch.pi * delta_exp)
            / (alphaS_i - 4 * torch.pi * delta_exp)
        )

        return -0.5 * torch.sum(r_gamma * log_ratio, dim=-1)

    def get_K_Gamma(
        self, alphaS_f: torch.Tensor, alphaS_i: torch.Tensor, Nf: int
    ) -> torch.Tensor:

        aS = alphaS_f / 4 / torch.pi
        a0 = alphaS_i / 4 / torch.pi

        rbar = self.rbar.T[Nf]
        rbar0prime = self.rbar0prime[Nf]
        delta = self.delta.T[Nf]
        r_Gamma = self.r_Gamma.T[Nf]

        # --Handle broadcasting for multiple Q2 values
        if aS.dim() > 0:
            aS_expanded = aS.unsqueeze(-1)  # (nQ2, 1)
            delta_expanded = delta.unsqueeze(0)  # (1, 4)
            # --Compute L for all delta values at once
            L = torch.log(
                (aS_expanded - delta_expanded) / (a0 - delta_expanded)
            )  # (nQ2, 4)
        else:
            L = torch.log((aS - delta) / (a0 - delta))  # (4,)

        # --first line
        L0 = L[..., 0] if L.dim() > 1 else L[0]
        line1 = (
            0.25 * r_Gamma[0] * rbar0prime * (1 / aS + (L0 - 1) / a0)
            + 0.125 * r_Gamma[0] * rbar[0] * L0**2
        )

        # Combined logic for lines 2, 3, 4 - single conditional evaluation
        if self.order > 0:
            # Create index arrays once
            i_indices = torch.arange(1, self.order + 1, device=aS.device)

            # Line 2: Vectorized computation
            if aS.dim() > 0:
                # Multi-dimensional case
                delta_i = delta[i_indices]  # (order,)

                # Vectorized dilogarithm: (nQ2, 1) / (1, order) = (nQ2, order)
                delta_i_expanded = delta_i.unsqueeze(0)  # (1, order)

                dilog_aS_terms = self.Li2_torch(
                    aS_expanded / delta_i_expanded
                )  # (nQ2, order)
                dilog_a0_terms = self.Li2_torch(a0 / delta_i_expanded)  # (1, order)

                log_terms = L0.unsqueeze(-1) * torch.log(
                    1 - a0 / delta_i_expanded
                )  # (nQ2, order)

                line2_coeffs = (
                    r_Gamma[i_indices] * rbar[0] - r_Gamma[0] * rbar[i_indices]
                )  # (order,)
                line2_terms = (
                    log_terms + dilog_aS_terms - dilog_a0_terms
                )  # (nQ2, order)

                line2 = 0.25 * torch.sum(line2_coeffs * line2_terms, dim=-1)  # (nQ2,)

                # Line 3: Vectorized
                L_i = L[:, i_indices]  # (nQ2, order)
                L0_expanded = L0.unsqueeze(-1)  # (nQ2, 1)

                term1 = 0.5 * rbar[i_indices] * L_i**2
                term2 = rbar[0] * L0_expanded * L_i
                term3 = rbar0prime * ((L0_expanded - L_i) / delta_i_expanded + L_i / a0)

                line3 = 0.25 * torch.sum(
                    r_Gamma[i_indices] * (term1 + term2 + term3), dim=-1
                )

                # Line 4: Vectorized double sum
                if self.order > 1:
                    # Create meshgrid for i,j pairs where i≠j
                    i_mesh, j_mesh = torch.meshgrid(i_indices, i_indices, indexing="ij")
                    mask = i_mesh != j_mesh
                    i_pairs = i_mesh[mask]
                    j_pairs = j_mesh[mask]

                    L_j = L[:, j_pairs]  # (nQ2, n_pairs)
                    delta_i_pairs = delta[i_pairs].unsqueeze(0)  # (1, n_pairs)
                    delta_j_pairs = delta[j_pairs].unsqueeze(0)  # (1, n_pairs)

                    log_terms = L_j * torch.log(
                        (delta_i_pairs - aS.unsqueeze(-1))
                        / (delta_i_pairs - delta_j_pairs)
                    )

                    # Vectorized dilogarithm for pairs
                    dilog_fraction_aS = (aS.unsqueeze(-1) - delta_j_pairs) / (
                        delta_i_pairs - delta_j_pairs
                    )
                    dilog_aS_pairs = self.Li2_torch(dilog_fraction_aS)  # (nQ2, n_pairs)

                    dilog_fraction_a0 = (a0 - delta_j_pairs) / (
                        delta_i_pairs - delta_j_pairs
                    )
                    dilog_a0_pairs = self.Li2_torch(dilog_fraction_a0)  # (1, n_pairs)

                    line4_coeffs = r_Gamma[i_pairs] * rbar[j_pairs]  # (n_pairs,)
                    line4 = 0.25 * torch.sum(
                        line4_coeffs * (log_terms + dilog_aS_pairs - dilog_a0_pairs),
                        dim=-1,
                    )
                else:
                    line4 = torch.zeros_like(line2)

            else:
                # Scalar case - simpler
                delta_i = delta[i_indices]

                dilog_aS_terms = self.Li2_torch(aS / delta_i)
                dilog_a0_terms = self.Li2_torch(a0 / delta_i)
                log_terms = L0 * torch.log(1 - a0 / delta_i)

                line2_coeffs = (
                    r_Gamma[i_indices] * rbar[0] - r_Gamma[0] * rbar[i_indices]
                )
                line2 = 0.25 * torch.sum(
                    line2_coeffs * (log_terms + dilog_aS_terms - dilog_a0_terms)
                )

                # Line 3 and 4 for scalar case...
                L_i = L[i_indices]
                term1 = 0.5 * rbar[i_indices] * L_i**2
                term2 = rbar[0] * L0 * L_i
                term3 = rbar0prime * ((L0 - L_i) / delta_i + L_i / a0)
                line3 = 0.25 * torch.sum(r_Gamma[i_indices] * (term1 + term2 + term3))

                if self.order > 1:
                    # Double sum for scalar case
                    i_mesh, j_mesh = torch.meshgrid(i_indices, i_indices, indexing="ij")
                    mask = i_mesh != j_mesh
                    i_pairs = i_mesh[mask]
                    j_pairs = j_mesh[mask]

                    L_j = L[j_pairs]
                    delta_i_pairs = delta[i_pairs]
                    delta_j_pairs = delta[j_pairs]

                    log_terms = L_j * torch.log(
                        (delta_i_pairs - aS) / (delta_i_pairs - delta_j_pairs)
                    )
                    dilog_aS_pairs = self.Li2_torch(
                        (aS - delta_j_pairs) / (delta_i_pairs - delta_j_pairs)
                    )
                    dilog_a0_pairs = self.Li2_torch(
                        (a0 - delta_j_pairs) / (delta_i_pairs - delta_j_pairs)
                    )

                    line4_coeffs = r_Gamma[i_pairs] * rbar[j_pairs]
                    line4 = 0.25 * torch.sum(
                        line4_coeffs * (log_terms + dilog_aS_pairs - dilog_a0_pairs)
                    )
                else:
                    line4 = torch.zeros_like(line2)
        else:
            line2 = torch.zeros_like(line1)
            line3 = torch.zeros_like(line1)
            line4 = torch.zeros_like(line1)

        result = line1 + line2 + line3 + line4
        return result.real

    def get_Ktilde(self, bT: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
        """
        Collins-Soper kernel at small bT can be expanded in alphaS, Eq. (69) of PhysRevD.96.054011
        """

        aS = self.alphaS.get_a(mu**2)  # - alphaS/(4 pi)
        log = self.tmdmodel.get_Log(bT, mu)  # - Typical logarithm
        Nf = self.alphaS.get_Nf(mu**2)

        CF = self.aux.CF
        CA = self.aux.CA
        TR = self.aux.TR
        zeta2 = self.aux.zeta2
        zeta3 = self.aux.zeta3
        zeta4 = self.aux.zeta4
        zeta5 = self.aux.zeta5

        ##### for N3LL term - see 1604.07869 #####
        Gamma1 = 1
        Gamma2 = (67 / 9 - torch.pi**2 / 3) * CA - (20 / 9) * TR * Nf
        Gamma3 = (
            CA**2
            * (
                245 / 6
                - 134 * torch.pi**2 / 27
                + 11 * torch.pi**4 / 45
                + 22 * zeta3 / 3
            )
            + CA * TR * Nf * (-418 / 27 + 40 * torch.pi**2 / 27 - 56 * zeta3 / 3)
            + CF * TR * Nf * (-55 / 3 + 16 * zeta3)
            - (16 / 27) * TR**2 * Nf**2
        )
        beta1 = (11 / 3) * CA - (4 / 3) * TR * Nf  #
        beta2 = 34 * CA**2 / 3 - 20 * CA * TR * Nf / 3 - 4 * CF * TR * Nf  #

        d20 = CA * (404 / 27 - 14 * zeta3) - 112 * TR * Nf / 27
        d33 = (2 / 3) * Gamma1 * beta1**2
        d32 = 2 * Gamma2 * beta1 + Gamma1 * beta2
        d31 = 2 * beta1 * d20 + 2 * Gamma3
        d30 = (
            (-(CA**2) / 2)
            * (
                -176 * zeta3 * zeta2 / 3
                + 6392 * zeta2 / 81
                + 12328 * zeta3 / 27
                + 154 * zeta4 / 3
                - 192 * zeta5
                - 297029 / 729
            )
            - CA
            * TR
            * Nf
            * (-824 * zeta2 / 81 - 904 * zeta3 / 27 + 20 * zeta4 / 3 + 62626 / 729)
            - 2 * TR**2 * Nf**2 * (-32 * zeta3 / 9 - 1856 / 729)
            - CF * TR * Nf * (-304 * zeta3 / 9 - 16 * zeta4 + 1711 / 27)
        )
        ######################################

        Ktilde = torch.zeros_like(bT)

        if self.order > 0:

            Ktilde += -8 * CF * aS * log

        if self.order > 1:

            Ktilde += (
                8
                * CF
                * aS**2
                * (
                    +(2 / 3 * Nf - 11 / 3 * CA) * log**2
                    + (-67 / 9 * CA + torch.pi**2 / 3 * CA + 10 / 9 * Nf) * log
                    + (7 / 2 * zeta3 - 101 / 27) * CA
                    + 14 / 27 * Nf
                )
            )

        if self.order > 2:  # Eq. (D9) on p. 36 of 1604.07869, using that Ktilde =  -2D

            Ktilde += (
                -2
                * CF
                * aS**3
                * (d30 + 2 * log * d31 + 4 * log**2 * d32 + 8 * log**3 * d33)
            )  # see 1604.07869

        return Ktilde

    def compute_evolution_components(
        self, alphaS_f: torch.Tensor, alphaS_i: torch.Tensor, Nf0: int, Nf: int
    ) -> tuple:
        """Compute evolution components with threshold crossing"""
        if Nf0 == Nf:
            eta_Gamma = self.get_eta_Gamma(alphaS_f, alphaS_i, Nf)

            K_gamma = self.get_K_gamma(alphaS_f, alphaS_i, Nf)

            K_Gamma = self.get_K_Gamma(alphaS_f, alphaS_i, Nf)

        elif Nf0 == 4 and Nf == 5:
            eta_Gamma = self.get_eta_Gamma(
                self.alphaS_mb, alphaS_i, Nf0
            ) + self.get_eta_Gamma(alphaS_f, self.alphaS_mb, Nf)

            K_gamma = self.get_K_gamma(
                self.alphaS_mb, alphaS_i, Nf0
            ) + self.get_K_gamma(alphaS_f, self.alphaS_mb, Nf)

            K_Gamma = self.get_K_Gamma(
                self.alphaS_mb, alphaS_i, Nf0
            ) + self.get_K_Gamma(alphaS_f, self.alphaS_mb, Nf)

        elif Nf0 == 3 and Nf == 4:
            eta_Gamma = self.get_eta_Gamma(
                self.alphaS_mc, alphaS_i, Nf0
            ) + self.get_eta_Gamma(alphaS_f, self.alphaS_mc, Nf)

            K_gamma = self.get_K_gamma(
                self.alphaS_mc, alphaS_i, Nf0
            ) + self.get_K_gamma(alphaS_f, self.alphaS_mc, Nf)

            K_Gamma = self.get_K_Gamma(
                self.alphaS_mc, alphaS_i, Nf0
            ) + self.get_K_Gamma(alphaS_f, self.alphaS_mc, Nf)

        elif Nf0 == 3 and Nf == 5:
            eta_Gamma = (
                self.get_eta_Gamma(self.alphaS_mc, alphaS_i, Nf0)
                + self.get_eta_Gamma(self.alphaS_mb, self.alphaS_mc, 4)
                + self.get_eta_Gamma(alphaS_f, self.alphaS_mb, Nf)
            )

            K_gamma = (
                self.get_K_gamma(self.alphaS_mc, alphaS_i, Nf0)
                + self.get_K_gamma(self.alphaS_mb, self.alphaS_mc, 4)
                + self.get_K_gamma(alphaS_f, self.alphaS_mb, Nf)
            )

            K_Gamma = (
                self.get_K_Gamma(self.alphaS_mc, alphaS_i, Nf0)
                + self.get_K_Gamma(self.alphaS_mb, self.alphaS_mc, 4)
                + self.get_K_Gamma(alphaS_f, self.alphaS_mb, Nf)
            )

        else:
            raise ValueError(f"Unsupported Nf transition: {Nf0} -> {Nf}")

        return eta_Gamma, K_gamma, K_Gamma

    def get_S_rapidity(
        self, bT: torch.Tensor, Q20: torch.Tensor, Q2: torch.Tensor
    ) -> torch.Tensor:

        mub = self.tmdmodel.get_mub(bT)
        bstar = self.tmdmodel.get_bstar(bT)
        Ktilde = self.get_Ktilde(bstar, mub)  # .unsqueeze(-1)

        alphaS_mub = self.alphaS.get_alphaS(mub**2)
        alphaS_i = self.alphaS.get_alphaS(Q20)
        Nf = self.alphaS.get_Nf(Q2)
        Nf0 = self.alphaS.get_Nf(Q20)

        mc2 = params.mc2
        mb2 = params.mb2

        if Nf0 == Nf:
            eta_Gamma = self.get_eta_Gamma(alphaS_mub, alphaS_i, Nf)

        elif Nf0 == 4 and Nf == 5:
            alphaS_mb = self.alphaS.get_alphaS(mb2)

            eta_Gamma = self.get_eta_Gamma(alphaS_mb, alphaS_i, Nf0)

            eta_Gamma += self.get_eta_Gamma(alphaS_mub, alphaS_mb, Nf)

        elif Nf0 == 3 and Nf == 4:
            alphaS_mc = self.alphaS.get_alphaS(mc2)

            eta_Gamma = self.get_eta_Gamma(alphaS_mc, alphaS_i, Nf0)

            eta_Gamma += self.get_eta_Gamma(alphaS_mub, alphaS_mc, Nf)

        elif Nf0 == 3 and Nf == 5:
            alphaS_mb = self.alphaS.get_alphaS(mb2)
            alphaS_mc = self.alphaS.get_alphaS(mc2)

            eta_Gamma = self.get_eta_Gamma(alphaS_mc, alphaS_i, Nf0)

            eta_Gamma += self.get_eta_Gamma(alphaS_mb, alphaS_mc, 4)

            eta_Gamma += self.get_eta_Gamma(alphaS_mub, alphaS_mb, Nf)

        # eta_Gamma = eta_Gamma.unsqueeze(-1)
        log_ratio = torch.log(Q2 / Q20).unsqueeze(-1)
        # print('log_ratio.shape', log_ratio.shape)
        # print('Ktilde.shape', Ktilde.shape)
        # print('eta_Gamma.shape', eta_Gamma.shape)

        return torch.exp(0.5 * log_ratio * (Ktilde + eta_Gamma))

    def forward(
        self, bT: torch.Tensor, Q20: torch.Tensor, Q2: torch.Tensor
    ) -> torch.Tensor:
        """
        Complete forward pass of the perturbative evolution.
        """

        alphaS_f = self.alphaS.get_alphaS(Q2)
        alphaS_i = self.alphaS.get_alphaS(Q20)
        Nf = self.alphaS.get_Nf(Q2)
        Nf0 = self.alphaS.get_Nf(Q20)

        # mc2 = params.mc2
        # mb2 = params.mb2

        eta_Gamma, K_gamma, K_Gamma = self.compute_evolution_components(
            alphaS_f, alphaS_i, Nf0, Nf
        )
        # print('eta_Gamma.shape', eta_Gamma.shape)

        # --get rapidity evolution factor
        S_rapidity = self.get_S_rapidity(bT, Q20, Q2)
        # print('S_rapidity.shape', S_rapidity.shape)

        eta_Gamma_2d = eta_Gamma.unsqueeze(-1)
        K_gamma_2d = K_gamma.unsqueeze(-1)
        K_Gamma_2d = K_Gamma.unsqueeze(-1)
        log_ratio_2d = torch.log(Q2 / Q20).unsqueeze(-1)

        RGE_factor = torch.exp(
            K_gamma_2d - 0.5 * log_ratio_2d * eta_Gamma_2d + K_Gamma_2d
        )
        # print('RGE_factor.shape', RGE_factor.shape)
        total_evolution = S_rapidity * RGE_factor  # (nQ2,nbT)
        # total_evolution = RGE_factor #(nQ2,nbT)

        return total_evolution.real


if __name__ == "__main__":
    import pathlib

    rootdir = pathlib.Path(__file__).resolve().parent

    import time

    torch.set_default_dtype(torch.float64)

    t0 = time.time()
    pert_evo = PERTURBATIVE_EVOLUTION(order=3)
    bT = torch.linspace(0.01, 10, 100)
    Q20 = torch.tensor([1.28**2])
    Q2 = torch.linspace(1.28**2, 100, 1000000)
    sudakov = pert_evo.forward(bT, Q20, Q2)
    t1 = time.time()
    print(f"Time taken for evolution of shape {sudakov.shape}: {t1-t0} seconds")
