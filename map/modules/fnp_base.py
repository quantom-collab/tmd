"""
Base classes for TMD PDF and FF non-perturbative parameterizations.

This module contains the fundamental building blocks for TMD fNP parameterizations:
- Evolution factor module (shared across PDFs and FFs)
- Base TMD PDF class with MAP22 parameterization
- Base TMD FF class with MAP22 parameterization

Author: Chiara Bissolotti (cbissolotti@anl.gov)
Based on: MAP22g52.h from NangaParbat C++ implementation
"""

import torch
import torch.nn as nn
from typing import List, Optional
import math

# IPython imports - handle gracefully if not available
try:
    import importlib

    ipython_display = importlib.import_module("IPython.display")
    display = ipython_display.display
    Latex = ipython_display.Latex
    HAS_IPYTHON = True
except ImportError:
    HAS_IPYTHON = False

    def display(obj):
        """Fallback display function when IPython is not available."""
        print(obj)

    def Latex(text):
        """Fallback Latex function when IPython is not available."""
        return text


###############################################################################
# 1. Evolution Factor Module (Shared between PDFs and FFs)
###############################################################################
class fNP_evolution(nn.Module):
    """
    Evolution factor module implementing the shared non-perturbative evolution.

    This computes the Sudakov-like evolution factor that appears in both TMD PDFs and FFs:
    S_NP(ζ, b_T) = exp[-g₂² b_T²/4 x ln(ζ/Q₀²)]

    where:
    - g2: evolution parameter (trainable)
    - b_T: fourier-conjugate of k_T (GeV⁻¹)
    - ζ: rapidity scale (GeV²)
    - Q02: reference scale (Q_0^2 = 1 GeV^2 in MAP22)
    """

    def __init__(self, init_g2: float, free_mask: List[bool]):
        """
        Initialize evolution factor module.

        Args:
            init_g2 (float): Initial value for g2 parameter
            free_mask (List[bool]): Single-element list [True/False] for g2 trainability
        """
        # First, call the constructor of the parent class (nn.Module)
        # to initialize internal machinery.
        super().__init__()

        # Validate input
        if len(free_mask) != 1:
            raise ValueError(
                f"[fnp_base.py] \033[91mEvolution free_mask must have exactly 1 element, got {len(free_mask)}\033[0m"
            )

        # Reference scale Q_0^2 = 1 GeV^2 (MAP22 standard)
        self.register_buffer("Q0_squared", torch.tensor(1.0, dtype=torch.float32))

        # Parameter masking system
        mask = torch.tensor(free_mask, dtype=torch.float32)
        self.register_buffer("g2_mask", mask)

        init_tensor = torch.tensor([init_g2], dtype=torch.float32)

        # Fixed part (non-trainable)
        fixed_init = init_tensor * (1 - mask)
        self.register_buffer("fixed_g2", fixed_init)

        # Free part (trainable)
        free_init = init_tensor * mask
        self.free_g2 = nn.Parameter(free_init)

        # Gradient hook to ensure only free parameters get gradients
        self.free_g2.register_hook(lambda grad: grad * self.g2_mask)

    @property
    def g2(self):
        """Return the full g2 parameter (fixed + free parts)."""
        return (self.fixed_g2 + self.free_g2)[0]

    def forward(self, b: torch.Tensor, zeta: torch.Tensor) -> torch.Tensor:
        """
        Compute evolution factor S_NP(ζ, b_T).

        Args:
            b (torch.Tensor): fourier-conjugate of k_T (GeV⁻¹)
            zeta (torch.Tensor): Rapidity scale ζ (GeV²)

        Returns:
            torch.Tensor: Evolution factor exp[-g2² b_T²/4 x ln(ζ/Q₀²)]
        """
        return torch.exp(
            -(self.g2**2) * (b**2) * torch.log(zeta / self.Q0_squared) / 4.0
        )


###############################################################################
# 2. Base TMD PDF Class (MAP22 Implementation)
###############################################################################
class TMDPDFBase(nn.Module):
    """
    Base TMD PDF class implementing the MAP22 parameterization.

    This implements the exact MAP22 TMD PDF parameterization from the C++ code:

    f_NP(x, b_T) = NP_evol × [numerator] / [denominator]

    where:
    numerator = g₁×exp(-g₁×(b/2)²) + λ²×g₁ᵦ²×(1-g₁ᵦ×(b/2)²)×exp(-g₁ᵦ×(b/2)²) + g₁ᶜ×λ₂²×exp(-g₁ᶜ×(b/2)²)
    denominator = g₁ + λ²×g₁ᵦ² + g₁ᶜ×λ₂²

    g₁ = N₁ × (x/x̂)^σ₁ × ((1-x)/(1-x̂))^α₁²
    g₁ᵦ = N₁ᵦ × (x/x̂)^σ₂ × ((1-x)/(1-x̂))^α₂²
    g₁ᶜ = N₁ᶜ × (x/x̂)^σ₃ × ((1-x)/(1-x̂))^α₃²

    with x̂ = 0.1
    """

    def __init__(self, n_flavors: int, init_params: List[float], free_mask: List[bool]):
        """
        Initialize TMD PDF base class with MAP22 parameterization.

        Args:
            n_flavors (int): Number of flavor instances (typically 1)
            init_params (List[float]): 10 parameters [N₁, α₁, σ₁, λ, N₁ᵦ, N₁ᶜ, λ₂, α₂, α₃, σ₂, σ₃]
            free_mask (List[bool]): Trainability mask for each parameter
        """
        # First, call the constructor of the parent class (nn.Module)
        # to initialize internal machinery.
        super().__init__()

        # Validate parameters
        if len(init_params) != 11:
            raise ValueError(
                f"[fnp_base.py] \033[91mMAP22 TMD PDF requires 11 parameters, got {len(init_params)}\033[0m"
            )
        if len(free_mask) != len(init_params):
            raise ValueError(
                f"[fnp_base.py] \033[91mfree_mask length ({len(free_mask)}) must match init_params length ({len(init_params)})\033[0m"
            )

        self.n_flavors = n_flavors
        self.n_params = len(init_params)

        # Reference point x_hat = 0.1 (MAP22 standard)
        self.register_buffer("x_hat", torch.tensor(0.1, dtype=torch.float32))

        # Parameter setup with masking
        init_tensor = (
            torch.tensor(init_params, dtype=torch.float32)
            .unsqueeze(0)
            .repeat(n_flavors, 1)
        )
        mask = torch.tensor(free_mask, dtype=torch.float32).unsqueeze(0)
        self.register_buffer("mask", mask)

        # Fixed parameters (non-trainable)
        fixed_init = init_tensor * (1 - mask)
        self.register_buffer("fixed_params", fixed_init)

        # Free parameters (trainable)
        free_init = init_tensor * mask
        self.free_params = nn.Parameter(free_init)

        # Gradient hook
        self.free_params.register_hook(lambda grad: grad * self.mask)

    @property
    def get_params_tensor(self):
        """Return the full parameter tensor (fixed + free parts)."""
        return self.fixed_params + self.free_params

    def forward(
        self,
        x: torch.Tensor,
        b: torch.Tensor,
        zeta: torch.Tensor,
        NP_evol: torch.Tensor,
        flavor_idx: int = 0,
    ) -> torch.Tensor:
        """
        Compute TMD PDF using MAP22 parameterization.

        Args:
            x (torch.Tensor): Bjorken x variable
            b (torch.Tensor): fourier-conjugate of k_T (GeV⁻¹)
            zeta (torch.Tensor): Rapidity scale ζ (GeV²) [not used in this parameterization]
            NP_evol (torch.Tensor): Evolution factor from fNP_evolution
            flavor_idx (int): Flavor index (typically 0)

        Returns:
            torch.Tensor: TMD PDF f_NP(x, b)
        """
        # Handle x >= 1 case (return zero)
        if torch.any(x >= 1):
            mask_val = (x < 1).type_as(NP_evol)
        else:
            mask_val = torch.ones_like(x)

        # Extract parameters (MAP22 order matching C++ implementation)
        p = self.get_params_tensor[flavor_idx]
        N1 = p[0]  # N₁
        alpha1 = p[1]  # α₁
        sigma1 = p[2]  # σ₁
        lam = p[3]  # λ
        N1B = p[4]  # N₁ᵦ
        N1C = p[5]  # N₁ᶜ
        lam2 = p[6]  # λ₂
        alpha2 = p[7]  # α₂
        alpha3 = p[8]  # α₃
        sigma2 = p[9]  # σ₂
        sigma3 = p[10]  # σ₃

        # Compute intermediate g-functions (MAP22 exact implementation)
        g1 = (
            N1
            * torch.pow(x / self.x_hat, sigma1)
            * torch.pow((1 - x) / (1 - self.x_hat), alpha1**2)
        )
        g1B = (
            N1B
            * torch.pow(x / self.x_hat, sigma2)
            * torch.pow((1 - x) / (1 - self.x_hat), alpha2**2)
        )
        g1C = (
            N1C
            * torch.pow(x / self.x_hat, sigma3)
            * torch.pow((1 - x) / (1 - self.x_hat), alpha3**2)
        )

        # Compute (b/2)² term
        b_half_sq = (b / 2.0) ** 2

        # Numerator (exact MAP22 formula)
        numerator = (
            g1 * torch.exp(-g1 * b_half_sq)
            + (lam**2) * (g1B**2) * (1 - g1B * b_half_sq) * torch.exp(-g1B * b_half_sq)
            + g1C * (lam2**2) * torch.exp(-g1C * b_half_sq)
        )

        # Denominator (exact MAP22 formula)
        denominator = g1 + (lam**2) * (g1B**2) + g1C * (lam2**2)

        # Complete TMD PDF
        result = NP_evol * numerator / denominator

        return result * mask_val

    @property
    def latex_formula(self) -> str:
        """Return LaTeX formula for MAP22 TMD PDF parameterization."""
        return r"""$$
        f_{\rm NP}(x, b_T) = S_{\rm NP}(\zeta, b_T) \cdot \frac{
            g_1(x) \exp\left(-g_1(x) \frac{b_T^2}{4}\right) + 
            \lambda^2 g_{1B}^2(x) \left(1 - g_{1B}(x) \frac{b_T^2}{4}\right) \exp\left(-g_{1B}(x) \frac{b_T^2}{4}\right) +
            \lambda_2^2 g_{1C}(x) \exp\left(-g_{1C}(x) \frac{b_T^2}{4}\right)
        }{
            g_1(x) + \lambda^2 g_{1B}^2(x) + \lambda_2^2 g_{1C}(x)
        }
        $$
        $$
        \text{where} \quad
        g_{1,1B,1C}(x) = N_{1,1B,1C} \frac{x^{\sigma_{1,2,3}}(1-x)^{\alpha^2_{1,2,3}}}{\hat{x}^{\sigma_{1,2,3}}(1-\hat{x})^{\alpha^2_{1,2,3}}}, \quad \hat{x} = 0.1
        $$"""

    @property
    def show_latex_formula(self) -> None:
        """Display LaTeX formula in Jupyter notebook."""
        if HAS_IPYTHON:
            display(Latex(self.latex_formula))
        else:
            print("LaTeX formula:")
            print(self.latex_formula)


###############################################################################
# 3. Base TMD FF Class (MAP22 Implementation)
###############################################################################
class TMDFFBase(nn.Module):
    """
    Base TMD FF class implementing the MAP22 parameterization.

    This implements the MAP22 TMD FF parameterization from the C++ code:

    D_NP(z, b_T) = NP_evol x [numerator] / [denominator]

    where:
    numerator = g₃ × exp(-g₃×(b/2)²/z²) + (λ_F/z²)×g₃ᵦ²×(1-g₃ᵦ×(b/2)²/z²)×exp(-g₃ᵦ×(b/2)²/z²)
    denominator = g₃ + (λ_F/z²)×g₃ᵦ²

    g₃ = N₃ × [(z^β₁ + δ₁²)/(ẑ^β₁ + δ₁²)] × ((1-z)/(1-ẑ))^γ₁²
    g₃ᵦ = N₃ᵦ × [(z^β₂ + δ₂²)/(ẑ^β₂ + δ₂²)] × ((1-z)/(1-ẑ))^γ₂²

    with ẑ = 0.5
    """

    def __init__(self, n_flavors: int, init_params: List[float], free_mask: List[bool]):
        """
        Initialize TMD FF base class with MAP22 parameterization.

        Args:
            n_flavors (int): Number of flavor instances (typically 1)
            init_params (List[float]): 9 parameters [N₃, β₁, δ₁, γ₁, λ_F, N₃ᵦ, β₂, δ₂, γ₂]
            free_mask (List[bool]): Trainability mask for each parameter
        """
        # First, call the constructor of the parent class (nn.Module)
        # to initialize internal machinery.
        super().__init__()

        # Validate parameters
        if len(init_params) != 9:
            raise ValueError(
                f"[fnp_base.py] \033[91mMAP22 TMD FF requires 9 parameters, got {len(init_params)}\033[0m"
            )
        if len(free_mask) != len(init_params):
            raise ValueError(
                f"[fnp_base.py] \033[91mfree_mask length ({len(free_mask)}) must match init_params length ({len(init_params)})\033[0m"
            )

        self.n_flavors = n_flavors
        self.n_params = len(init_params)

        # Reference point z_hat = 0.5 (MAP22 standard)
        self.register_buffer("z_hat", torch.tensor(0.5, dtype=torch.float32))

        # Parameter setup with masking
        init_tensor = (
            torch.tensor(init_params, dtype=torch.float32)
            .unsqueeze(0)
            .repeat(n_flavors, 1)
        )
        mask = torch.tensor(free_mask, dtype=torch.float32).unsqueeze(0)
        self.register_buffer("mask", mask)

        # Fixed parameters (non-trainable)
        fixed_init = init_tensor * (1 - mask)
        self.register_buffer("fixed_params", fixed_init)

        # Free parameters (trainable)
        free_init = init_tensor * mask
        self.free_params = nn.Parameter(free_init)

        # Gradient hook
        self.free_params.register_hook(lambda grad: grad * self.mask)

    @property
    def get_params_tensor(self):
        """Return the full parameter tensor (fixed + free parts)."""
        return self.fixed_params + self.free_params

    def forward(
        self,
        z: torch.Tensor,
        b: torch.Tensor,
        zeta: torch.Tensor,
        NP_evol: torch.Tensor,
        flavor_idx: int = 0,
    ) -> torch.Tensor:
        """
        Compute TMD FF using MAP22 parameterization.

        Args:
            z (torch.Tensor): Energy fraction z variable
            b (torch.Tensor): fourier-conjugate of k_T (GeV⁻¹)
            zeta (torch.Tensor): Rapidity scale ζ (GeV²) [not used in this parameterization]
            NP_evol (torch.Tensor): Evolution factor from fNP_evolution
            flavor_idx (int): Flavor index (typically 0)

        Returns:
            torch.Tensor: TMD FF D_NP(z, b)
        """
        # Handle z >= 1 case (return zero)
        if torch.any(z >= 1):
            mask_val = (z < 1).type_as(NP_evol)
        else:
            mask_val = torch.ones_like(z)

        # Extract parameters (MAP22 order matching C++ implementation)
        p = self.get_params_tensor[flavor_idx]
        N3 = p[0]  # N₃
        beta1 = p[1]  # β₁
        delta1 = p[2]  # δ₁
        gamma1 = p[3]  # γ₁
        lambdaF = p[4]  # λ_F
        N3B = p[5]  # N₃ᵦ
        beta2 = p[6]  # β₂
        delta2 = p[7]  # δ₂
        gamma2 = p[8]  # γ₂

        # Compute intermediate functions (MAP22 exact implementation)
        cmn1 = (
            (z**beta1 + delta1**2) / (self.z_hat**beta1 + delta1**2)
        ) * torch.pow((1 - z) / (1 - self.z_hat), gamma1**2)
        cmn2 = (
            (z**beta2 + delta2**2) / (self.z_hat**beta2 + delta2**2)
        ) * torch.pow((1 - z) / (1 - self.z_hat), gamma2**2)

        g3 = N3 * cmn1
        g3B = N3B * cmn2

        # z² factor (important for FF parameterization)
        z2 = z * z

        # Compute (b/2)² term
        b_half_sq = (b / 2.0) ** 2

        # Numerator (exact MAP22 formula)
        numerator = g3 * torch.exp(-g3 * b_half_sq / z2) + (lambdaF / z2) * (
            g3B**2
        ) * (1 - g3B * b_half_sq / z2) * torch.exp(-g3B * b_half_sq / z2)

        # Denominator (exact MAP22 formula)
        denominator = g3 + (lambdaF / z2) * (g3B**2)

        # Complete TMD FF
        result = NP_evol * numerator / denominator

        return result * mask_val

    @property
    def latex_formula(self) -> str:
        """Return LaTeX formula for MAP22 TMD FF parameterization."""
        return r"""$$
        D_{\rm NP}(z, b_T) = S_{\rm NP}(\zeta, b_T) \cdot \frac{
            g_3(z) \exp\left(-g_3(z) \frac{b_T^2}{4z^2}\right) + 
            \frac{\lambda_F}{z^2} g_{3B}^2(z) \left(1 - g_{3B}(z) \frac{b_T^2}{4z^2}\right) \exp\left(-g_{3B}(z) \frac{b_T^2}{4z^2}\right)
        }{
            g_3(z) + \frac{\lambda_F}{z^2} g_{3B}^2(z)
        }
        $$
        $$
        \text{where} \quad
        g_{3,3B}(z) = N_{3,3B} \frac{(z^{\beta_{1,2}}+\delta^2_{1,2})(1-z)^{\gamma^2_{1,2}}}{(\hat{z}^{\beta_{1,2}}+\delta^2_{1,2})(1-\hat{z})^{\gamma^2_{1,2}}}, \quad \hat{z} = 0.5
        $$"""

    @property
    def show_latex_formula(self) -> None:
        """Display LaTeX formula in Jupyter notebook."""
        if HAS_IPYTHON:
            display(Latex(self.latex_formula))
        else:
            print("LaTeX formula:")
            print(self.latex_formula)


# Default parameter sets based on MAP22g52.h
MAP22_DEFAULT_EVOLUTION = {
    "init_g2": 0.12840,  # g2 from C++ implementation
    "free_mask": [True],
}

MAP22_DEFAULT_PDF_PARAMS = {
    "init_params": [
        0.28516,  # N₁
        0.29755,  # α₁ (scales to 2.9755 in C++)
        0.17293,  # σ₁
        0.39432,  # λ
        0.28516,  # N₁ᵦ
        0.28516,  # N₁ᶜ
        0.39432,  # λ₂
        0.29755,  # α₂ (scales to 2.9755 in C++)
        0.29755,  # α₃ (scales to 2.9755 in C++)
        0.17293,  # σ₂
        0.17293,  # σ₃
    ],
    "free_mask": [True] * 11,
}

MAP22_DEFAULT_FF_PARAMS = {
    "init_params": [
        0.21012,  # N₃
        2.12062,  # β₁
        0.093554,  # δ₁
        0.25246,  # γ₁
        5.2915,  # λ_F
        0.033798,  # N₃ᵦ
        2.1012,  # β₂
        0.093554,  # δ₂
        0.25246,  # γ₂
    ],
    "free_mask": [True] * 9,
}
