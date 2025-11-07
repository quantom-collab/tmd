"""
Standard Flavor-Dependent fNP Implementation

This is a module that provides a specific combination of TMD PDF and TMD FF
non-perturbative parameterizations in a way that is inspired by the MAP22 parameterization.
This combo implements the standard flavor-dependent
system where each quark flavor (u, d, s, ubar, dbar, sbar, c, cbar) has its own
independent set of parameters.

Contents of this file:
- Evolution factor module (fNP_evolution): Shared across PDFs and FFs
- TMD PDF class (TMDPDFBase): MAP22 parameterization with flavor-specific parameters
- TMD FF class (TMDFFBase): MAP22 parameterization with flavor-specific parameters
- Default parameter dictionaries: MAP22_DEFAULT_EVOLUTION,
                                  MAP22_DEFAULT_PDF_PARAMS,
                                  MAP22_DEFAULT_FF_PARAMS
- Manager class (fNPManager): Orchestrates the standard combo implementation for PDFs and FFs

Author: Chiara Bissolotti (cbissolotti@anl.gov)
Based on: MAP22g52.h from NangaParbat C++ implementation
"""

import torch
import torch.nn as nn
from typing import List, Optional, Dict, Any
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

        Initialization example:
        evolution = fNP_evolution(init_g2=0.12840, free_mask=[True])
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
        return self.fixed_g2 + self.free_g2

    def forward(self, b: torch.Tensor, zeta: torch.Tensor) -> torch.Tensor:
        """
        Compute evolution factor S_NP(ζ, b_T).

        Args:
            b (torch.Tensor): fourier-conjugate of k_T (GeV⁻¹) (can be 2D: [n_events, n_b])
            zeta (torch.Tensor): Rapidity scale ζ (GeV²) (1D: [n_events])

        Returns:
            torch.Tensor: Evolution factor exp[-g2² b_T²/4 x ln(ζ/Q₀²)]. Tensor
            has the same shape as b.
        """
        # Ensure zeta can broadcast with b
        # If b is 2D [n_events, n_b] and zeta is 1D [n_events], unsqueeze zeta
        if b.dim() > zeta.dim():
            zeta = zeta.unsqueeze(-1)
        return torch.exp(
            -(self.g2**2) * (b**2) * torch.log(zeta / self.Q0_squared) / 4.0
        )


###############################################################################
# 2. Base TMD PDF Class - Flavor-Dependent
###############################################################################
class TMDPDFBase(nn.Module):
    """
    Base TMD PDF class implementing the MAP22 parameterization for each flavor.

    This implements the exact MAP22 TMD PDF parameterization from the C++ code.
    """

    def __init__(self, n_flavors: int, init_params: List[float], free_mask: List[bool]):
        """
        Initialize TMD PDF base class with MAP22 parameterization.

        Args:
            n_flavors (int): Number of flavor instances sharing the same parameter set.
                           - n_flavors=1: Standard case - each flavor (u, d, s, etc.) gets its own
                             independent TMDPDFBase instance with independent parameters.
                           - n_flavors=2: Two flavors share the same parameter set and evolve together
                             during training (like a mini flavor-blind system for just 2 flavors).
                           - n_flavors>1: Useful if you want to group certain flavors to have identical
                             parameters while others remain independent.
            init_params (List[float]): 11 parameters [N₁, α₁, σ₁, λ, N₁ᵦ, N₁ᶜ, λ₂, α₂, α₃, σ₂, σ₃]
            free_mask (List[bool]): Trainability mask for each parameter

        Note:
            In the standard flavor-dependent system, the manager creates separate TMDPDFBase
            instances for each flavor (u, d, s, etc.) with n_flavors=1. This allows each flavor
            to have completely independent parameters. If you set n_flavors=2, you would have
            two flavors sharing the same parameter set, which would evolve together during
            optimization - this is useful for grouping flavors that should be parameterized
            identically (e.g., u and d quarks, or all sea quarks).
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

        # Set attributes
        self.n_flavors = n_flavors  # Number of flavors sharing this parameter set
        self.n_params = len(init_params)

        # Reference point x_hat = 0.1 (MAP22 standard)
        self.register_buffer("x_hat", torch.tensor(0.1, dtype=torch.float32))

        # Parameter setup with masking. We create a (F, P) tensor where:
        # - F = n_flavors: number of flavors sharing this parameter set
        # - P = number of parameters (11 for MAP22 PDF)
        #
        # IMPORTANT: When n_flavors=1 (standard case), each flavor gets its own independent
        # TMDPDFBase instance, so parameters are completely independent per flavor.
        # When n_flavors>1, multiple flavors share the same parameter set and evolve together.
        #
        # Example: If n_flavors=2, you might group u and d quarks to share parameters,
        # while s quark has its own separate instance with n_flavors=1.
        #
        # Take a 1-D parameter vector (e.g., initial parameters),
        # turn it into a row, and duplicate that row so each flavor in this group
        # starts with the same initial values.
        # .repeat(n_flavors, 1): repeats the single row n_flavors times along dim 0, and
        # 1 time along dim 1. Therefore, the final shape is (n_flavors, P).
        # .repeat copies memory; that's desired here because each flavor's params must be
        # independently trainable (even if they start identical, they can diverge during training).
        # If you only needed read-only broadcasting, .expand would avoid copies.
        init_tensor = (
            torch.tensor(init_params, dtype=torch.float32)  # shape: (P,)
            .unsqueeze(0)  # shape: (1, P)
            .repeat(n_flavors, 1)  # shape: (F, P)
        )

        # Mask of shape (1, P) to broadcast correctly
        mask = torch.tensor(free_mask, dtype=torch.float32).unsqueeze(0)
        # Registering mask as a buffer means it moves with the module (.to(device)),
        # is saved/loaded in the state_dict, but is not trainable.
        self.register_buffer("mask", mask)

        # Split parameters into fixed (masked-off) and free (masked-on) parts.
        # Both are (F, P): each flavor has its own copy of every parameter.
        # This is flavor-dependent: parameters are per flavor. Even though
        # they start identical (due to .repeat), each flavor’s row in
        # free_params is a distinct trainable vector that can diverge during training.
        #
        # Fixed parameters (non-trainable)
        fixed_init = init_tensor * (1 - mask)
        self.register_buffer("fixed_params", fixed_init)

        # Free parameters (trainable)
        free_init = init_tensor * mask
        self.free_params = nn.Parameter(free_init)

        # Gradient hook. Gradients are zeroed where mask == 0.
        self.free_params.register_hook(lambda grad: grad * self.mask)

    @property
    def get_params_tensor(self):
        """Return the full parameter tensor (fixed + free parts)."""
        return self.fixed_params + self.free_params

    def forward(
        self,
        x: torch.Tensor,
        b: torch.Tensor,
        NP_evol: torch.Tensor,
        flavor_idx: int = 0,
    ) -> torch.Tensor:
        """
        Compute TMD PDF using MAP22 parameterization.

        Args:
            x (torch.Tensor): Bjorken x variable
            b (torch.Tensor): fourier-conjugate of k_T (GeV⁻¹)
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
        result = numerator / denominator
        # result = NP_evol * numerator / denominator

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
    Base TMD FF class implementing the MAP22 FF parameterization for each flavor.
    The MAP22 FF parameterization is the same for all flavors, but the parameters
    are flavor-dependent.
    """

    def __init__(self, n_flavors: int, init_params: List[float], free_mask: List[bool]):
        """
        Initialize TMD FF base class with MAP22 FF parameterization for each flavor.

        Args:
            n_flavors (int): Number of flavor instances sharing the same parameter set.
                           - n_flavors=1: Standard case - each flavor (u, d, s, etc.) gets its own
                             independent TMDFFBase instance with independent parameters.
                           - n_flavors=2: Two flavors share the same parameter set and evolve together
                             during training (like a mini flavor-blind system for just 2 flavors).
                           - n_flavors>1: Useful if you want to group certain flavors to have identical
                             parameters while others remain independent.
            init_params (List[float]): 9 parameters [N₃, β₁, δ₁, γ₁, λ_F, N₃ᵦ, β₂, δ₂, γ₂]
            free_mask (List[bool]): Trainability mask for each parameter

        Note:
            In the standard flavor-dependent system, the manager creates separate TMDFFBase
            instances for each flavor (u, d, s, etc.) with n_flavors=1. This allows each flavor
            to have completely independent parameters. If you set n_flavors=2, you would have
            two flavors sharing the same parameter set, which would evolve together during
            optimization - this is useful for grouping flavors that should be parameterized
            identically (e.g., u and d quarks, or all sea quarks).
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

        self.n_flavors = n_flavors  # Number of flavors sharing this parameter set
        self.n_params = len(init_params)

        # Reference point z_hat = 0.5 (MAP22 standard)
        self.register_buffer("z_hat", torch.tensor(0.5, dtype=torch.float32))

        # Parameter setup with masking
        # Same logic as TMDPDFBase: creates (F, P) tensor where F=n_flavors, P=9 parameters
        # When n_flavors=1, each flavor gets its own independent instance.
        # When n_flavors>1, multiple flavors share parameters and evolve together.
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
        NP_evol: torch.Tensor,
        flavor_idx: int = 0,
    ) -> torch.Tensor:
        """
        Compute TMD FF using MAP22 parameterization.

        Args:
            z (torch.Tensor): Momentum fraction variable
            b (torch.Tensor): fourier-conjugate of P_T (GeV⁻¹)
            NP_evol (torch.Tensor): Evolution factor from fNP_evolution
            flavor_idx (int): Flavor index (typically 0)

        Returns:
            torch.Tensor: TMD FF D_NP(z, b)

        Note:
            MAP22 parameterization doesn't use zeta evolution, so it's removed
            from the interface for clarity.
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


###############################################################################
# 4. Manager Class
###############################################################################
class fNPManager(nn.Module):
    """
    Manager for standard (flavor-dependent) fNP system.

    This manager orchestrates the standard combo implementation for PDFs and FFs
    where each flavor has its own independent set of parameters.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the fNP manager with unified PDF and FF configurations for each flavor.

        Args:
            config (Dict): Configuration dictionary containing:
                - hadron: target hadron type
                - evolution: shared evolution parameters for all flavors
                - pdfs: PDF flavor configurations for each flavor
                - ffs: FF flavor configurations for each flavor
        """
        super().__init__()

        self.hadron = config.get("hadron", "proton")
        self.pdf_flavor_keys = ["u", "ubar", "d", "dbar", "s", "sbar", "c", "cbar"]
        self.ff_flavor_keys = ["u", "ubar", "d", "dbar", "s", "sbar", "c", "cbar"]

        # Print out messages to the user
        print(f"\033[94m\n[fNPManager] Initializing flavor-dependent fNP manager")
        print(f"  Hadron: {self.hadron}")
        print(f"  Total number of flavors: {len(self.pdf_flavor_keys)}\n\033[0m")

        # Setup evolution
        evolution_config = config.get("evolution", {})
        init_g2 = evolution_config.get("init_g2", 0.12840)
        free_mask = evolution_config.get("free_mask", [True])
        self.evolution = fNP_evolution(init_g2=init_g2, free_mask=free_mask)

        # Setup PDF modules
        pdf_config = config.get("pdfs", {})
        pdf_modules = {}
        for flavor in self.pdf_flavor_keys:
            flavor_cfg = pdf_config.get(flavor, None)
            if flavor_cfg is None:
                print(
                    f"\033[93m[fNPManager] Warning: Using MAP22 defaults for PDF flavor '{flavor}'\033[0m"
                )
                flavor_cfg = MAP22_DEFAULT_PDF_PARAMS.copy()
            else:
                print(
                    f"\033[34m[fNPManager] Using user-defined PDF flavor '{flavor}'\033[0m"
                )
            # Create TMD PDF module for this flavor
            pdf_modules[flavor] = TMDPDFBase(
                n_flavors=1,
                init_params=flavor_cfg["init_params"],
                free_mask=flavor_cfg["free_mask"],
            )
        # Register the PDF modules as a module dictionary
        self.pdf_modules = nn.ModuleDict(pdf_modules)
        print(
            f"\033[92m[fNPManager] Initialized {len(self.pdf_modules)} PDF flavor modules\n\033[0m"
        )

        # Setup FF modules
        ff_config = config.get("ffs", {})
        ff_modules = {}
        for flavor in self.ff_flavor_keys:
            flavor_cfg = ff_config.get(flavor, None)
            if flavor_cfg is None:
                print(
                    f"\033[93m[fNPManager] Warning: Using MAP22 defaults for FF flavor '{flavor}'\033[0m"
                )
            else:
                print(
                    f"\033[34m[fNPManager] Using user-defined FF flavor '{flavor}'\033[0m"
                )
            ff_modules[flavor] = TMDFFBase(
                n_flavors=1,
                init_params=flavor_cfg.get(
                    "init_params", MAP22_DEFAULT_FF_PARAMS["init_params"]
                ),
                free_mask=flavor_cfg.get(
                    "free_mask", MAP22_DEFAULT_FF_PARAMS["free_mask"]
                ),
            )
        self.ff_modules = nn.ModuleDict(ff_modules)
        print(
            f"\033[92m[fNPManager] Initialized {len(self.ff_modules)} FF flavor modules\n\033[0m"
        )

    def _compute_zeta(self, Q: torch.Tensor) -> torch.Tensor:
        """
        Compute rapidity scale zeta from hard scale Q.

        zeta = Q²

        This is a standard choice in the TMD framework.
        """
        return Q**2

    def forward_pdf(
        self,
        x: torch.Tensor,
        b: torch.Tensor,
        Q: torch.Tensor,
        flavors: Optional[List[str]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Evaluate TMD PDFs for specified flavors."""
        if flavors is None:
            flavors = self.pdf_flavor_keys

        zeta = self._compute_zeta(Q)
        shared_evol = self.evolution(b, zeta)

        outputs = {}
        for flavor in flavors:
            if flavor in self.pdf_modules:
                outputs[flavor] = self.pdf_modules[flavor](x, b, shared_evol, 0)
            else:
                raise ValueError(f"Unknown PDF flavor: {flavor}")

        # # Print out message to the user
        # print(
        #     f"\033[92m[fNPManager] Outputs for {len(outputs)} PDF flavors: {outputs[list(outputs.keys())]} (events, b_T, flavors)\n\033[0m"
        # )
        return outputs

    def forward_ff(
        self,
        z: torch.Tensor,
        b: torch.Tensor,
        Q: torch.Tensor,
        flavors: Optional[List[str]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Evaluate TMD FFs for specified flavors."""
        if flavors is None:
            flavors = self.ff_flavor_keys

        zeta = self._compute_zeta(Q)
        shared_evol = self.evolution(b, zeta)

        outputs = {}
        for flavor in flavors:
            if flavor in self.ff_modules:
                outputs[flavor] = self.ff_modules[flavor](z, b, shared_evol, 0)
            else:
                raise ValueError(f"Unknown FF flavor: {flavor}")
        return outputs

    def forward(
        self,
        x: torch.Tensor,
        z: torch.Tensor,
        b: torch.Tensor,
        Q: torch.Tensor,
        pdf_flavors: Optional[List[str]] = None,
        ff_flavors: Optional[List[str]] = None,
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Evaluate both TMD PDFs and FFs simultaneously.

        Args:
            x (torch.Tensor): Bjorken x values
            z (torch.Tensor): Energy fraction z values
            b (torch.Tensor): Impact parameter values
            Q (torch.Tensor): Hard scale Q values
            pdf_flavors (Optional[List[str]]): List of PDF flavors to evaluate
            ff_flavors (Optional[List[str]]): List of FF flavors to evaluate

        Returns:
            Dict containing 'pdfs' and 'ffs' sub-dictionaries with flavor results
        Note:
            The inputs are 1D tensors: [n_events]
            The outputs are 3D tensors: [n_events, b_T, n_flavors]
        """
        x = x.unsqueeze(-1)
        z = z.unsqueeze(-1)
        return {
            "pdfs": self.forward_pdf(x, b, Q, pdf_flavors),
            "ffs": self.forward_ff(z, b, Q, ff_flavors),
        }
