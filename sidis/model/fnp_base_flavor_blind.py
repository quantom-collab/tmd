"""
Flavor-Blind fNP Implementation

This is a "combo" file that provides a specific combination of TMD PDF and TMD FF
non-perturbative parameterizations. This combo implements a flavor-blind system where
ALL flavors (u, d, s, ubar, dbar, sbar, c, cbar) share the exact same parameters and
evolve together. Unlike the flavor dependent system where each flavor has its own parameter set,
here all flavors use identical parameters that change simultaneously during optimization.

Content of this file:
- Evolution factor module (fNP_evolution): Shared across PDFs and FFs
- TMD PDF class (TMDPDFFlavorBlind):       MAP22 parameterization with shared parameters for all flavors
- TMD FF class (TMDFFFlavorBlind):         MAP22 parameterization with shared parameters for all flavors
- Default parameter dictionaries: MAP22_DEFAULT_EVOLUTION_FLAVOR_BLIND,
                                  MAP22_DEFAULT_PDF_PARAMS_FLAVOR_BLIND,
                                  MAP22_DEFAULT_FF_PARAMS_FLAVOR_BLIND.
- Manager class (fNPManager): Orchestrates the flavor-blind combo implementation,
                              managing the evolution, PDF, and FF modules all together.

Author: Chiara Bissolotti (cbissolotti@anl.gov)
Based on MAP22 parameterization from NangaParbat (MAP22g52.h)
"""

import torch
import torch.nn as nn
from typing import List, Dict, Any, Optional
import numpy as np

# Import tcolors - handle both relative and absolute imports
try:
    from ..utilities.colors import tcolors
except ImportError:
    try:
        from utilities.colors import tcolors
    except ImportError:
        from sidis.utilities.colors import tcolors


###############################################################################
# 1. Shared Evolution (same as standard fNP)
###############################################################################
class fNP_evolution(nn.Module):
    """
    Evolution factor S_NP(ζ, b_T) = exp[-g₂² b_T²/4 · ln(ζ/Q₀²)]

    This is identical to the standard fNP evolution since evolution is already
    shared across flavors in the standard implementation.
    """

    def __init__(self, init_g2: float = 0.12840, free_mask: List[bool] = [True]):
        """
        Initialize evolution module.

        Args:
            init_g2 (float): Initial value for g₂ parameter
            free_mask (List[bool]): Single-element list [True/False] for g2 trainability

        This method is the same as the fnp evolution module for the flavor-dependent case.
        This makes sense as the non-perturbative part of the evolution is not flavor dependent.
        """
        # First, call the constructor of the parent class (nn.Module)
        super().__init__()

        # Validate input
        if len(free_mask) != 1:
            raise ValueError(f"Evolution expects 1 mask element, got {len(free_mask)}")

        # Reference scale Q₀² = 1 GeV² (MAP22 standard)
        self.register_buffer("Q0_squared", torch.tensor(1.0, dtype=torch.float32))

        # Parameter setup with masking
        mask = torch.tensor(free_mask, dtype=torch.float32)
        self.register_buffer("mask", mask)

        # Fixed part
        fixed_init = torch.tensor([init_g2]) * (1 - mask)
        self.register_buffer("fixed_g2", fixed_init)

        # Free part (trainable)
        free_init = torch.tensor([init_g2]) * mask
        self.free_g2 = nn.Parameter(free_init)

        # Gradient hook
        self.free_g2.register_hook(lambda grad: grad * self.mask)

    @property
    def g2(self):
        """Return the full g₂ value (fixed + free parts)."""
        return self.fixed_g2 + self.free_g2

    def forward(self, b: torch.Tensor, zeta: torch.Tensor) -> torch.Tensor:
        """
        Compute evolution factor S_NP(ζ, b_T).

        This computes the non-perturbative evolution factor that appears in both
        TMD PDFs and FFs. The evolution factor depends on the rapidity scale zeta
        (computed from Q as zeta = Q²) and the impact parameter b.

        Formula: S_NP(ζ, b_T) = exp[-g₂² b_T²/4 × ln(ζ/Q₀²)]

        where:
        - g₂: evolution parameter (trainable)
        - b_T: impact parameter (GeV⁻¹)
        - ζ: rapidity scale (GeV²), computed from Q as ζ = Q²
        - Q₀²: reference scale = 1 GeV² (MAP22 standard)

        Args:
            b (torch.Tensor): b_T in GeV⁻¹ (2D tensor: [n_events, n_b])
            zeta (torch.Tensor): Rapidity scale ζ in GeV² (1D tensor: [n_events])
                                 Zeta is computed in the manager class based on the Q of each event
                                 and is then passed to this forward method.

        Returns:
            torch.Tensor: Evolution factor S_NP with shape [n_events, n_b] (same as b)
        """
        # Ensure zeta can broadcast with b
        # If b is 2D [n_events, n_b] and zeta is 1D [n_events], unsqueeze zeta
        if b.dim() > zeta.dim():
            zeta = zeta.unsqueeze(-1)
        return torch.exp(-(self.g2**2) * b**2 * torch.log(zeta / self.Q0_squared) / 4.0)


###############################################################################
# 2. Flavor-Blind TMD PDF Class
###############################################################################
class TMDPDFFlavorBlind(nn.Module):
    """
    Flavor-blind TMD PDF class where ALL flavors share identical parameters.

    In contrast to the standard TMDPDFBase where each flavor has its own parameter set,
    this class uses a single parameter set that applies to all flavors. All flavors
    evolve together when parameters are updated during optimization.

    The parameterization follows MAP22. The same parameters are used for u, d, s,
    ubar, dbar, sbar, etc.
    """

    def __init__(self, init_params: List[float], free_mask: List[bool]):
        """
        Initialize flavor-blind TMD PDF.

        Args:
            init_params (List[float]): 11 parameters [N₁, α₁, σ₁, λ, N₁ᵦ, N₁ᶜ, λ₂, α₂, α₃, σ₂, σ₃]
            free_mask (List[bool]): Trainability mask for each parameter
        """
        # First, call the constructor of the parent class (nn.Module)
        super().__init__()

        # Validate parameters
        if len(init_params) != 11:
            raise ValueError(
                f"{tcolors.FAIL}[fnp_base_flavor_blind.py] MAP22 TMD PDF requires 11 parameters, got {len(init_params)}{tcolors.ENDC}"
            )
        if len(free_mask) != len(init_params):
            raise ValueError(
                f"{tcolors.FAIL}[fnp_base_flavor_blind.py] free_mask length ({len(free_mask)}) must match init_params length ({len(init_params)}){tcolors.ENDC}"
            )

        self.n_params = len(init_params)

        # Reference point x_hat = 0.1 (MAP22 standard)
        self.register_buffer("x_hat", torch.tensor(0.1, dtype=torch.float32))

        # Parameter setup with masking - SINGLE SET FOR ALL FLAVORS.
        # As opposed to the flavor dependent case, TMDPDFBase, here
        # everything is 1-D: params are a single vector: shape (P,)
        # shared for all flavors
        mask = torch.tensor(free_mask, dtype=torch.float32)
        self.register_buffer("mask", mask)

        # Fixed parameters (non-trainable)
        fixed_init = torch.tensor(init_params) * (1 - mask)
        self.register_buffer("fixed_params", fixed_init)

        # Free parameters (trainable) - SHARED ACROSS ALL FLAVORS
        free_init = torch.tensor(init_params) * mask
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
    ) -> torch.Tensor:
        """
        Compute TMD PDF using MAP22 parameterization.

        This is a flavor-blind implementation where all flavors use identical parameters.
        At difference with the flavor-dependent implementation (TMDPDFBase), here the flavor
        index is not passed as an argument to the forward method, as all flavors share the
        same parameters and produce identical results.

        The computation follows the MAP22 formula:
        f_NP(x, b_T) = [numerator] / [denominator]

        where the numerator and denominator are computed from the shared parameters.
        Note: The non-perturbative evolution factor S_NP(ζ, b_T) is applied in the manager.

        Args:
            x (torch.Tensor): Bjorken x variable (can be 1D [n_events] or 2D [n_events, n_b])
            b (torch.Tensor): Fourier-conjugate of k_T (GeV⁻¹) (2D: [n_events, n_b])

        Returns:
            torch.Tensor: TMD PDF f_NP(x, b) without evolution factor - identical for all flavors.
                         Shape: [n_events, n_b] (same as b)

        Note:
            - Unlike TMDPDFBase, this flavor-blind version doesn't accept a flavor_idx parameter
              since all flavors share the same parameters
            - The evolution factor is computed and applied in the manager
            - All flavors (u, d, s, etc.) will produce identical results
        """
        # Ensure x can broadcast with b (x: [n_events], b: [n_events, n_b])
        if b.dim() > x.dim():
            x = x.unsqueeze(-1)

        # Handle x >= 1 case (return zero)
        if torch.any(x >= 1):
            mask_val = (x < 1).type_as(b)
        else:
            mask_val = torch.ones_like(x)

        # Extract parameters (same for ALL flavors)
        p = self.get_params_tensor
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
            + lam**2 * g1B**2 * (1 - g1B * b_half_sq) * torch.exp(-g1B * b_half_sq)
            + g1C * lam2**2 * torch.exp(-g1C * b_half_sq)
        )

        # Denominator (exact MAP22 formula)
        denominator = g1 + lam**2 * g1B**2 + g1C * lam2**2

        # Avoid division by zero
        denominator = torch.clamp(denominator, min=1e-12)

        # Final result with x-boundary handling (evolution factor applied in manager)
        result = (numerator / denominator) * mask_val

        return result


###############################################################################
# 2b. Flavor-Blind Sivers Function Class
###############################################################################
class SiversFlavorBlind(nn.Module):
    """
    Flavor-blind Sivers function class where ALL flavors share identical parameters.

    This is a simplified parameterization for the Sivers TMD PDF, used for
    transverse single-spin asymmetries. All flavors share the same g1_T parameter.

    The Sivers function formula:
        f_{1T}^{perp}(x, b_T) = S_NP(ζ, b_T) × exp(-g1_T × b_T²)
    """

    def __init__(self, g1_T: float = 0.045, free_mask: List[bool] = [True]):
        """
        Initialize flavor-blind Sivers function.

        Args:
            g1_T (float): Initial value for g1_T parameter
            free_mask (List[bool]): Single-element list [True/False] for g1_T trainability
        """
        super().__init__()

        # Validate input
        if len(free_mask) != 1:
            raise ValueError(f"Sivers expects 1 mask element, got {len(free_mask)}")

        self.n_params = 1

        # Parameter setup with masking
        mask = torch.tensor(free_mask, dtype=torch.float32)
        self.register_buffer("mask", mask)

        # Fixed part (non-trainable)
        fixed_init = torch.tensor([g1_T]) * (1 - mask)
        self.register_buffer("fixed_g1_T", fixed_init)

        # Free part (trainable)
        free_init = torch.tensor([g1_T]) * mask
        self.free_g1_T = nn.Parameter(free_init)

        # Gradient hook
        self.free_g1_T.register_hook(lambda grad: grad * self.mask)

    @property
    def g1_T(self):
        """Return the full g1_T value (fixed + free parts)."""
        return (self.fixed_g1_T + self.free_g1_T)[0]

    def forward(
        self,
        x: torch.Tensor,
        b: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute Sivers function using simplified parameterization.

        Formula: f_{1T}^{perp}(x, b_T) = exp(-g1_T × b_T²)
        Note: The non-perturbative evolution factor S_NP(ζ, b_T) is applied in the manager.

        Args:
            x (torch.Tensor): Bjorken x variable (can be 1D [n_events] or 2D [n_events, n_b])
            b (torch.Tensor): Impact parameter b_T (GeV⁻¹) (2D: [n_events, n_b])

        Returns:
            torch.Tensor: Sivers function f_{1T}^{perp}(x, b) without evolution factor - identical for all flavors.
                         Shape: [n_events, n_b] (same as b)
        """
        # Compute Sivers function: exp(-g1_T * b²) (evolution factor applied in manager)
        result = torch.exp(-self.g1_T * b * b)

        return result


###############################################################################
# 3. Flavor-Blind TMD FF Class
###############################################################################
class TMDFFFlavorBlind(nn.Module):
    """
    Flavor-blind TMD FF class where ALL flavors share identical parameters.

    Similar to TMDPDFFlavorBlind, this class uses a single parameter set for all
    fragmentation functions. The MAP22 parameterization is applied uniformly
    across all flavors.
    """

    def __init__(self, init_params: List[float], free_mask: List[bool]):
        """
        Initialize flavor-blind TMD FF.

        Args:
            init_params (List[float]): 9 parameters [N₃, β₁, δ₁, γ₁, λ_F, N₃ᵦ, β₂, δ₂, γ₂]
            free_mask (List[bool]): Trainability mask for each parameter
        """
        super().__init__()

        # Validate parameters
        if len(init_params) != 9:
            raise ValueError(
                f"[fnp_base_flavor_blind.py] MAP22 TMD FF requires 9 parameters, got {len(init_params)}"
            )
        if len(free_mask) != len(init_params):
            raise ValueError(
                f"[fnp_base_flavor_blind.py] free_mask length must match init_params length"
            )

        self.n_params = len(init_params)

        # Reference point z_hat = 0.5 (MAP22 standard)
        self.register_buffer("z_hat", torch.tensor(0.5, dtype=torch.float32))

        # Parameter setup with masking - SINGLE SET FOR ALL FLAVORS
        mask = torch.tensor(free_mask, dtype=torch.float32)
        self.register_buffer("mask", mask)

        # Fixed parameters (non-trainable)
        fixed_init = torch.tensor(init_params) * (1 - mask)
        self.register_buffer("fixed_params", fixed_init)

        # Free parameters (trainable) - SHARED ACROSS ALL FLAVORS
        free_init = torch.tensor(init_params) * mask
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
    ) -> torch.Tensor:
        """
        Compute TMD FF using MAP22 parameterization.

        This is a flavor-blind implementation where all flavors use identical parameters.
        At difference with the flavor-dependent implementation (TMDFFBase), here the flavor
        index is not passed as an argument to the forward method, as all flavors share the
        same parameters and produce identical results.

        The computation follows the MAP22 formula:
        D_NP(z, b_T) = [numerator] / [denominator]

        where the numerator and denominator are computed from the shared parameters, and
        includes z² factors specific to fragmentation functions.
        Note: The non-perturbative evolution factor S_NP(ζ, b_T) is applied in the manager.

        Args:
            z (torch.Tensor): Energy fraction z variable (can be 1D [n_events] or 2D [n_events, n_b])
            b (torch.Tensor): Impact parameter (GeV⁻¹) (2D: [n_events, n_b])

        Returns:
            torch.Tensor: TMD FF D_NP(z, b) without evolution factor - identical for all flavors.
                         Shape: [n_events, n_b] (same as b)

        Note:
            - Unlike TMDFFBase, this flavor-blind version doesn't accept a flavor_idx parameter
              since all flavors share the same parameters
            - The evolution factor is computed and applied in the manager
            - All flavors (u, d, s, etc.) will produce identical results
            - The FF parameterization includes z² factors that are important for fragmentation
        """
        # Ensure z can broadcast with b (z: [n_events], b: [n_events, n_b])
        if b.dim() > z.dim():
            z = z.unsqueeze(-1)

        # Handle z >= 1 case (return zero)
        if torch.any(z >= 1):
            mask_val = (z < 1).type_as(b)
        else:
            mask_val = torch.ones_like(z)

        # Extract parameters (same for ALL flavors)
        p = self.get_params_tensor
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
        numerator = g3 * torch.exp(-g3 * b_half_sq / z2) + (lambdaF / z2) * g3B**2 * (
            1 - g3B * b_half_sq / z2
        ) * torch.exp(-g3B * b_half_sq / z2)

        # Denominator (exact MAP22 formula)
        denominator = g3 + (lambdaF / z2) * g3B**2

        # Avoid division by zero
        denominator = torch.clamp(denominator, min=1e-12)

        # Final result with z-boundary handling (evolution factor applied in manager)
        result = (numerator / denominator) * mask_val

        return result


# Default parameter sets for flavor-blind system
MAP22_DEFAULT_EVOLUTION_FLAVOR_BLIND = {
    "init_g2": 0.12840,  # g2 from MAP22
    "free_mask": [True],
}

MAP22_DEFAULT_PDF_PARAMS_FLAVOR_BLIND = {
    "init_params": [
        0.28516,  # N₁
        2.9755,  # α₁
        0.17293,  # σ₁
        0.39432,  # λ
        0.28516,  # N₁ᵦ
        0.28516,  # N₁ᶜ
        0.39432,  # λ₂
        2.9755,  # α₂
        2.9755,  # α₃
        0.17293,  # σ₂
        0.17293,  # σ₃
    ],
    "free_mask": [True] * 11,  # All parameters trainable by default
}

MAP22_DEFAULT_FF_PARAMS_FLAVOR_BLIND = {
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
    "free_mask": [True] * 9,  # All parameters trainable by default
}


###############################################################################
# 4. Manager Class
###############################################################################
class fNPManager(nn.Module):
    """
    Manager for flavor-blind fNP system.

    This manager orchestrates the flavor-blind combo implementation where
    all flavors share identical parameters.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize flavor-blind fNP manager from configuration.

        This manager creates a single PDF module and a single FF module that serve
        all flavors. All flavors (u, d, s, ubar, dbar, sbar, c, cbar) share identical
        parameters and evolve together during optimization.

        Args:
            config (Dict[str, Any]): Configuration dictionary containing:
                - hadron: Hadron type (e.g., "proton")
                - evolution: Evolution configuration (shared across all flavors)
                - pdf: Single PDF configuration (applies to all flavors)
                - ff: Single FF configuration (applies to all flavors)

        Note:
            Unlike the flavor-dependent system where each flavor gets its own module,
            this flavor-blind system uses a single module for all flavors, dramatically
            reducing the parameter count from ~161 to 21 parameters.
        """
        super().__init__()

        self.config = config
        self.hadron = config.get("hadron", "proton")
        self.flavor_keys = ["u", "ubar", "d", "dbar", "s", "sbar", "c", "cbar"]

        # Print out messages to the user
        print(f"{tcolors.BLUE}\n[fNPManager] Initializing flavor-blind fNP manager")
        print(f"  Hadron: {self.hadron}")
        print(f"  Total number of flavors: {len(self.flavor_keys)}")
        print(f"  All flavors will share identical parameters\n{tcolors.ENDC}")

        # Setup evolution (shared across PDFs and FFs)
        evolution_config = config.get("evolution", {})
        if not evolution_config:
            print(
                f"{tcolors.WARNING}[fNPManager] Warning: Using MAP22 defaults for evolution parameters{tcolors.ENDC}"
            )
            evolution_config = MAP22_DEFAULT_EVOLUTION_FLAVOR_BLIND.copy()
        else:
            print(
                f"{tcolors.OKLIGHTBLUE}[fNPManager] Using user-defined evolution parameters{tcolors.ENDC}"
            )
        self.evolution = fNP_evolution(
            init_g2=evolution_config.get("init_g2", 0.12840),
            free_mask=evolution_config.get("free_mask", [True]),
        )
        print(
            f"{tcolors.GREEN}[fNPManager] Initialized shared evolution module{tcolors.ENDC}"
        )

        # Setup PDF module (single module serves all flavors)
        pdf_config = config.get("pdf", {})
        if not pdf_config:
            print(
                f"{tcolors.WARNING}[fNPManager] Warning: Using MAP22 defaults for PDF parameters (applies to all flavors){tcolors.ENDC}"
            )
            pdf_config = MAP22_DEFAULT_PDF_PARAMS_FLAVOR_BLIND.copy()
        else:
            print(
                f"{tcolors.OKLIGHTBLUE}[fNPManager] Using user-defined PDF parameters (applies to all flavors){tcolors.ENDC}"
            )
        self.pdf_module = TMDPDFFlavorBlind(
            init_params=pdf_config.get(
                "init_params", MAP22_DEFAULT_PDF_PARAMS_FLAVOR_BLIND["init_params"]
            ),
            free_mask=pdf_config.get(
                "free_mask", MAP22_DEFAULT_PDF_PARAMS_FLAVOR_BLIND["free_mask"]
            ),
        )
        print(
            f"{tcolors.GREEN}[fNPManager] Initialized single PDF module serving all {len(self.flavor_keys)} flavors\n{tcolors.ENDC}"
        )

        # Setup FF module (single module serves all flavors)
        ff_config = config.get("ff", {})
        if not ff_config:
            print(
                f"{tcolors.WARNING}[fNPManager] Warning: Using MAP22 defaults for FF parameters (applies to all flavors){tcolors.ENDC}"
            )
            ff_config = MAP22_DEFAULT_FF_PARAMS_FLAVOR_BLIND.copy()
        else:
            print(
                f"{tcolors.OKLIGHTBLUE}[fNPManager] Using user-defined FF parameters (applies to all flavors){tcolors.ENDC}"
            )
        self.ff_module = TMDFFFlavorBlind(
            init_params=ff_config.get(
                "init_params", MAP22_DEFAULT_FF_PARAMS_FLAVOR_BLIND["init_params"]
            ),
            free_mask=ff_config.get(
                "free_mask", MAP22_DEFAULT_FF_PARAMS_FLAVOR_BLIND["free_mask"]
            ),
        )
        print(
            f"{tcolors.GREEN}[fNPManager] Initialized single FF module serving all {len(self.flavor_keys)} flavors\n{tcolors.ENDC}"
        )

        # Setup Sivers module (flavor-blind)
        g1_T = evolution_config.get("g1_T", 0.045)
        self.sivers_module = SiversFlavorBlind(g1_T=g1_T, free_mask=[True])
        print(
            f"{tcolors.GREEN}[fNPManager] Initialized Sivers module with g1_T={g1_T}{tcolors.ENDC}"
        )

    def _compute_zeta(self, Q: torch.Tensor) -> torch.Tensor:
        """
        Compute rapidity scale zeta from hard scale Q.

        ZETA COMPUTATION: zeta = Q² (standard SIDIS)

        This is the standard choice for SIDIS processes. The rapidity scale zeta
        is computed event-by-event from the hard scale Q. If you need a different
        formula (e.g., zeta = Q² * z), modify this method.

        Args:
            Q (torch.Tensor): Hard scale Q in GeV (1D tensor: [n_events])

        Returns:
            torch.Tensor: Rapidity scale zeta = Q² in GeV² (1D tensor: [n_events])
        """
        return Q**2

    def forward_pdf(
        self,
        x: torch.Tensor,
        b: torch.Tensor,
        Q: torch.Tensor,
        flavors: Optional[List[str]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Evaluate TMD PDFs for specified flavors using shared parameters.

        In the flavor-blind system, all flavors share the same parameter set, so
        the PDF computation is performed once and the same result is returned for
        all requested flavors.

        Args:
            x (torch.Tensor): Bjorken x values (1D: [n_events])
            b (torch.Tensor): Impact parameter values (2D: [n_events, n_b])
            Q (torch.Tensor): Hard scale Q in GeV (1D: [n_events], used to compute zeta = Q²)
            flavors (Optional[List[str]]): List of flavors to evaluate (None = all flavors)

        Returns:
            Dict[str, torch.Tensor]: PDF values for each requested flavor.
                                    All flavors have identical values since they share parameters.
                                    Shape: [n_events, n_b] for each flavor
        """
        if flavors is None:
            flavors = self.flavor_keys

        # Compute zeta from Q (zeta = Q² for standard SIDIS)
        zeta = self._compute_zeta(Q)

        # Compute shared evolution factor (same for all flavors)
        shared_evol = self.evolution(b, zeta)

        # Evaluate PDF using shared parameters (same result for all flavors)
        # Note: NP_evol is now applied in the manager, not in the module
        shared_pdf_result = self.pdf_module(x, b)

        # Apply evolution factor to the result
        shared_pdf_result = shared_pdf_result * shared_evol

        # Return the same result for all requested flavors
        outputs = {}
        for flavor in flavors:
            if flavor in self.flavor_keys:
                outputs[flavor] = shared_pdf_result
            else:
                raise ValueError(f"Unknown PDF flavor: {flavor}")

        # # Print out message to the user
        # print(
        #     f"{tcolors.GREEN}[fNPManager] Outputs for {len(outputs)} PDF flavors: all identical (flavor-blind)\n{tcolors.ENDC}"
        # )
        return outputs

    def forward_ff(
        self,
        z: torch.Tensor,
        b: torch.Tensor,
        Q: torch.Tensor,
        flavors: Optional[List[str]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Evaluate TMD FFs for specified flavors using shared parameters.

        In the flavor-blind system, all flavors share the same parameter set, so
        the FF computation is performed once and the same result is returned for
        all requested flavors.

        Args:
            z (torch.Tensor): Energy fraction z values (1D: [n_events])
            b (torch.Tensor): Impact parameter values (2D: [n_events, n_b])
            Q (torch.Tensor): Hard scale Q in GeV (1D: [n_events], used to compute zeta = Q²)
            flavors (Optional[List[str]]): List of flavors to evaluate (None = all flavors)

        Returns:
            Dict[str, torch.Tensor]: FF values for each requested flavor.
                                    All flavors have identical values since they share parameters.
                                    Shape: [n_events, n_b] for each flavor
        """
        if flavors is None:
            flavors = self.flavor_keys

        # Compute zeta from Q (zeta = Q² for standard SIDIS)
        zeta = self._compute_zeta(Q)

        # Compute shared evolution factor (same for all flavors)
        shared_evol = self.evolution(b, zeta)

        # Evaluate FF using shared parameters (same result for all flavors)
        # Note: NP_evol is now applied in the manager, not in the module
        shared_ff_result = self.ff_module(z, b)

        # Apply evolution factor to the result
        shared_ff_result = shared_ff_result * shared_evol

        # Return the same result for all requested flavors
        outputs = {}
        for flavor in flavors:
            if flavor in self.flavor_keys:
                outputs[flavor] = shared_ff_result
            else:
                raise ValueError(f"Unknown FF flavor: {flavor}")
        return outputs

    def forward_sivers(
        self,
        x: torch.Tensor,
        b: torch.Tensor,
        Q: torch.Tensor,
    ) -> torch.Tensor:
        """
        Evaluate Sivers function using shared parameters.

        In the flavor-blind system, all flavors share the same Sivers parameters,
        so the computation returns a single result for all flavors.

        Args:
            x (torch.Tensor): Bjorken x values (1D: [n_events])
            b (torch.Tensor): Impact parameter values (2D: [n_events, n_b])
            Q (torch.Tensor): Hard scale Q in GeV (1D: [n_events], used to compute zeta = Q²)

        Returns:
            torch.Tensor: Sivers function values - identical for all flavors.
                         Shape: [n_events, n_b] (same as b)
        """
        # Compute zeta from Q (zeta = Q² for standard SIDIS)
        zeta = self._compute_zeta(Q)

        # Compute shared evolution factor
        shared_evol = self.evolution(b, zeta)

        # Evaluate Sivers using shared parameters
        # Note: NP_evol is now applied in the manager, not in the module
        sivers_result = self.sivers_module(x, b)

        # Apply evolution factor to the result
        sivers_result = sivers_result * shared_evol

        return sivers_result

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
        Evaluate both TMD PDFs and FFs simultaneously using shared parameters.

        This method computes both PDFs and FFs in a single call. In the flavor-blind
        system, all flavors share identical parameters, so the computation is performed
        once and the same result is returned for all flavors.

        Args:
            x (torch.Tensor): Bjorken x values (1D: [n_events])
            z (torch.Tensor): Energy fraction z values (1D: [n_events])
            b (torch.Tensor): Impact parameter values (2D: [n_events, n_b])
            Q (torch.Tensor): Hard scale Q in GeV (1D: [n_events], used to compute zeta = Q²)
            pdf_flavors (Optional[List[str]]): List of PDF flavors to evaluate (None = all flavors)
            ff_flavors (Optional[List[str]]): List of FF flavors to evaluate (None = all flavors)

        Returns:
            Dict containing 'pdfs' and 'ffs' sub-dictionaries with flavor results.
            Structure: {
                "pdfs": {flavor: tensor, ...},  # All flavors have identical values
                "ffs": {flavor: tensor, ...}   # All flavors have identical values
            }

        Note:
            The inputs are 1D tensors: [n_events] for x, z, Q
            The outputs are 2D tensors: [n_events, n_b] for each flavor
            All flavors have identical values since they share parameters (flavor-blind system)
        """
        return {
            "pdfs": self.forward_pdf(x, b, Q, pdf_flavors),
            "ffs": self.forward_ff(z, b, Q, ff_flavors),
        }
