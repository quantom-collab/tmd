import torch
import torch.nn as nn
from typing import List, Optional, Dict, Any, Tuple

# Import tcolors - handle both relative and absolute imports
try:
    from ..utilities.colors import tcolors
except ImportError:
    try:
        from utilities.colors import tcolors
    except ImportError:
        from sidis.utilities.colors import tcolors

from ..fnp_config import ExpressionEvaluator, ParameterRegistry
from ..fnp_linked_params import (
    build_bounds_list,
    get_params_tensor_from_state,
    populate_linked_params,
)


###############################################################################
# 1. Flexible PDF Base Class
###############################################################################
class TMDPDFFlexible(nn.Module):
    """
    Flexible TMD PDF class with parameter linking support.
    it implements the MAP22 parameterization of the TMD PDF.
    Parameters:
      [N₁, α₁, σ₁, λ, N₁ᵦ, N₁ᶜ, λ₂, α₂, α₃, σ₂, σ₃] (11 params)
    """

    def __init__(
        self,
        flavor: str,
        init_params: List[float],
        free_mask: List[Any],
        registry: ParameterRegistry,
        evaluator: ExpressionEvaluator,
        param_type: str = "pdfs",
        param_bounds: Optional[List[Any]] = None,
        param_bounds_map: Optional[
            Dict[Tuple[str, str, int], Tuple[float, float]]
        ] = None,
    ):
        super().__init__()
        self.param_bounds_map = param_bounds_map or {}

        if len(init_params) != 11:
            raise ValueError(
                f"{tcolors.FAIL}[fnp/tmdpdf.py] MAP22 TMD PDF requires 11 parameters, got {len(init_params)}{tcolors.ENDC}"
            )
        if len(free_mask) != len(init_params):
            raise ValueError(
                f"{tcolors.FAIL}[fnp/tmdpdf.py] free_mask length ({len(free_mask)}) must match init_params length ({len(init_params)}){tcolors.ENDC}"
            )

        self.flavor = flavor
        self.param_type = param_type
        self.n_params = len(init_params)
        self.registry = registry
        self.evaluator = evaluator

        self.register_buffer("x_hat", torch.tensor(0.1, dtype=torch.float32))

        bounds_list = build_bounds_list(
            param_bounds,
            self.n_params,
            self.param_type,
            self.flavor,
            warn_tag="[fnp/tmdpdf.py]",
        )
        populate_linked_params(
            self,
            flavor=flavor,
            param_type=param_type,
            init_params=init_params,
            free_mask=free_mask,
            registry=registry,
            bounds_list=bounds_list,
        )

    def get_params_tensor(self) -> torch.Tensor:
        """Return parameter tensor while preserving gradients for trainable params."""
        return get_params_tensor_from_state(self)

    def forward(
        self,
        x: torch.Tensor,
        b: torch.Tensor,
        flavor_idx: int = 0,
    ) -> torch.Tensor:
        """Compute TMD PDF using MAP22 parameterization."""
        if b.dim() > x.dim():
            x = x.unsqueeze(-1)

        if torch.any(x >= 1):
            mask_val = (x < 1).type_as(b)
        else:
            mask_val = torch.ones_like(x)

        p = self.get_params_tensor()

        N1 = p[0]
        alpha1 = p[1]
        sigma1 = p[2]
        lam = p[3]
        N1B = p[4]
        N1C = p[5]
        lam2 = p[6]
        alpha2 = p[7]
        alpha3 = p[8]
        sigma2 = p[9]
        sigma3 = p[10]

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

        b_half_sq = (b / 2.0) ** 2

        numerator = (
            g1 * torch.exp(-g1 * b_half_sq)
            + (lam**2) * (g1B**2) * (1 - g1B * b_half_sq) * torch.exp(-g1B * b_half_sq)
            + g1C * (lam2**2) * torch.exp(-g1C * b_half_sq)
        )

        denominator = g1 + (lam**2) * (g1B**2) + g1C * (lam2**2)

        result = numerator / denominator

        return result * mask_val


###############################################################################
# 2. Simple Exponential PDF
###############################################################################
class TMDPDFSimple(nn.Module):
    """
    Simple exponential PDF parametrization with linking support.

    Formula:
      f_NP(x, b) = exp[-(b/2)^2 * lambda_f^2 * x^alpha * (1-x)^2]
    Parameters:
      [lambda_f, alpha]
    """

    def __init__(
        self,
        flavor: str,
        init_params: List[float],
        free_mask: List[Any],
        registry: ParameterRegistry,
        evaluator: ExpressionEvaluator,
        param_type: str = "pdfs",
        param_bounds: Optional[List[Any]] = None,
        param_bounds_map: Optional[
            Dict[Tuple[str, str, int], Tuple[float, float]]
        ] = None,
    ):
        super().__init__()
        self.param_bounds_map = param_bounds_map or {}

        if len(init_params) != 2:
            raise ValueError(
                f"{tcolors.FAIL}[tmdpdf.py] Simple PDF requires 2 params [lambda_f, alpha], got {len(init_params)}{tcolors.ENDC}"
            )
        if len(free_mask) != 2:
            raise ValueError(
                f"{tcolors.FAIL}[tmdpdf.py] free_mask length must match init_params (2){tcolors.ENDC}"
            )

        self.flavor = flavor
        self.param_type = param_type
        self.n_params = len(init_params)
        self.registry = registry
        self.evaluator = evaluator

        bounds_list = build_bounds_list(
            param_bounds,
            self.n_params,
            self.param_type,
            self.flavor,
            warn_tag="[tmdpdf.py]",
        )
        populate_linked_params(
            self,
            flavor=flavor,
            param_type=param_type,
            init_params=init_params,
            free_mask=free_mask,
            registry=registry,
            bounds_list=bounds_list,
        )

    def get_params_tensor(self) -> torch.Tensor:
        """Return parameter tensor while preserving gradients for trainable params."""
        return get_params_tensor_from_state(self)

    def forward(
        self,
        x: torch.Tensor,
        b: torch.Tensor,
        flavor_idx: int = 0,
    ) -> torch.Tensor:
        if b.dim() > x.dim():
            x = x.unsqueeze(-1)
        if torch.any(x >= 1):
            mask_val = (x < 1).type_as(b)
        else:
            mask_val = torch.ones_like(x)

        p = self.get_params_tensor()
        lam_f, alpha = p[0], p[1]
        x_safe = torch.clamp(x, min=1e-10)
        exponent = (
            -((b / 2.0) ** 2) * (lam_f**2) * torch.pow(x_safe, alpha) * ((1 - x) ** 2)
        )
        return torch.exp(exponent) * mask_val
