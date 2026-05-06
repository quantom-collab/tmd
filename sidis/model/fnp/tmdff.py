import torch
import torch.nn as nn
from typing import List, Optional, Dict, Any, Tuple

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
# 6. Flexible FF Base Class
###############################################################################
class TMDFFFlexible(nn.Module):
    """
    Flexible TMD FF class with parameter linking support.
    it implements the MAP22 parameterization of the TMD FF.
    Parameters:
      [N₃, β₁, δ₁, γ₁, λ_F, N₃_B, β₂, δ₂, γ₂] (9 params)
    """

    def __init__(
        self,
        flavor: str,
        init_params: List[float],
        free_mask: List[Any],
        registry: ParameterRegistry,
        evaluator: ExpressionEvaluator,
        param_type: str = "ffs",
        param_bounds: Optional[List[Any]] = None,
        param_bounds_map: Optional[
            Dict[Tuple[str, str, int], Tuple[float, float]]
        ] = None,
    ):
        super().__init__()

        if len(init_params) != 9:
            raise ValueError(
                f"{tcolors.FAIL}[fnp/tmdff.py] MAP22 TMD FF requires 9 parameters, got {len(init_params)}{tcolors.ENDC}"
            )
        if len(free_mask) != len(init_params):
            raise ValueError(
                f"{tcolors.FAIL}[fnp/tmdff.py] free_mask length ({len(free_mask)}) must match init_params length ({len(init_params)}){tcolors.ENDC}"
            )
        self.param_bounds_map = param_bounds_map or {}

        self.flavor = flavor
        self.param_type = param_type
        self.n_params = len(init_params)
        self.registry = registry
        self.evaluator = evaluator

        self.register_buffer("z_hat", torch.tensor(0.5, dtype=torch.float32))

        bounds_list = build_bounds_list(
            param_bounds,
            self.n_params,
            self.param_type,
            self.flavor,
            warn_tag="[fnp/tmdff.py]",
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
        z: torch.Tensor,
        b: torch.Tensor,
        flavor_idx: int = 0,
    ) -> torch.Tensor:
        """Compute TMD FF using MAP22 parameterization."""
        if b.dim() > z.dim():
            z = z.unsqueeze(-1)

        if torch.any(z >= 1):
            mask_val = (z < 1).type_as(b)
        else:
            mask_val = torch.ones_like(z)

        p = self.get_params_tensor()

        N3 = p[0]
        beta1 = p[1]
        delta1 = p[2]
        gamma1 = p[3]
        lambdaF = p[4]
        N3B = p[5]
        beta2 = p[6]
        delta2 = p[7]
        gamma2 = p[8]

        cmn1 = (
            (z**beta1 + delta1**2) / (self.z_hat**beta1 + delta1**2)
        ) * torch.pow((1 - z) / (1 - self.z_hat), gamma1**2)
        cmn2 = (
            (z**beta2 + delta2**2) / (self.z_hat**beta2 + delta2**2)
        ) * torch.pow((1 - z) / (1 - self.z_hat), gamma2**2)

        g3 = N3 * cmn1
        g3B = N3B * cmn2

        z2 = z * z

        b_half_sq = (b / 2.0) ** 2

        numerator = g3 * torch.exp(-g3 * b_half_sq / z2) + (lambdaF / z2) * (
            g3B**2
        ) * (1 - g3B * b_half_sq / z2) * torch.exp(-g3B * b_half_sq / z2)

        denominator = g3 + (lambdaF / z2) * (g3B**2)

        result = numerator / denominator

        return result * mask_val


###############################################################################
# 7. Simple Exponential FF with linking
###############################################################################
class TMDFFSimple(nn.Module):
    """
    Simple exponential FF parametrization with linking support.

    Formula:
      D_NP(z, b) = exp[-(b/2)^2 * lambda_D^2 * z^beta * (1-z)^2]
    Parameters:
      [lambda_D, beta]
    """

    def __init__(
        self,
        flavor: str,
        init_params: List[float],
        free_mask: List[Any],
        registry: ParameterRegistry,
        evaluator: ExpressionEvaluator,
        param_type: str = "ffs",
        param_bounds: Optional[List[Any]] = None,
        param_bounds_map: Optional[
            Dict[Tuple[str, str, int], Tuple[float, float]]
        ] = None,
    ):
        super().__init__()
        self.param_bounds_map = param_bounds_map or {}

        if len(init_params) != 2:
            raise ValueError(
                f"{tcolors.FAIL}[tmdff.py] Simple FF requires 2 params [lambda_D, beta], got {len(init_params)}{tcolors.ENDC}"
            )
        if len(free_mask) != 2:
            raise ValueError(
                f"{tcolors.FAIL}[tmdff.py] free_mask length must match init_params (2){tcolors.ENDC}"
            )

        self.flavor = flavor
        self.param_type = param_type
        self.n_params = 2
        self.registry = registry
        self.evaluator = evaluator

        bounds_list = build_bounds_list(
            param_bounds,
            self.n_params,
            self.param_type,
            self.flavor,
            warn_tag="[tmdff.py]",
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
        z: torch.Tensor,
        b: torch.Tensor,
        flavor_idx: int = 0,
    ) -> torch.Tensor:
        if b.dim() > z.dim():
            z = z.unsqueeze(-1)
        if torch.any(z >= 1):
            mask_val = (z < 1).type_as(b)
        else:
            mask_val = torch.ones_like(z)

        p = self.get_params_tensor()
        lam_D, beta = p[0], p[1]
        z_safe = torch.clamp(z, min=1e-10)
        exponent = (
            -((b / 2.0) ** 2) * (lam_D**2) * torch.pow(z_safe, beta) * ((1 - z) ** 2)
        )
        return torch.exp(exponent) * mask_val
