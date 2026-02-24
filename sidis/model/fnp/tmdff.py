import torch
import torch.nn as nn
from typing import List, Optional, Dict, Any, Union, Tuple

# Import tcolors - handle both relative and absolute imports
try:
    from ..utilities.colors import tcolors
except ImportError:
    try:
        from utilities.colors import tcolors
    except ImportError:
        from sidis.utilities.colors import tcolors

# Import config parsing utilities (shared across all models)
from .fnp_config import (
    ParameterLinkParser,
    ParameterRegistry,
    ExpressionEvaluator,
)

###############################################################################
# 6. Flexible FF Base Class
###############################################################################
class TMDFFFlexible(nn.Module):
    """
    Flexible TMD FF class with parameter linking support.
    """

    def __init__(
        self,
        flavor: str,
        init_params: List[float],
        free_mask: List[Any],
        registry: ParameterRegistry,
        evaluator: ExpressionEvaluator,
        param_type: str = "ffs",
    ):
        super().__init__()

        if len(init_params) != 9:
            raise ValueError(
                f"{tcolors.FAIL}[fnp_base_flexible.py] MAP22 TMD FF requires 9 parameters, got {len(init_params)}{tcolors.ENDC}"
            )
        if len(free_mask) != len(init_params):
            raise ValueError(
                f"{tcolors.FAIL}[fnp_base_flexible.py] free_mask length ({len(free_mask)}) must match init_params length ({len(init_params)}){tcolors.ENDC}"
            )

        self.flavor = flavor
        self.param_type = param_type
        self.n_params = len(init_params)
        self.registry = registry
        self.evaluator = evaluator
        self.parser = ParameterLinkParser()

        # Reference point z_hat = 0.5 (MAP22 standard)
        self.register_buffer("z_hat", torch.tensor(0.5, dtype=torch.float32))

        # Parse free_mask entries (same logic as PDF)
        self.param_configs = []
        self.fixed_params = []
        self.free_params_list = []

        for param_idx, (init_val, entry) in enumerate(zip(init_params, free_mask)):
            parsed = self.parser.parse_entry(entry, param_type, flavor)
            self.param_configs.append(
                {"idx": param_idx, "init_val": init_val, "parsed": parsed}
            )

            if parsed["is_fixed"]:
                self.fixed_params.append((param_idx, init_val))
            elif parsed["type"] == "boolean" and parsed["value"]:
                param = nn.Parameter(torch.tensor([init_val], dtype=torch.float32))
                self.free_params_list.append((param_idx, param))
                registry.register_parameter(param_type, flavor, param_idx, param)
            elif parsed["type"] == "reference":
                ref = parsed["value"]
                ref_type = ref["type"] if ref["type"] else param_type
                shared_param = registry.create_shared_parameter(
                    ref_type, ref["flavor"], ref["param_idx"], init_val
                )
                self.free_params_list.append((param_idx, shared_param))
                registry.register_parameter(
                    param_type,
                    flavor,
                    param_idx,
                    shared_param,
                    source=(ref_type, ref["flavor"], ref["param_idx"]),
                )
            elif parsed["type"] == "expression":
                param = nn.Parameter(torch.tensor([init_val], dtype=torch.float32))
                self.free_params_list.append((param_idx, param))
                registry.register_parameter(param_type, flavor, param_idx, param)
                parsed["expression"] = parsed["value"]

        # Register fixed parameters as buffers
        for param_idx, val in self.fixed_params:
            self.register_buffer(
                f"fixed_param_{param_idx}", torch.tensor([val], dtype=torch.float32)
            )

        # Register free parameters
        for param_idx, param in self.free_params_list:
            self.register_parameter(f"free_param_{param_idx}", param)

    def get_params_tensor(self) -> torch.Tensor:
        """Return the full parameter tensor, evaluating expressions dynamically."""
        params = [0.0] * self.n_params

        for param_idx, val in self.fixed_params:
            params[param_idx] = val

        for param_idx, param in self.free_params_list:
            config = self.param_configs[param_idx]
            parsed = config["parsed"]

            if parsed["type"] == "boolean" or parsed["type"] == "reference":
                if param.numel() == 1:
                    params[param_idx] = param.item()
                else:
                    params[param_idx] = (
                        param[0].item() if len(param.shape) > 0 else param.item()
                    )
            elif parsed["type"] == "expression":
                expr_value = self.evaluator.evaluate(
                    parsed["expression"], self.param_type, self.flavor
                )
                params[param_idx] = expr_value.item()
                param.data = expr_value

        return torch.tensor(params, dtype=torch.float32)

    def forward(
        self,
        z: torch.Tensor,
        b: torch.Tensor,
        flavor_idx: int = 0,
    ) -> torch.Tensor:
        """Compute TMD FF using MAP22 parameterization."""
        # Ensure z can broadcast with b (z: [n_events], b: [n_events, n_b])
        if b.dim() > z.dim():
            z = z.unsqueeze(-1)

        # Handle z >= 1 case (return zero)
        if torch.any(z >= 1):
            mask_val = (z < 1).type_as(b)
        else:
            mask_val = torch.ones_like(z)

        # Get parameters (evaluates expressions dynamically)
        p = self.get_params_tensor()

        # Extract parameters (MAP22 order)
        N3 = p[0]
        beta1 = p[1]
        delta1 = p[2]
        gamma1 = p[3]
        lambdaF = p[4]
        N3B = p[5]
        beta2 = p[6]
        delta2 = p[7]
        gamma2 = p[8]

        # Compute intermediate functions (MAP22 exact implementation)
        cmn1 = (
            (z**beta1 + delta1**2) / (self.z_hat**beta1 + delta1**2)
        ) * torch.pow((1 - z) / (1 - self.z_hat), gamma1**2)
        cmn2 = (
            (z**beta2 + delta2**2) / (self.z_hat**beta2 + delta2**2)
        ) * torch.pow((1 - z) / (1 - self.z_hat), gamma2**2)

        g3 = N3 * cmn1
        g3B = N3B * cmn2

        # z² factor
        z2 = z * z

        # Compute (b/2)² term
        b_half_sq = (b / 2.0) ** 2

        # Numerator (exact MAP22 formula)
        numerator = g3 * torch.exp(-g3 * b_half_sq / z2) + (lambdaF / z2) * (
            g3B**2
        ) * (1 - g3B * b_half_sq / z2) * torch.exp(-g3B * b_half_sq / z2)

        # Denominator (exact MAP22 formula)
        denominator = g3 + (lambdaF / z2) * (g3B**2)

        # Complete TMD FF (evolution factor applied in manager)
        result = numerator / denominator

        return result * mask_val
