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
# 1. Flexible PDF Base Class
###############################################################################
class TMDPDFFlexible(nn.Module):
    """
    Flexible TMD PDF class with parameter linking support.
    """

    def __init__(
        self,
        flavor: str,
        init_params: List[float],
        free_mask: List[Any],
        registry: ParameterRegistry,
        evaluator: ExpressionEvaluator,
        param_type: str = "pdfs",
    ):
        super().__init__()

        if len(init_params) != 11:
            raise ValueError(
                f"{tcolors.FAIL}[fnp_base_flexible.py] MAP22 TMD PDF requires 11 parameters, got {len(init_params)}{tcolors.ENDC}"
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

        # Reference point x_hat = 0.1 (MAP22 standard)
        self.register_buffer("x_hat", torch.tensor(0.1, dtype=torch.float32))

        # Parse free_mask entries
        self.param_configs = []
        self.fixed_params = []
        self.free_params_list = []

        for param_idx, (init_val, entry) in enumerate(zip(init_params, free_mask)):
            parsed = self.parser.parse_entry(entry, param_type, flavor)
            self.param_configs.append(
                {"idx": param_idx, "init_val": init_val, "parsed": parsed}
            )

            if parsed["is_fixed"]:
                # Fixed parameter
                self.fixed_params.append((param_idx, init_val))
            elif parsed["type"] == "boolean" and parsed["value"]:
                # Independent free parameter
                param = nn.Parameter(torch.tensor([init_val], dtype=torch.float32))
                self.free_params_list.append((param_idx, param))
                registry.register_parameter(param_type, flavor, param_idx, param)
            elif parsed["type"] == "reference":
                # Linked parameter - use shared parameter
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
                # Expression-based parameter - will be evaluated dynamically
                # Store expression and create a placeholder parameter for gradient flow
                param = nn.Parameter(torch.tensor([init_val], dtype=torch.float32))
                self.free_params_list.append((param_idx, param))
                registry.register_parameter(param_type, flavor, param_idx, param)
                # Store expression for dynamic evaluation
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

        # Set fixed parameters
        for param_idx, val in self.fixed_params:
            params[param_idx] = val

        # Set free parameters (including linked and expression-based)
        for param_idx, param in self.free_params_list:
            config = self.param_configs[param_idx]
            parsed = config["parsed"]

            if parsed["type"] == "boolean" or parsed["type"] == "reference":
                # Direct or linked parameter
                if param.numel() == 1:
                    params[param_idx] = param.item()
                else:
                    params[param_idx] = (
                        param[0].item() if len(param.shape) > 0 else param.item()
                    )
            elif parsed["type"] == "expression":
                # Evaluate expression dynamically
                expr_value = self.evaluator.evaluate(
                    parsed["expression"], self.param_type, self.flavor
                )
                params[param_idx] = expr_value.item()
                # Update the parameter for gradient tracking
                param.data = expr_value

        return torch.tensor(params, dtype=torch.float32)

    def forward(
        self,
        x: torch.Tensor,
        b: torch.Tensor,
        flavor_idx: int = 0,
    ) -> torch.Tensor:
        """Compute TMD PDF using MAP22 parameterization."""
        # Ensure x can broadcast with b (x: [n_events], b: [n_events, n_b])
        if b.dim() > x.dim():
            x = x.unsqueeze(-1)

        # Handle x >= 1 case (return zero)
        if torch.any(x >= 1):
            mask_val = (x < 1).type_as(b)
        else:
            mask_val = torch.ones_like(x)

        # Get parameters (evaluates expressions dynamically)
        p = self.get_params_tensor()

        # Extract parameters (MAP22 order)
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

        # Complete TMD PDF (evolution factor applied in manager)
        result = numerator / denominator

        return result * mask_val

