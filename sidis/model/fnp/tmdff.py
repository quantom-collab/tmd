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
from ..fnp_config import (
    ParameterLinkParser,
    ParameterRegistry,
    ExpressionEvaluator,
    parse_bound,
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
        param_bounds: Optional[List[Any]] = None,
        param_bounds_map: Optional[Dict[Tuple[str, str, int], Tuple[float, float]]] = None,
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
        param_bounds_map: Optional[Dict[Tuple[str, str, int], Tuple[float, float]]] = None,
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
        self.parser = ParameterLinkParser()

        bounds_list = []
        if param_bounds is not None:
            try:
                for idx in range(min(len(param_bounds), 2)):
                    b = parse_bound(param_bounds[idx] if idx < len(param_bounds) else None)
                    bounds_list.append(b)
            except (TypeError, KeyError):
                pass
        while len(bounds_list) < 2:
            bounds_list.append(None)

        self.param_configs = []
        self.fixed_params = []
        self.free_params_list = []

        for param_idx, (init_val, entry) in enumerate(zip(init_params, free_mask)):
            parsed = self.parser.parse_entry(entry, param_type, flavor)
            bounds = bounds_list[param_idx] if param_idx < len(bounds_list) else None
            self.param_configs.append(
                {"idx": param_idx, "init_val": init_val, "parsed": parsed, "bounds": bounds}
            )

            if parsed["is_fixed"]:
                self.fixed_params.append((param_idx, init_val))
            elif parsed["type"] == "boolean" and parsed["value"]:
                if bounds is not None:
                    lo, hi = bounds
                    u = (init_val - lo) / (hi - lo)
                    u = max(1e-6, min(1 - 1e-6, u))
                    theta = torch.tensor(
                        float(torch.logit(torch.tensor(u)).item()), dtype=torch.float32
                    )
                    param = nn.Parameter(theta.unsqueeze(0))
                else:
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

        for param_idx, val in self.fixed_params:
            self.register_buffer(
                f"fixed_param_{param_idx}", torch.tensor([val], dtype=torch.float32)
            )
        for param_idx, param in self.free_params_list:
            self.register_parameter(f"free_param_{param_idx}", param)

    def get_params_tensor(self) -> torch.Tensor:
        """Return parameter tensor while preserving gradients for trainable params."""
        try:
            dev = next(self.parameters()).device
        except StopIteration:
            try:
                dev = next(self.buffers()).device
            except StopIteration:
                dev = torch.device("cpu")

        param_vals = [None] * self.n_params

        for param_idx, val in self.fixed_params:
            param_vals[param_idx] = torch.tensor([float(val)], dtype=torch.float32, device=dev)

        for param_idx, param in self.free_params_list:
            config = self.param_configs[param_idx]
            parsed = config["parsed"]
            if parsed["type"] == "boolean" or parsed["type"] == "reference":
                bounds = config.get("bounds")
                if bounds is None and parsed["type"] == "reference":
                    ref = parsed["value"]
                    ref_type = ref["type"] if ref["type"] else self.param_type
                    key = (ref_type, ref["flavor"], ref["param_idx"])
                    bounds = self.param_bounds_map.get(key)
                if bounds is not None:
                    lo, hi = bounds
                    raw = torch.sigmoid(param)
                    val_t = lo + (hi - lo) * raw.flatten()[0]
                    param_vals[param_idx] = val_t.unsqueeze(0)
                else:
                    p = param.flatten()[0]
                    param_vals[param_idx] = p.unsqueeze(0)
            elif parsed["type"] == "expression":
                expr_value = self.evaluator.evaluate(
                    parsed["expression"], self.param_type, self.flavor
                )
                param_vals[param_idx] = expr_value
                param.data = expr_value.detach()

        vals = [
            param_vals[i]
            if param_vals[i] is not None
            else torch.tensor([0.0], dtype=torch.float32, device=dev)
            for i in range(self.n_params)
        ]
        return torch.cat([v.flatten()[:1] for v in vals])

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
        exponent = -((b / 2.0) ** 2) * (lam_D**2) * torch.pow(z_safe, beta) * ((1 - z) ** 2)
        return torch.exp(exponent) * mask_val
