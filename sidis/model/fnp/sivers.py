# Here we will implement the Sivers function fNP

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
    parse_bound
)

class Sivers(nn.Module):
    def __init__(
        self, 
        flavor: str,
        init_params: List[float], 
        free_mask: List[Any],
        registry: ParameterRegistry,
        evaluator: ExpressionEvaluator,
        param_type: str = "sivers",
        param_bounds: Optional[List[Any]] = None,
        param_bounds_map: Optional[Dict[Tuple[str, str, int], Tuple[float, float]]] = None, 

        ):

        super().__init__()
                
        if len(init_params) != 1:
            raise ValueError(
                f"{tcolors.FAIL}[fnp/sivers.py] MAP22 Sivers requires 1 parameter, got {len(init_params)}{tcolors.ENDC}. This is the Sivers Class (non AV)."
            )
        if len(free_mask) != len(init_params):
            raise ValueError(
                f"{tcolors.FAIL}[fnp/sivers.py] free_mask length ({len(free_mask)}) must match init_params length ({len(init_params)}){tcolors.ENDC}"
            )
        self.param_bounds_map = param_bounds_map or {}
        self.flavor = flavor
        self.param_type = param_type
        self.n_params = len(init_params)
        self.registry = registry
        self.evaluator = evaluator
        self.parser = ParameterLinkParser()

        
        # Parse free_mask entries
        self.param_configs = []
        self.fixed_params = []
        self.free_params_list = []
        bounds_list = []

        if param_bounds is not None:
            try:
                for idx in range(min(len(param_bounds), self.n_params)):
                    b = parse_bound(param_bounds[idx] if idx < len(param_bounds) else None)
                    bounds_list.append(b)
            except (TypeError, KeyError):
                pass
        while len(bounds_list) < self.n_params:
            bounds_list.append(None)

        for param_idx, (init_val, entry) in enumerate(zip(init_params, free_mask)):
            parsed = self.parser.parse_entry(entry, param_type, flavor)
            bounds = bounds_list[param_idx] if param_idx < len(bounds_list) else None
            self.param_configs.append(
                {"idx": param_idx, "init_val": init_val, "parsed": parsed, "bounds": bounds}
            )

            if parsed["is_fixed"]:
                # Fixed parameter
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
                registry.register_parameter(
                    param_type, flavor, param_idx, param, bounds=bounds
                )
            elif parsed["type"] == "reference":
                # Linked parameter - use shared parameter
                ref = parsed["value"]
                ref_type = ref["type"] if ref["type"] else param_type
                shared_init = init_val
                if bounds is not None:
                    lo, hi = bounds
                    u = (init_val - lo) / (hi - lo)
                    u = max(1e-6, min(1 - 1e-6, u))
                    shared_init = float(torch.logit(torch.tensor(u)).item())
                shared_param = registry.create_shared_parameter(
                    ref_type, ref["flavor"], ref["param_idx"], shared_init
                )
                self.free_params_list.append((param_idx, shared_param))
                registry.register_parameter(
                    param_type,
                    flavor,
                    param_idx,
                    shared_param,
                    source=(ref_type, ref["flavor"], ref["param_idx"]),
                    bounds=bounds,
                )
            elif parsed["type"] == "expression":
                # Expression-based parameter - will be evaluated dynamically
                # Store expression and create a placeholder parameter for gradient flow
                param = nn.Parameter(torch.tensor([init_val], dtype=torch.float32))
                self.free_params_list.append((param_idx, param))
                registry.register_parameter(
                    param_type, flavor, param_idx, param, bounds=bounds
                )
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



    # put x in for later
    def forward(self, x: torch.tensor, b: torch.Tensor) -> torch.Tensor:

        # Ensure x can broadcast with b (x: [n_events], b: [n_events, n_b])

        if b.dim() > x.dim():
            x = x.unsqueeze(-1)
        # Handle x >= 1 case (return zero)
        if torch.any(x >= 1):
            mask_val = (x < 1).type_as(b)
        else:
            mask_val = torch.ones_like(x)

        p = self.get_params_tensor()

        w = p[0]

        return torch.exp(-w * b**2) * mask_val

class SiversAV(nn.Module):
    def __init__(
        self, 
        flavor: str,
        init_params: List[float], 
        free_mask: List[Any],
        registry: ParameterRegistry,
        evaluator: ExpressionEvaluator,
        param_type: str = "sivers",
        param_bounds: Optional[List[Any]] = None,
        param_bounds_map: Optional[Dict[Tuple[str, str, int], Tuple[float, float]]] = None,

        ):

        super().__init__()
                
        if len(init_params) != 2:
            raise ValueError(
                f"{tcolors.FAIL}[fnp/sivers.py] MAP22 Sivers requires 1 parameter, got {len(init_params)}{tcolors.ENDC}. This is the Sivers AV Class."
            )
        if len(free_mask) != len(init_params):
            raise ValueError(
                f"{tcolors.FAIL}[fnp/sivers.py] free_mask length ({len(free_mask)}) must match init_params length ({len(init_params)}){tcolors.ENDC}"
            )
        self.param_bounds_map = param_bounds_map or {}
        self.flavor = flavor
        self.param_type = param_type
        self.n_params = len(init_params)
        self.registry = registry
        self.evaluator = evaluator
        self.parser = ParameterLinkParser()

        
        # Parse free_mask entries
        self.param_configs = []
        self.fixed_params = []
        self.free_params_list = []
        bounds_list = []

        if param_bounds is not None:
            try:
                for idx in range(min(len(param_bounds), self.n_params)):
                    b = parse_bound(param_bounds[idx] if idx < len(param_bounds) else None)
                    bounds_list.append(b)
            except (TypeError, KeyError):
                pass
        while len(bounds_list) < self.n_params:
            bounds_list.append(None)

        for param_idx, (init_val, entry) in enumerate(zip(init_params, free_mask)):
            parsed = self.parser.parse_entry(entry, param_type, flavor)
            bounds = bounds_list[param_idx] if param_idx < len(bounds_list) else None
            self.param_configs.append(
                {"idx": param_idx, "init_val": init_val, "parsed": parsed, "bounds": bounds}
            )

            if parsed["is_fixed"]:
                # Fixed parameter
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
                registry.register_parameter(
                    param_type, flavor, param_idx, param, bounds=bounds
                )
            elif parsed["type"] == "reference":
                # Linked parameter - use shared parameter
                ref = parsed["value"]
                ref_type = ref["type"] if ref["type"] else param_type
                shared_init = init_val
                if bounds is not None:
                    lo, hi = bounds
                    u = (init_val - lo) / (hi - lo)
                    u = max(1e-6, min(1 - 1e-6, u))
                    shared_init = float(torch.logit(torch.tensor(u)).item())
                shared_param = registry.create_shared_parameter(
                    ref_type, ref["flavor"], ref["param_idx"], shared_init
                )
                self.free_params_list.append((param_idx, shared_param))
                registry.register_parameter(
                    param_type,
                    flavor,
                    param_idx,
                    shared_param,
                    source=(ref_type, ref["flavor"], ref["param_idx"]),
                    bounds=bounds,
                )
            elif parsed["type"] == "expression":
                # Expression-based parameter - will be evaluated dynamically
                # Store expression and create a placeholder parameter for gradient flow
                param = nn.Parameter(torch.tensor([init_val], dtype=torch.float32))
                self.free_params_list.append((param_idx, param))
                registry.register_parameter(
                    param_type, flavor, param_idx, param, bounds=bounds
                )
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



    # put x in for later
    def forward(self, x: torch.tensor, b: torch.Tensor) -> torch.Tensor:

        # Ensure x can broadcast with b (x: [n_events], b: [n_events, n_b])

        if b.dim() > x.dim():
            x = x.unsqueeze(-1)
        # Handle x >= 1 case (return zero)
        if torch.any(x >= 1):
            mask_val = (x < 1).type_as(b)
        else:
            mask_val = torch.ones_like(x)

        p = self.get_params_tensor()

        p1 = p[0]
        p2 = p[1]

        # Look at Eq. (2.31) in https://doi.org/10.1007/JHEP05(2021)151 for the parameterized Sivers function. We are only implementing the exponential part here. 

        return torch.exp(-(p1/x + p2)* b) * mask_val