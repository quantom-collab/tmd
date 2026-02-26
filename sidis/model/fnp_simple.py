"""
Simple exponential fNP parametrization for TMD PDFs and FFs.

Supports parameter linking and expressions (same as flexible model):
- Boolean: true/false (independent/fixed)
- References: u[0], d[1], pdfs.u[0], ffs.d[1]
- Expressions: "0.5*u[0]", "0.5*u[1]", "2*u[0]+0.1"

Formulas:
  PDF: f̃_{1,NP}^q(x, b_T) = exp[-b_T²/4 · λ_f² · x^{alpha} · (1-x)²]
  FF:  D̃_{1,NP}^q(z, b_T) = exp[-b_T²/4 · λ_D² · z^{beta} · (1-z)²]
  Evolution: S_NP(ζ, b_T) = exp[-g₂² b_T²/4 · ln(ζ/Q₀²)]

Per-flavor parameters: PDF [λ_f, alpha], FF [λ_D, beta]
NOTE: the evolution is imported from fnp_base_flavor_dep.py.

Author: Chiara Bissolotti (cbissolotti@anl.gov)
"""

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

# TODO move fNP_evolution,
# it's not optimal that a model calls another model.
from .fnp_base_flavor_dep import fNP_evolution
from .fnp_config import (
    ParameterLinkParser,
    ParameterRegistry,
    ExpressionEvaluator,
    DependencyResolver,
    parse_bound,
)


###############################################################################
# 1. TMD PDF - Exponential with linking
###############################################################################
class TMDPDFExponential(nn.Module):
    """
    Exponential simple parametrization for TMD PDFs, flavor dependent.
    TMD PDF: f̃_{1,NP}^q(x, b_T) = exp[-b_T²/4 · λ_f² · x^{alpha} · (1-x)²]
    Parameters: [λ_f, alpha] (index 0, 1). Supports linking and expressions in free_mask.

    Arguments for __init__:
      1. flavor: str
          The flavor key for this PDF (e.g., 'u', 'd', etc.).
      2. init_params: List[float]
          Initial parameter values for this flavor, must be a list of length 2 ([λ_f, alpha]).
      3. free_mask: List[Any]
          List of length 2, describing how each parameter is handled:
            - Boolean: True (= independent/fit), False (= fixed)
            - Reference: binds parameter to another (e.g. 'u[0]', 'pdfs.u[0]')
            - Expression: algebraic mix (e.g. '0.5*u[0]')
      4. registry: ParameterRegistry
          Shared instance to manage and resolve parameter references/links across flavors.
      5. evaluator: ExpressionEvaluator
          Used to parse and evaluate expressions in free_mask.
      6. distrib_type: str, default "pdfs"
          Parameter type ('pdfs' beacuse this is a PDF module).
      7. param_bounds: Optional[List[Any]]
          Per-parameter bounds (list of [lo, hi] or None).
      8. param_bounds_map: Optional[Dict[Tuple[str, str, int], Tuple[float, float]]]
          Map for parameter bounds across all types/flavors for reference/resolution.

    TODO: Understand why bounded parameters use sigmoid rescaling: value = lo + (hi - lo) * sigmoid(theta).
    """

    def __init__(
        self,
        flavor: str,
        init_params: List[float],
        free_mask: List[Any],
        registry: ParameterRegistry,
        evaluator: ExpressionEvaluator,
        distrib_type: str = "pdfs",
        param_bounds: Optional[List[Any]] = None,
        param_bounds_map: Optional[
            Dict[Tuple[str, str, int], Tuple[float, float]]
        ] = None,
    ):
        # Initialize the module. This calls the constructor of the parent class
        # (nn.Module) to initialize internal machinery.
        super().__init__()

        # Initialize parameter bounds map. Filled in below.
        # This is used to resolve parameter bounds for references.
        self.param_bounds_map = param_bounds_map or {}

        # Check if the number of initial parameters is correct
        # for this flavor and parametrization.
        if len(init_params) != 2:
            raise ValueError(
                f"{tcolors.FAIL}[fnp_simple.py] Exponential PDF requires 2 params [λ_f, alpha], got {len(init_params)}{tcolors.ENDC}"
            )
        # Check if the number of free mask is correct
        # for this flavor and parametrization.
        if len(free_mask) != 2:
            raise ValueError(
                f"{tcolors.FAIL}[fnp_simple.py] free_mask length must match init_params (2){tcolors.ENDC}"
            )

        # Initialize the flavor, parameter type, number of parameters,
        # registry, evaluator, and parser.
        self.flavor = flavor
        self.distrib_type = distrib_type
        self.n_params = 2
        self.registry = registry
        self.evaluator = evaluator
        self.parser = ParameterLinkParser()

        # Initialize the bounds list. Config must match parametrization: 2 params.
        if param_bounds is not None and len(param_bounds) != 2:
            raise ValueError(
                f"{tcolors.FAIL}[fnp_simple.py] param_bounds must have length 2 for simple PDF"
                f"[λ_f, alpha], got {len(param_bounds)}{tcolors.ENDC}"
            )

        # Initialize the bounds list. Config must match parametrization: 2 params.
        # Declare the type of bounds_list and set its initial value to [None, None].
        # Optional[X] means "either X or None", so each element of bounds_list
        # is either a valid (lo, hi) or None
        bounds_list: List[Optional[Tuple[float, float]]] = [None, None]

        # If param_bounds is not None, parse the bounds for each parameter.
        if param_bounds is not None:
            for idx in range(2):
                # Parse the bound for the parameter at index idx.
                # using a the function parse_bound from fnp_config.py.
                b = parse_bound(param_bounds[idx])
                if b is not None:
                    # If the bound is valid, set the element of
                    # bounds_list at index idx to the bound.
                    bounds_list[idx] = b

        # Initialize the parameter configurations list.
        # This stores metadata for each parameter: index, initial value,
        # parsed entry, and bounds, so that the code knows how to treat it at runtime.
        self.param_configs = []
        # Initialize the fixed parameters list.
        # This will be used to store the fixed parameters.
        self.fixed_params = []
        # Initialize the free parameters list.
        # This will be used to store the free parameters.
        self.free_params_list = []

        for param_idx, (init_val, entry) in enumerate(zip(init_params, free_mask)):
            # Parse the entry for the parameter at index param_idx.
            # using a the function parse_entry from fnp_config.py.
            # parsed is a dictionary with the following keys:
            # - "type": "boolean", "reference", or "expression"
            # - "value": boolean value, or reference info, or expression string
            # - "is_fixed": True if parameter is fixed (False)
            parsed = self.parser.parse_entry(entry, distrib_type, flavor)

            # Get the bounds for the parameter at index param_idx.
            bounds = bounds_list[param_idx]
            self.param_configs.append(
                {
                    "idx": param_idx,
                    "init_val": init_val,
                    "parsed": parsed,
                    "bounds": bounds,
                }
            )

            # If the parameter is fixed, add it to the fixed parameters list.
            if parsed["is_fixed"]:
                self.fixed_params.append((param_idx, init_val))

            # If the parameter is a boolean and is True, add it to the
            # free parameters list. Also, if the bounds are specified,
            # use them to rescale the parameter.
            # NOTE: if no bounds are specified, the parameter is not rescaled.
            elif parsed["type"] == "boolean" and parsed["value"]:
                # If the bounds are not None, use them to
                # rescale the parameter.
                if bounds is not None:
                    # Get the lower and upper bounds.
                    lo, hi = bounds
                    # Rescale the parameter to the range [0, 1].
                    u = (init_val - lo) / (hi - lo)
                    # Clamp the parameter to the range [1e-6, 1 - 1e-6].
                    u = max(1e-6, min(1 - 1e-6, u))

                    # Convert the bounded parameter u into the unbounded parameter theta,
                    # the one that will be updated by the optimizer.
                    # This is done using the logit function, which is the inverse of
                    # the sigmoid function, defined as: logit(x) = log(x / (1 - x)).
                    theta = torch.tensor(
                        float(torch.logit(torch.tensor(u)).item()),
                        dtype=torch.float32,
                    )
                    # Register the parameter theta as a nn.Parameter.
                    # The unsqueeze is used to add a dimension to the tensor,
                    # so that it can be used in the forward pass.
                    # theta is a scalar (0-dimensional tensor).
                    # nn.Parameter is usually used with at least 1D tensors.
                    # unsqueeze(0) turns a scalar into a 1D tensor of shape (1,)
                    param = nn.Parameter(theta.unsqueeze(0))
                else:
                    param = nn.Parameter(torch.tensor([init_val], dtype=torch.float32))

                # Add the parameter to the free parameters list.
                self.free_params_list.append((param_idx, param))

                # Register the parameter in the registry.
                registry.register_parameter(distrib_type, flavor, param_idx, param)

            # This is for shared parameters, i.e. the ones that have a
            # reference instead of a boolean under the key "fixed"
            elif parsed["type"] == "reference":
                ref = parsed["value"]
                # Get the type of the reference parameter. If it has been parsed under the key "type",
                # use it. Otherwise, use the distribution type, as it means that the reference is
                # a within the same distribution.
                ref_type = ref["type"] if ref["type"] else distrib_type
                # Create a shared parameter in the registry
                shared_param = registry.create_shared_parameter(
                    ref_type, ref["flavor"], ref["param_idx"], init_val
                )
                # Add the shared parameter to the free parameters list.
                self.free_params_list.append((param_idx, shared_param))

                # Register the shared parameter in the registry.
                registry.register_parameter(
                    distrib_type,
                    flavor,
                    param_idx,
                    shared_param,
                    source=(ref_type, ref["flavor"], ref["param_idx"]),
                )
            # If there is an expresson under the key "is_fixed" in the yaml file,
            # that means that the parameter is free, so we add it to the free
            # parameters list and register it in the registry.
            elif parsed["type"] == "expression":
                # Create a new parameter with the initial value.
                param = nn.Parameter(torch.tensor([init_val], dtype=torch.float32))
                # Add the parameter to the free parameters list.
                self.free_params_list.append((param_idx, param))
                # Register the parameter in the registry.
                registry.register_parameter(distrib_type, flavor, param_idx, param)
                # Store the expression for dynamic evaluation.
                parsed["expression"] = parsed["value"]

        # Register fixed parameters as buffers.
        for param_idx, val in self.fixed_params:
            self.register_buffer(
                f"fixed_param_{param_idx}", torch.tensor([val], dtype=torch.float32)
            )
        # Register free parameters as parameters.
        for param_idx, param in self.free_params_list:
            self.register_parameter(f"free_param_{param_idx}", param)

    def get_params_tensor(self) -> torch.Tensor:
        """
        Return a 1D tensor of physical parameter values [p0, p1, ...] for this flavor.

        Fixed params use their stored value. Free params: boolean/reference use
        sigmoid rescaling; expression uses evaluator. Gradient flow is preserved.
        """
        # Infer device from existing parameters or buffers so output matches.
        # next(iterable): returns first item; raises StopIteration if empty.
        # .parameters() returns an iterator over nn.Parameter objects.
        try:
            dev = next(self.parameters()).device
        except StopIteration:
            try:
                dev = next(self.buffers()).device
            except StopIteration:
                dev = torch.device("cpu")

        # Initialize the list of parameter values, created empty and filled in below.
        # param_vals[i] will hold the physical value for parameter i (or None).
        # [None] * n: list of n copies of None.
        param_vals = [None] * self.n_params

        # Fixed params: use stored scalar value, wrap in tensor (no grad).
        for param_idx, val in self.fixed_params:
            param_vals[param_idx] = torch.tensor(
                [float(val)], dtype=torch.float32, device=dev
            )

        # Free params: compute physical value from theta or expression.
        # self.free_params_list is a list of tuples, not a map. Each element is (param_idx, param):
        # param_idx: int — index of the parameter (0 or 1 for the simple model)
        # param: nn.Parameter — the trainable tensor (theta or raw value)
        for param_idx, param in self.free_params_list:
            config = self.param_configs[param_idx]
            # Get the parsed entry for the parameter at index param_idx.
            parsed = config["parsed"]

            if parsed["type"] == "boolean" or parsed["type"] == "reference":
                # Get the bounds (tuple) for the parameter at index param_idx.
                bounds = config.get("bounds")

                # If there are no bounds, but the parameter is a reference
                # based on another parameter, get the bounds from the
                # source parameter.
                if bounds is None and parsed["type"] == "reference":
                    ref = parsed["value"]
                    ref_type = ref["type"] if ref["type"] else self.distrib_type
                    # Get the key for the source parameter.
                    key = (ref_type, ref["flavor"], ref["param_idx"])
                    # Get the bounds from the source parameter.
                    bounds = self.param_bounds_map.get(key)

                # If there are bounds, use them to rescale the parameter.
                if bounds is not None:
                    # Get the lower and upper bounds.
                    lo, hi = bounds
                    # u is the bounded parameter, in the range [0,1], differentiable
                    # param is the unbounded parameter theta
                    u = torch.sigmoid(param)  # [0,1], differentiable
                    val_t = lo + (hi - lo) * u.flatten()[0]
                    param_vals[param_idx] = val_t.unsqueeze(0)
                else:
                    # Unbounded: use param value directly.
                    p = param.flatten()[0]
                    param_vals[param_idx] = p.unsqueeze(0)

            # If the parameter is an expression, evaluate it and use the result.
            # NOTE: bounds are ignored for expression parameters. An expression
            # can produce a value outside the bounds.
            elif parsed["type"] == "expression":
                # Evaluate the expression and use the result.
                expr_value = self.evaluator.evaluate(
                    parsed["expression"], self.distrib_type, self.flavor
                )
                # Store the result in the param_vals list.
                param_vals[param_idx] = expr_value
                # Detach the result from the computational graph to avoid
                # unnecessary gradients.
                param.data = expr_value.detach()

        # Fill any missing slots with 0.0 (safety fallback).
        # NOTE: This should never happen. It means that the parameter is not
        # defined as fixed or free in the yaml file.
        # List comprehension: [x for i in range(n)] builds a list.
        vals = [
            (
                param_vals[i]
                if param_vals[i] is not None
                else torch.tensor([0.0], dtype=torch.float32, device=dev)
            )
            for i in range(self.n_params)
        ]

        # Print an error message if any parameter is None.
        if any(v is None for v in vals):
            print(
                f"{tcolors.FAIL}[fnp_simple.py] get_params_tensor: some parameters are None{tcolors.ENDC}"
            )

        # Ensure each param must be a single scalar; fail loudly if misshapen.
        for i, v in enumerate(vals):
            # numel() is the number of elements in a tensor
            if v is not None and v.numel() != 1:
                raise ValueError(
                    f"{tcolors.FAIL}[fnp_simple.py] get_params_tensor: param {i} should have 1 element, got {v.numel()} (shape {v.shape}){tcolors.ENDC}"
                )
        # torch.cat(seq, dim=0): concatenate tensors along dimension 0.
        # v.flatten()[:1]: flatten to 1D, take first element.
        return torch.cat([v.flatten()[:1] for v in vals])

    def forward(
        self,
        x: torch.Tensor,
        b: torch.Tensor,
        flavor_idx: int = 0,
    ) -> torch.Tensor:
        """
        Evaluate TMD PDF: f̃_{1,NP}^q(x, b_T) = exp[-b_T²/4 · λ_f² · x^{alpha} · (1-x)²].
        """
        # Align dimensions: if b has more dims than x, broadcast x for element-wise ops.
        # .dim(): number of dimensions. unsqueeze(-1): add dimension at the end.
        if b.dim() > x.dim():
            x = x.unsqueeze(-1)

        # x must be in [0, 1) (Bjorken x). Mask out x >= 1 to avoid invalid (1-x)².
        # torch.any(cond): True if any element satisfies cond.
        # (x < 1).type_as(b): boolean mask cast to b's dtype for multiplication.
        # torch.ones_like(x): tensor of ones with same shape as x.
        if torch.any(x >= 1):
            # Print a warning message if x >= 1.
            print(
                f"{tcolors.WARNING}[fnp_simple.py] forward: x >= 1, setting mask_val to 0{tcolors.ENDC}"
            )
            mask_val = (x < 1).type_as(b)
        else:
            mask_val = torch.ones_like(x)

        # Get physical params [λ_f, α] (handles linking, expressions, sigmoid).
        p = self.get_params_tensor()
        lam_f, alpha = p[0], p[1]

        # Avoid log(0) or 0^negative in x^α.
        # torch.clamp(x, min=1e-10): element-wise lower bound.
        x_safe = torch.clamp(x, min=1e-10)

        # Exponent: -b_T²/4 · λ_f² · x^{α} · (1-x)²
        # torch.pow(base, exp): element-wise base^exp.
        exponent = (
            -((b / 2.0) ** 2) * (lam_f**2) * torch.pow(x_safe, alpha) * ((1 - x) ** 2)
        )
        return torch.exp(exponent) * mask_val


###############################################################################
# 2. TMD FF - Exponential with linking
###############################################################################
class TMDFFExponential(nn.Module):
    """
    TMD FF: D̃_{1,NP}^q(z, b_T) = exp[-b_T²/4 · λ_D² · z^{β} · (1-z)²]
    Parameters: [λ_D, β] (index 0, 1). Supports linking and expressions in free_mask.
    Bounded params use sigmoid rescaling: value = lo + (hi - lo) * sigmoid(theta).
    """

    def __init__(
        self,
        flavor: str,
        init_params: List[float],
        free_mask: List[Any],
        registry: ParameterRegistry,
        evaluator: ExpressionEvaluator,
        distrib_type: str = "ffs",
        param_bounds: Optional[List[Any]] = None,
        param_bounds_map: Optional[
            Dict[Tuple[str, str, int], Tuple[float, float]]
        ] = None,
    ):
        super().__init__()
        self.param_bounds_map = param_bounds_map or {}
        if len(init_params) != 2:
            raise ValueError(
                f"{tcolors.FAIL}[fnp_simple.py] Exponential FF requires 2 params [λ_D, β], got {len(init_params)}{tcolors.ENDC}"
            )
        if len(free_mask) != 2:
            raise ValueError(
                f"{tcolors.FAIL}[fnp_simple.py] free_mask length must match init_params (2){tcolors.ENDC}"
            )

        self.flavor = flavor
        self.distrib_type = distrib_type
        self.n_params = 2
        self.registry = registry
        self.evaluator = evaluator
        self.parser = ParameterLinkParser()

        # Config must match parametrization: 2 params [λ_D, β].
        if param_bounds is not None and len(param_bounds) != 2:
            raise ValueError(
                f"{tcolors.FAIL}[fnp_simple.py] param_bounds must have length 2 for simple FF [λ_D, β], got {len(param_bounds)}{tcolors.ENDC}"
            )
        bounds_list: List[Optional[Tuple[float, float]]] = [None, None]
        if param_bounds is not None:
            for idx in range(2):
                b = parse_bound(param_bounds[idx])
                if b is not None:
                    bounds_list[idx] = b

        self.param_configs = []
        self.fixed_params = []
        self.free_params_list = []

        for param_idx, (init_val, entry) in enumerate(zip(init_params, free_mask)):
            parsed = self.parser.parse_entry(entry, distrib_type, flavor)
            bounds = bounds_list[param_idx]
            self.param_configs.append(
                {
                    "idx": param_idx,
                    "init_val": init_val,
                    "parsed": parsed,
                    "bounds": bounds,
                }
            )

            if parsed["is_fixed"]:
                self.fixed_params.append((param_idx, init_val))
            elif parsed["type"] == "boolean" and parsed["value"]:
                if bounds is not None:
                    lo, hi = bounds
                    u = (init_val - lo) / (hi - lo)
                    u = max(1e-6, min(1 - 1e-6, u))
                    theta = torch.tensor(
                        float(torch.logit(torch.tensor(u)).item()),
                        dtype=torch.float32,
                    )
                    param = nn.Parameter(theta.unsqueeze(0))
                else:
                    param = nn.Parameter(torch.tensor([init_val], dtype=torch.float32))
                self.free_params_list.append((param_idx, param))
                registry.register_parameter(distrib_type, flavor, param_idx, param)
            elif parsed["type"] == "reference":
                ref = parsed["value"]
                ref_type = ref["type"] if ref["type"] else distrib_type
                shared_param = registry.create_shared_parameter(
                    ref_type, ref["flavor"], ref["param_idx"], init_val
                )
                self.free_params_list.append((param_idx, shared_param))
                registry.register_parameter(
                    distrib_type,
                    flavor,
                    param_idx,
                    shared_param,
                    source=(ref_type, ref["flavor"], ref["param_idx"]),
                )
            elif parsed["type"] == "expression":
                param = nn.Parameter(torch.tensor([init_val], dtype=torch.float32))
                self.free_params_list.append((param_idx, param))
                registry.register_parameter(distrib_type, flavor, param_idx, param)
                parsed["expression"] = parsed["value"]

        for param_idx, val in self.fixed_params:
            self.register_buffer(
                f"fixed_param_{param_idx}", torch.tensor([val], dtype=torch.float32)
            )
        for param_idx, param in self.free_params_list:
            self.register_parameter(f"free_param_{param_idx}", param)

    def get_params_tensor(self) -> torch.Tensor:
        """
        Return a 1D tensor of physical parameter values [p0, p1, ...] for this flavor.

        Fixed params use their stored value. Free params: boolean/reference use
        sigmoid rescaling; expression uses evaluator. Gradient flow is preserved.
        """
        # Infer device from existing parameters or buffers so output matches.
        # next(iterable): returns first item; raises StopIteration if empty.
        # .parameters() returns an iterator over nn.Parameter objects.
        try:
            dev = next(self.parameters()).device
        except StopIteration:
            try:
                dev = next(self.buffers()).device
            except StopIteration:
                dev = torch.device("cpu")

        # param_vals[i] will hold the physical value for parameter i (or None).
        # [None] * n: list of n copies of None.
        param_vals = [None] * self.n_params

        # Fixed params: use stored scalar value, wrap in tensor (no grad).
        for param_idx, val in self.fixed_params:
            param_vals[param_idx] = torch.tensor(
                [float(val)], dtype=torch.float32, device=dev
            )

        # Free params: compute physical value from theta or expression.
        for param_idx, param in self.free_params_list:
            config = self.param_configs[param_idx]
            parsed = config["parsed"]

            if parsed["type"] == "boolean" or parsed["type"] == "reference":
                # Bounded or unbounded theta -> physical value.
                bounds = config.get("bounds")
                if bounds is None and parsed["type"] == "reference":
                    # Reference: bounds come from the source param.
                    ref = parsed["value"]
                    ref_type = ref["type"] if ref["type"] else self.distrib_type
                    key = (ref_type, ref["flavor"], ref["param_idx"])
                    bounds = self.param_bounds_map.get(key)
                if bounds is not None:
                    lo, hi = bounds
                    raw = torch.sigmoid(param)  # [0,1], differentiable
                    val_t = lo + (hi - lo) * raw.flatten()[0]
                    param_vals[param_idx] = val_t.unsqueeze(0)
                else:
                    # Unbounded: use param value directly.
                    p = param.flatten()[0]
                    param_vals[param_idx] = p.unsqueeze(0)

            elif parsed["type"] == "expression":
                # Expression: evaluate and use result; sync param.data for consistency.
                expr_value = self.evaluator.evaluate(
                    parsed["expression"], self.distrib_type, self.flavor
                )
                param_vals[param_idx] = expr_value
                param.data = expr_value.detach()

        # Fill any missing slots with 0.0 (safety fallback).
        # List comprehension: [x for i in range(n)] builds a list.
        vals = [
            (
                param_vals[i]
                if param_vals[i] is not None
                else torch.tensor([0.0], dtype=torch.float32, device=dev)
            )
            for i in range(self.n_params)
        ]
        # Ensure each param is a single scalar; fail loudly if misshapen.
        for i, v in enumerate(vals):
            if v is not None and v.numel() != 1:
                raise ValueError(
                    f"{tcolors.FAIL}[fnp_simple.py] get_params_tensor: param {i} should have 1 element, got {v.numel()} (shape {v.shape}){tcolors.ENDC}"
                )
        # torch.cat(seq, dim=0): concatenate tensors along dim.
        return torch.cat([v.flatten()[:1] for v in vals])

    def forward(
        self,
        z: torch.Tensor,
        b: torch.Tensor,
        flavor_idx: int = 0,
    ) -> torch.Tensor:
        """
        Evaluate TMD FF: D̃_{1,NP}^q(z, b_T) = exp[-b_T²/4 · λ_D² · z^{β} · (1-z)²].
        """
        # Align dimensions: if b has more dims than z, broadcast z.
        if b.dim() > z.dim():
            z = z.unsqueeze(-1)

        # z must be in [0, 1) (fragmentation fraction). Mask out z >= 1.
        if torch.any(z >= 1):
            mask_val = (z < 1).type_as(b)
        else:
            mask_val = torch.ones_like(z)

        # Get physical params [λ_D, β].
        p = self.get_params_tensor()
        lam_D, beta = p[0], p[1]

        # Avoid 0^negative in z^β.
        z_safe = torch.clamp(z, min=1e-10)

        # Exponent: -b_T²/4 · λ_D² · z^{β} · (1-z)²
        exponent = (
            -((b / 2.0) ** 2) * (lam_D**2) * torch.pow(z_safe, beta) * ((1 - z) ** 2)
        )
        return torch.exp(exponent) * mask_val


###############################################################################
# 3. Default parameters
###############################################################################
DEFAULT_EVOLUTION = {"init_g2": 0.12840, "free_mask": [True]}
DEFAULT_PDF_PARAMS = {"init_params": [0.5, 0.3], "free_mask": [True, True]}
DEFAULT_FF_PARAMS = {"init_params": [0.5, 0.3], "free_mask": [True, True]}


###############################################################################
# 4. Manager Class
###############################################################################
class fNPManager(nn.Module):
    """
    Manager for simple exponential fNP parametrization with parameter linking.

    Orchestrates PDF and FF modules, evolution, and shared parameter registry.
    Supports free_mask: true/false, u[0], pdfs.u[1], "0.5*u[0]", "2*u[1]+0.1", etc.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the fNPManager.
        """
        # Initialize the parent class nn.Module.
        super().__init__()

        # Hadron target and flavor lists for PDFs and FFs.
        # These are hardcoded because they'll be used to check if such keys are in the config.
        self.hadron = config.get("hadron", "proton")
        self.pdf_flavor_keys = ["u", "ubar", "d", "dbar", "s", "sbar", "c", "cbar"]
        self.ff_flavor_keys = ["u", "ubar", "d", "dbar", "s", "sbar", "c", "cbar"]

        # Print out messages to the user.
        print(
            f"{tcolors.BLUE}\n[fNPManager] Initializing simple exponential fNP parametrization (with linking)"
        )
        print(f"  Hadron: {self.hadron}")
        print(
            "  PDF: exp[-b²/4 · λ_f² · x^{α} · (1-x)²],  FF: exp[-b²/4 · λ_D² · z^{β} · (1-z)²]"
        )
        print(
            f"  Params: PDF [λ_f, α], FF [λ_D, β]. Supports links and expressions.\n{tcolors.ENDC}"
        )

        # Evolution factor S_NP(ζ, b_T) = exp[-g₂² b_T²/4 × ln(ζ/Q₀²)].
        evolution_config = config.get("evolution", {})
        init_g2 = evolution_config.get("init_g2", 0.12840)
        free_mask = evolution_config.get("free_mask", [True])
        # Initialize the evolution factor module.
        # NOTE This is called in fnp_base_flavor_dep.py.
        # TODO Move this call, it's not optimal that a model calls another model.
        self.evolution = fNP_evolution(init_g2=init_g2, free_mask=free_mask)

        # Parse evolution param bounds (g2 must be in [lo, hi] for clamping).
        self.evolution_bounds: List[Tuple[float, float]] = []
        ev_bounds = evolution_config.get("param_bounds")
        if ev_bounds is not None:
            if len(ev_bounds) != 1:
                raise ValueError(
                    f"{tcolors.FAIL}[fnp_simple.py] evolution param_bounds must have length 1 (g2), got {len(ev_bounds)}{tcolors.ENDC}"
                )
            parsed = parse_bound(ev_bounds[0])
            if parsed is not None:
                self.evolution_bounds.append(parsed)

        # Build global param_bounds map: (distrib_type, flavor, param_idx) -> (lo, hi).
        # Used by PDF/FF modules to resolve bounds for references (e.g. d[0]=u[0]).
        self.param_bounds: Dict[Tuple[str, str, int], Tuple[float, float]] = {}
        for distrib_type in ("pdfs", "ffs"):
            type_config = config.get(distrib_type, {})
            flavor_keys = (
                self.pdf_flavor_keys if distrib_type == "pdfs" else self.ff_flavor_keys
            )
            for flavor in flavor_keys:
                flavor_cfg = type_config.get(flavor, {})
                if flavor_cfg is None:
                    continue
                bounds_list = (
                    flavor_cfg.get("param_bounds")
                    if hasattr(flavor_cfg, "get")
                    else None
                )
                if bounds_list is None:
                    continue
                if len(bounds_list) != 2:
                    raise ValueError(
                        f"{tcolors.FAIL}[fnp_simple.py] param_bounds for {distrib_type}.{flavor} must have length 2, got {len(bounds_list)}{tcolors.ENDC}"
                    )
                for idx in range(2):
                    b = parse_bound(bounds_list[idx])
                    if b is not None:
                        self.param_bounds[(distrib_type, flavor, idx)] = b

        # Shared registry and evaluator for parameter linking and expressions.
        self.registry = ParameterRegistry()
        self.evaluator = ExpressionEvaluator(self.registry)

        # Build and resolve dependency graphs (for circular ref detection).
        pdf_graph = DependencyResolver.build_dependency_graph(
            config, "pdfs", self.pdf_flavor_keys
        )
        ff_graph = DependencyResolver.build_dependency_graph(
            config, "ffs", self.ff_flavor_keys
        )
        DependencyResolver.resolve_circular_dependencies(pdf_graph)
        DependencyResolver.resolve_circular_dependencies(ff_graph)

        # Create PDF modules (one per flavor). Use defaults if flavor not in config.
        pdf_config = config.get("pdfs", {})
        pdf_modules = {}
        for flavor in self.pdf_flavor_keys:
            flavor_cfg = pdf_config.get(flavor, None)
            if flavor_cfg is None:
                flavor_cfg = DEFAULT_PDF_PARAMS.copy()
                print(
                    f"{tcolors.WARNING}[fNPManager] Using defaults for PDF flavor '{flavor}'{tcolors.ENDC}"
                )
            else:
                print(
                    f"{tcolors.OKLIGHTBLUE}[fNPManager] Using user-defined PDF flavor '{flavor}'{tcolors.ENDC}"
                )
            pdf_modules[flavor] = TMDPDFExponential(
                flavor=flavor,
                init_params=flavor_cfg["init_params"],
                free_mask=flavor_cfg["free_mask"],
                registry=self.registry,
                evaluator=self.evaluator,
                distrib_type="pdfs",
                param_bounds=flavor_cfg.get("param_bounds"),
                param_bounds_map=self.param_bounds,
            )
        # nn.ModuleDict: submodule container so PDFs appear in .parameters().
        self.pdf_modules = nn.ModuleDict(pdf_modules)
        print(
            f"{tcolors.GREEN}[fNPManager] Initialized {len(self.pdf_modules)} PDF flavor modules\n{tcolors.ENDC}"
        )

        # Create FF modules (one per flavor).
        ff_config = config.get("ffs", {})
        ff_modules = {}
        for flavor in self.ff_flavor_keys:
            flavor_cfg = ff_config.get(flavor, None)
            if flavor_cfg is None:
                flavor_cfg = DEFAULT_FF_PARAMS.copy()
                print(
                    f"{tcolors.WARNING}[fNPManager] Using defaults for FF flavor '{flavor}'{tcolors.ENDC}"
                )
            else:
                print(
                    f"{tcolors.OKLIGHTBLUE}[fNPManager] Using user-defined FF flavor '{flavor}'{tcolors.ENDC}"
                )
            ff_modules[flavor] = TMDFFExponential(
                flavor=flavor,
                init_params=flavor_cfg.get(
                    "init_params", DEFAULT_FF_PARAMS["init_params"]
                ),
                free_mask=flavor_cfg.get("free_mask", DEFAULT_FF_PARAMS["free_mask"]),
                registry=self.registry,
                evaluator=self.evaluator,
                distrib_type="ffs",
                param_bounds=flavor_cfg.get("param_bounds"),
                param_bounds_map=self.param_bounds,
            )
        self.ff_modules = nn.ModuleDict(ff_modules)
        print(
            f"{tcolors.GREEN}[fNPManager] Initialized {len(self.ff_modules)} FF flavor modules\n{tcolors.ENDC}"
        )

    def get_trainable_bounds(
        self,
    ) -> List[Tuple[nn.Parameter, Optional[float], Optional[float]]]:
        """
        Return (parameter, low, high) for each trainable parameter that has bounds.
        For fnp_simple, only evolution params need clamping (PDF/FF use sigmoid rescaling).
        """
        result: List[Tuple[nn.Parameter, Optional[float], Optional[float]]] = []
        # Evolution g2 is the only param that may need clamping (not sigmoid-rescaled).
        if self.evolution_bounds and hasattr(self.evolution, "free_g2"):
            p = self.evolution.free_g2
            if p.requires_grad and len(self.evolution_bounds) > 0:
                lo, hi = self.evolution_bounds[0]
                result.append((p, lo, hi))
        return result

    def clamp_parameters_to_bounds(self) -> None:
        """
        Clamp evolution parameters to their allowed intervals.
        PDF/FF params use sigmoid rescaling and need no clamping.
        """
        for param, low, high in self.get_trainable_bounds():
            if low is not None and high is not None:
                # torch.no_grad(): disable gradient tracking for in-place clamp.
                with torch.no_grad():
                    param.data.clamp_(low, high)

    def randomize_params_in_bounds(self, seed: int = 42) -> None:
        """
        Set bounded parameters to random values within bounds.
        For theta (sigmoid) params: theta = logit(Uniform(0,1)).
        For evolution: value = Uniform(lo, hi).
        """
        # Generator with seed for reproducibility.
        gen = torch.Generator(device=next(self.parameters()).device)
        gen.manual_seed(seed)
        # seen: avoid randomizing the same shared param twice (references).
        seen: set = set()
        for module in [*self.pdf_modules.values(), *self.ff_modules.values()]:
            if not hasattr(module, "free_params_list"):
                continue
            for param_idx, param in module.free_params_list:
                if id(param) in seen:
                    continue  # Shared param already randomized
                config = module.param_configs[param_idx]
                parsed = config["parsed"]
                bounds = config.get("bounds")
                # References: bounds come from the source param.
                if bounds is None and parsed["type"] == "reference":
                    ref = parsed["value"]
                    ref_type = ref["type"] if ref["type"] else module.distrib_type
                    key = (ref_type, ref["flavor"], ref["param_idx"])
                    bounds = module.param_bounds_map.get(key)
                if bounds is not None:
                    seen.add(id(param))
                    lo, hi = bounds
                    # Sigmoid params: u ~ Uniform(0,1), theta = logit(u).
                    u = torch.rand(
                        1, generator=gen, device=param.device, dtype=param.dtype
                    )
                    u = u.clamp(1e-6, 1 - 1e-6)
                    theta = torch.logit(u)
                    with torch.no_grad():
                        param.data.copy_(theta)
        # Evolution: value ~ Uniform(lo, hi) directly.
        if self.evolution_bounds and hasattr(self.evolution, "free_g2"):
            p = self.evolution.free_g2
            if p.requires_grad and len(self.evolution_bounds) > 0:
                lo, hi = self.evolution_bounds[0]
                u = torch.rand(1, generator=gen, device=p.device, dtype=p.dtype)
                with torch.no_grad():
                    p.data.copy_(lo + (hi - lo) * u)

    def _compute_zeta(self, Q: torch.Tensor) -> torch.Tensor:
        """Rapidity scale ζ = Q² for evolution factor."""
        return Q**2

    def get_evolution(self, b: torch.Tensor, Q: torch.Tensor) -> torch.Tensor:
        """Compute evolution factor S_NP(ζ, b_T)."""
        zeta = self._compute_zeta(Q)
        return self.evolution(b, zeta)

    def forward_pdf(
        self,
        x: torch.Tensor,
        b: torch.Tensor,
        flavors: Optional[List[str]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Evaluate TMD PDFs for given flavors. Returns dict flavor -> tensor."""
        if flavors is None:
            flavors = self.pdf_flavor_keys
        outputs = {}
        for flavor in flavors:
            if flavor in self.pdf_modules:
                outputs[flavor] = self.pdf_modules[flavor](x, b, 0)
            else:
                raise ValueError(f"Unknown PDF flavor: {flavor}")
        return outputs

    def forward_ff(
        self,
        z: torch.Tensor,
        b: torch.Tensor,
        flavors: Optional[List[str]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Evaluate TMD FFs for given flavors. Returns dict flavor -> tensor."""
        if flavors is None:
            flavors = self.ff_flavor_keys
        outputs = {}
        for flavor in flavors:
            if flavor in self.ff_modules:
                outputs[flavor] = self.ff_modules[flavor](z, b, 0)
            else:
                raise ValueError(f"Unknown FF flavor: {flavor}")
        return outputs

    def forward_sivers(
        self,
        x: torch.Tensor,
        b: torch.Tensor,
    ) -> torch.Tensor:
        """Sivers function: not implemented for simple model; returns zeros."""
        if b.dim() > x.dim():
            x = x.unsqueeze(-1)
        return torch.zeros_like(b)
