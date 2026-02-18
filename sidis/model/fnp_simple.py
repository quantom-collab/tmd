"""
Simple exponential fNP parametrization for TMD PDFs and FFs.

Supports parameter linking and expressions (same as flexible model):
- Boolean: true/false (independent/fixed)
- References: u[0], d[1], pdfs.u[0], ffs.d[1]
- Expressions: "0.5*u[0]", "0.5*u[1]", "2*u[0]+0.1"

Formulas:
  PDF: f̃_{1,NP}^q(x, b_T) = exp[-b_T²/4 · λ_f² · x^{α} · (1-x)²]
  FF:  D̃_{1,NP}^q(z, b_T) = exp[-b_T²/4 · λ_D² · z^{β} · (1-z)²]
  Evolution: S_NP(ζ, b_T) = exp[-g₂² b_T²/4 × ln(ζ/Q₀²)]

Per-flavor parameters: PDF [λ_f, α], FF [λ_D, β]

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
    TMD PDF: f̃_{1,NP}^q(x, b_T) = exp[-b_T²/4 · λ_f² · x^{α} · (1-x)²]
    Parameters: [λ_f, α] (index 0, 1). Supports linking and expressions in free_mask.
    Bounded params use sigmoid rescaling: value = lo + (hi - lo) * sigmoid(theta).
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
                f"{tcolors.FAIL}[fnp_simple.py] Exponential PDF requires 2 params [λ_f, α], got {len(init_params)}{tcolors.ENDC}"
            )
        if len(free_mask) != 2:
            raise ValueError(
                f"{tcolors.FAIL}[fnp_simple.py] free_mask length must match init_params (2){tcolors.ENDC}"
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
                    b = parse_bound(
                        param_bounds[idx] if idx < len(param_bounds) else None
                    )
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
        """Return param tensor with gradient flow preserved for trainable params."""
        try:
            dev = next(self.parameters()).device
        except StopIteration:
            try:
                dev = next(self.buffers()).device
            except StopIteration:
                dev = torch.device("cpu")
        param_vals = [None] * self.n_params
        for param_idx, val in self.fixed_params:
            param_vals[param_idx] = torch.tensor(
                [float(val)], dtype=torch.float32, device=dev
            )
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
            (
                param_vals[i]
                if param_vals[i] is not None
                else torch.tensor([0.0], dtype=torch.float32, device=dev)
            )
            for i in range(self.n_params)
        ]
        return torch.cat([v.flatten()[:1] for v in vals])

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
                f"{tcolors.FAIL}[fnp_simple.py] Exponential FF requires 2 params [λ_D, β], got {len(init_params)}{tcolors.ENDC}"
            )
        if len(free_mask) != 2:
            raise ValueError(
                f"{tcolors.FAIL}[fnp_simple.py] free_mask length must match init_params (2){tcolors.ENDC}"
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
                    b = parse_bound(
                        param_bounds[idx] if idx < len(param_bounds) else None
                    )
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
        """Return param tensor with gradient flow preserved for trainable params."""
        try:
            dev = next(self.parameters()).device
        except StopIteration:
            try:
                dev = next(self.buffers()).device
            except StopIteration:
                dev = torch.device("cpu")
        param_vals = [None] * self.n_params
        for param_idx, val in self.fixed_params:
            param_vals[param_idx] = torch.tensor(
                [float(val)], dtype=torch.float32, device=dev
            )
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
            (
                param_vals[i]
                if param_vals[i] is not None
                else torch.tensor([0.0], dtype=torch.float32, device=dev)
            )
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

    Supports free_mask: true/false, u[0], pdfs.u[1], "0.5*u[0]", "2*u[1]+0.1", etc.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__()

        self.hadron = config.get("hadron", "proton")
        self.pdf_flavor_keys = ["u", "ubar", "d", "dbar", "s", "sbar", "c", "cbar"]
        self.ff_flavor_keys = ["u", "ubar", "d", "dbar", "s", "sbar", "c", "cbar"]

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

        evolution_config = config.get("evolution", {})
        init_g2 = evolution_config.get("init_g2", 0.12840)
        free_mask = evolution_config.get("free_mask", [True])
        self.evolution = fNP_evolution(init_g2=init_g2, free_mask=free_mask)

        self.evolution_bounds: List[Tuple[float, float]] = []
        ev_bounds = evolution_config.get("param_bounds")
        if ev_bounds is not None:
            try:
                for b in ev_bounds:
                    parsed = parse_bound(b)
                    if parsed is not None:
                        self.evolution_bounds.append(parsed)
            except TypeError:
                pass

        self.param_bounds: Dict[Tuple[str, str, int], Tuple[float, float]] = {}
        for param_type in ("pdfs", "ffs"):
            type_config = config.get(param_type, {})
            flavor_keys = (
                self.pdf_flavor_keys if param_type == "pdfs" else self.ff_flavor_keys
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
                try:
                    for idx in range(min(len(bounds_list), 2)):
                        b = parse_bound(
                            bounds_list[idx] if idx < len(bounds_list) else None
                        )
                        if b is not None:
                            self.param_bounds[(param_type, flavor, idx)] = b
                except (TypeError, KeyError):
                    pass

        self.registry = ParameterRegistry()
        self.evaluator = ExpressionEvaluator(self.registry)

        pdf_graph = DependencyResolver.build_dependency_graph(
            config, "pdfs", self.pdf_flavor_keys
        )
        ff_graph = DependencyResolver.build_dependency_graph(
            config, "ffs", self.ff_flavor_keys
        )
        DependencyResolver.resolve_circular_dependencies(pdf_graph)
        DependencyResolver.resolve_circular_dependencies(ff_graph)

        # PDF modules
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
                param_type="pdfs",
                param_bounds=flavor_cfg.get("param_bounds"),
                param_bounds_map=self.param_bounds,
            )
        self.pdf_modules = nn.ModuleDict(pdf_modules)
        print(
            f"{tcolors.GREEN}[fNPManager] Initialized {len(self.pdf_modules)} PDF flavor modules\n{tcolors.ENDC}"
        )

        # FF modules
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
                param_type="ffs",
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
                with torch.no_grad():
                    param.data.clamp_(low, high)

    def randomize_params_in_bounds(self, seed: int = 42) -> None:
        """
        Set bounded parameters to random values within bounds.
        For theta (sigmoid) params: theta = logit(Uniform(0,1)).
        For evolution: value = Uniform(lo, hi).
        """
        gen = torch.Generator(device=next(self.parameters()).device)
        gen.manual_seed(seed)
        seen: set = set()
        for module in [*self.pdf_modules.values(), *self.ff_modules.values()]:
            if not hasattr(module, "free_params_list"):
                continue
            for param_idx, param in module.free_params_list:
                if id(param) in seen:
                    continue
                config = module.param_configs[param_idx]
                parsed = config["parsed"]
                bounds = config.get("bounds")
                if bounds is None and parsed["type"] == "reference":
                    ref = parsed["value"]
                    ref_type = ref["type"] if ref["type"] else module.param_type
                    key = (ref_type, ref["flavor"], ref["param_idx"])
                    bounds = module.param_bounds_map.get(key)
                if bounds is not None:
                    seen.add(id(param))
                    lo, hi = bounds
                    u = torch.rand(
                        1, generator=gen, device=param.device, dtype=param.dtype
                    )
                    u = u.clamp(1e-6, 1 - 1e-6)
                    theta = torch.logit(u)
                    with torch.no_grad():
                        param.data.copy_(theta)
        if self.evolution_bounds and hasattr(self.evolution, "free_g2"):
            p = self.evolution.free_g2
            if p.requires_grad and len(self.evolution_bounds) > 0:
                lo, hi = self.evolution_bounds[0]
                u = torch.rand(1, generator=gen, device=p.device, dtype=p.dtype)
                with torch.no_grad():
                    p.data.copy_(lo + (hi - lo) * u)

    def _compute_zeta(self, Q: torch.Tensor) -> torch.Tensor:
        return Q**2

    def get_evolution(self, b: torch.Tensor, Q: torch.Tensor) -> torch.Tensor:
        zeta = self._compute_zeta(Q)
        return self.evolution(b, zeta)

    def forward_pdf(
        self,
        x: torch.Tensor,
        b: torch.Tensor,
        flavors: Optional[List[str]] = None,
    ) -> Dict[str, torch.Tensor]:
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
        if b.dim() > x.dim():
            x = x.unsqueeze(-1)
        return torch.zeros_like(b)
