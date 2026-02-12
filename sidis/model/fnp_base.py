"""
Simple exponential fNP parametrization for TMD PDFs and FFs.

Supports parameter linking and expressions (same as flexible model):
- Boolean: true/false (independent/fixed)
- References: u[0], d[1], pdfs.u[0], ffs.d[1]
- Expressions: "0.5*u[0]", "0.5*u[1]", "2*u[0]+0.1"

Formulas:
  PDF: f̃_{1,NP}^q(x, b_T) = exp[-b_T²/4 · λ_f² · x^{-α} · (1-x)²]
  FF:  D̃_{1,NP}^q(z, b_T) = exp[-b_T²/4 · λ_D² · z^{-β} · (1-z)²]
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
from .fnp_base_flexible import (
    ParameterLinkParser,
    ParameterRegistry,
    ExpressionEvaluator,
    DependencyResolver,
)


###############################################################################
# 1. TMD PDF - Exponential with linking
###############################################################################
class TMDPDFExponential(nn.Module):
    """
    TMD PDF: f̃_{1,NP}^q(x, b_T) = exp[-b_T²/4 · λ_f² · x^{-α} · (1-x)²]
    Parameters: [λ_f, α] (index 0, 1). Supports linking and expressions in free_mask.
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
        if len(init_params) != 2:
            raise ValueError(
                f"{tcolors.FAIL}[fnp_base.py] Exponential PDF requires 2 params [λ_f, α], got {len(init_params)}{tcolors.ENDC}"
            )
        if len(free_mask) != 2:
            raise ValueError(
                f"{tcolors.FAIL}[fnp_base.py] free_mask length must match init_params (2){tcolors.ENDC}"
            )

        self.flavor = flavor
        self.param_type = param_type
        self.n_params = 2
        self.registry = registry
        self.evaluator = evaluator
        self.parser = ParameterLinkParser()

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

        for param_idx, val in self.fixed_params:
            self.register_buffer(
                f"fixed_param_{param_idx}", torch.tensor([val], dtype=torch.float32)
            )
        for param_idx, param in self.free_params_list:
            self.register_parameter(f"free_param_{param_idx}", param)

    def get_params_tensor(self) -> torch.Tensor:
        params = [0.0] * self.n_params
        for param_idx, val in self.fixed_params:
            params[param_idx] = val
        for param_idx, param in self.free_params_list:
            config = self.param_configs[param_idx]
            parsed = config["parsed"]
            if parsed["type"] == "boolean" or parsed["type"] == "reference":
                params[param_idx] = (
                    param.item()
                    if param.numel() == 1
                    else (param[0].item() if len(param.shape) > 0 else param.item())
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
            -((b / 2.0) ** 2)
            * (lam_f**2)
            * torch.pow(x_safe, -alpha)
            * ((1 - x) ** 2)
        )
        return torch.exp(exponent) * mask_val


###############################################################################
# 2. TMD FF - Exponential with linking
###############################################################################
class TMDFFExponential(nn.Module):
    """
    TMD FF: D̃_{1,NP}^q(z, b_T) = exp[-b_T²/4 · λ_D² · z^{-β} · (1-z)²]
    Parameters: [λ_D, β] (index 0, 1). Supports linking and expressions in free_mask.
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
        if len(init_params) != 2:
            raise ValueError(
                f"{tcolors.FAIL}[fnp_base.py] Exponential FF requires 2 params [λ_D, β], got {len(init_params)}{tcolors.ENDC}"
            )
        if len(free_mask) != 2:
            raise ValueError(
                f"{tcolors.FAIL}[fnp_base.py] free_mask length must match init_params (2){tcolors.ENDC}"
            )

        self.flavor = flavor
        self.param_type = param_type
        self.n_params = 2
        self.registry = registry
        self.evaluator = evaluator
        self.parser = ParameterLinkParser()

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

        for param_idx, val in self.fixed_params:
            self.register_buffer(
                f"fixed_param_{param_idx}", torch.tensor([val], dtype=torch.float32)
            )
        for param_idx, param in self.free_params_list:
            self.register_parameter(f"free_param_{param_idx}", param)

    def get_params_tensor(self) -> torch.Tensor:
        params = [0.0] * self.n_params
        for param_idx, val in self.fixed_params:
            params[param_idx] = val
        for param_idx, param in self.free_params_list:
            config = self.param_configs[param_idx]
            parsed = config["parsed"]
            if parsed["type"] == "boolean" or parsed["type"] == "reference":
                params[param_idx] = (
                    param.item()
                    if param.numel() == 1
                    else (param[0].item() if len(param.shape) > 0 else param.item())
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
            -((b / 2.0) ** 2)
            * (lam_D**2)
            * torch.pow(z_safe, -beta)
            * ((1 - z) ** 2)
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
    Manager for exponential fNP parametrization with parameter linking.

    Supports free_mask: true/false, u[0], pdfs.u[1], "0.5*u[0]", "2*u[1]+0.1", etc.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__()

        self.hadron = config.get("hadron", "proton")
        self.pdf_flavor_keys = ["u", "ubar", "d", "dbar", "s", "sbar", "c", "cbar"]
        self.ff_flavor_keys = ["u", "ubar", "d", "dbar", "s", "sbar", "c", "cbar"]

        print(
            f"{tcolors.BLUE}\n[fNPManager] Initializing exponential fNP parametrization (with linking)"
        )
        print(f"  Hadron: {self.hadron}")
        print(
            "  PDF: exp[-b²/4 · λ_f² · x^{-α} · (1-x)²],  FF: exp[-b²/4 · λ_D² · z^{-β} · (1-z)²]"
        )
        print(f"  Params: PDF [λ_f, α], FF [λ_D, β]. Supports links and expressions.\n{tcolors.ENDC}")

        evolution_config = config.get("evolution", {})
        init_g2 = evolution_config.get("init_g2", 0.12840)
        free_mask = evolution_config.get("free_mask", [True])
        self.evolution = fNP_evolution(init_g2=init_g2, free_mask=free_mask)

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
                free_mask=flavor_cfg.get(
                    "free_mask", DEFAULT_FF_PARAMS["free_mask"]
                ),
                registry=self.registry,
                evaluator=self.evaluator,
                param_type="ffs",
            )
        self.ff_modules = nn.ModuleDict(ff_modules)
        print(
            f"{tcolors.GREEN}[fNPManager] Initialized {len(self.ff_modules)} FF flavor modules\n{tcolors.ENDC}"
        )

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
