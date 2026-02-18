"""
Flexible fNP Implementation with Parameter Linking

This module provides a flexible fNP implementation that supports parameter linking
and constraints between flavors. Parameters can be:
- Independent (boolean true/false)
- Linked to another parameter (e.g., u[0] means link to parameter 0 of flavor u)
- Complex expressions (e.g., 2*u[1] + 0.1 means parameter = 2 * u[1] + 0.1, evaluated dynamically)

Contents of this file:
- Flexible PDF/FF Base Classes: Modified PDF/FF classes that support parameter linking
- Manager class (fNPManager): Orchestrates the flexible combo implementation

Config parsing (ParameterLinkParser, ParameterRegistry, etc.) is in fnp_config.py.

Author: Chiara Bissolotti (cbissolotti@anl.gov)
Based on: fnp_base_flavor_dep.py
"""

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

# Import evolution and base classes from flavor_dep
from .fnp_base_flavor_dep import (
    fNP_evolution,
    MAP22_DEFAULT_EVOLUTION,
    MAP22_DEFAULT_PDF_PARAMS,
    MAP22_DEFAULT_FF_PARAMS,
)

# Import config parsing utilities (shared across all models)
from .fnp_config import (
    ParameterLinkParser,
    ParameterRegistry,
    ExpressionEvaluator,
    DependencyResolver,
    parse_bound,
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


###############################################################################
# 7. Flexible Manager Class
###############################################################################
class fNPManager(nn.Module):
    """
    Manager for flexible fNP system with parameter linking.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__()

        self.hadron = config.get("hadron", "proton")
        self.pdf_flavor_keys = ["u", "ubar", "d", "dbar", "s", "sbar", "c", "cbar"]
        self.ff_flavor_keys = ["u", "ubar", "d", "dbar", "s", "sbar", "c", "cbar"]

        print(
            f"{tcolors.BLUE}\n[fNPManager] Initializing flexible fNP manager with parameter linking"
        )
        print(f"  Hadron: {self.hadron}")
        print(f"  Total number of flavors: {len(self.pdf_flavor_keys)}\n{tcolors.ENDC}")

        # Setup evolution (independent, no linking)
        evolution_config = config.get("evolution", {})
        init_g2 = evolution_config.get("init_g2", 0.12840)
        free_mask = evolution_config.get("free_mask", [True])
        self.evolution = fNP_evolution(init_g2=init_g2, free_mask=free_mask)

        # Initialize parameter registry and evaluator
        self.registry = ParameterRegistry()
        self.evaluator = ExpressionEvaluator(self.registry)

        # Build dependency graphs
        pdf_graph = DependencyResolver.build_dependency_graph(
            config, "pdfs", self.pdf_flavor_keys
        )
        ff_graph = DependencyResolver.build_dependency_graph(
            config, "ffs", self.ff_flavor_keys
        )

        # Resolve circular dependencies
        pdf_resolved = DependencyResolver.resolve_circular_dependencies(pdf_graph)
        ff_resolved = DependencyResolver.resolve_circular_dependencies(ff_graph)

        # Setup PDF modules
        pdf_config = config.get("pdfs", {})
        pdf_modules = {}
        for flavor in self.pdf_flavor_keys:
            flavor_cfg = pdf_config.get(flavor, None)
            if flavor_cfg is None:
                print(
                    f"{tcolors.WARNING}[fNPManager] Warning: Using MAP22 defaults for PDF flavor '{flavor}'{tcolors.ENDC}"
                )
                flavor_cfg = MAP22_DEFAULT_PDF_PARAMS.copy()
            else:
                print(
                    f"{tcolors.OKLIGHTBLUE}[fNPManager] Using user-defined PDF flavor '{flavor}'{tcolors.ENDC}"
                )

            pdf_modules[flavor] = TMDPDFFlexible(
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

        # Setup FF modules
        ff_config = config.get("ffs", {})
        ff_modules = {}
        for flavor in self.ff_flavor_keys:
            flavor_cfg = ff_config.get(flavor, None)
            if flavor_cfg is None:
                print(
                    f"{tcolors.WARNING}[fNPManager] Warning: Using MAP22 defaults for FF flavor '{flavor}'{tcolors.ENDC}"
                )
                flavor_cfg = MAP22_DEFAULT_FF_PARAMS.copy()
            else:
                print(
                    f"{tcolors.OKLIGHTBLUE}[fNPManager] Using user-defined FF flavor '{flavor}'{tcolors.ENDC}"
                )

            ff_modules[flavor] = TMDFFFlexible(
                flavor=flavor,
                init_params=flavor_cfg.get(
                    "init_params", MAP22_DEFAULT_FF_PARAMS["init_params"]
                ),
                free_mask=flavor_cfg.get(
                    "free_mask", MAP22_DEFAULT_FF_PARAMS["free_mask"]
                ),
                registry=self.registry,
                evaluator=self.evaluator,
                param_type="ffs",
            )

        self.ff_modules = nn.ModuleDict(ff_modules)
        print(
            f"{tcolors.GREEN}[fNPManager] Initialized {len(self.ff_modules)} FF flavor modules\n{tcolors.ENDC}"
        )

        # Build parameter bounds from config (only for source parameters; linked params inherit)
        # Support both plain dicts and OmegaConf (ListConfig/list-like for bounds)
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
                bounds_list = flavor_cfg.get("param_bounds") if hasattr(flavor_cfg, "get") else None
                if bounds_list is None:
                    continue
                n_params = 11 if param_type == "pdfs" else 9
                try:
                    bound_len = len(bounds_list)
                except TypeError:
                    continue
                for idx in range(min(bound_len, n_params)):
                    try:
                        b = bounds_list[idx]
                    except (TypeError, KeyError):
                        continue
                    parsed = parse_bound(b)
                    if parsed is None:
                        continue
                    lo, hi = parsed
                    key = (param_type, flavor, idx)
                    if key in self.registry.registry and key not in getattr(
                        self.registry, "shared_groups", {}
                    ):
                        self.param_bounds[key] = (lo, hi)

        self.evolution_bounds: List[Tuple[float, float]] = []
        ev_bounds = config.get("evolution", {}).get("param_bounds")
        if ev_bounds is not None:
            try:
                for b in ev_bounds:
                    parsed = parse_bound(b)
                    if parsed is not None:
                        self.evolution_bounds.append(parsed)
            except TypeError:
                pass

    def get_trainable_bounds(
        self,
    ) -> List[Tuple[nn.Parameter, Optional[float], Optional[float]]]:
        """
        Return (parameter, low, high) for each trainable parameter that has bounds.
        Used by the minimizer to clamp or to pass bounds to constrained optimizers.
        Each parameter appears at most once (shared params included once).
        """
        result: List[Tuple[nn.Parameter, Optional[float], Optional[float]]] = []
        seen: set = set()
        for key, (lo, hi) in self.param_bounds.items():
            param = self.registry.get_parameter(key[0], key[1], key[2])
            if param is not None and id(param) not in seen and param.requires_grad:
                seen.add(id(param))
                result.append((param, lo, hi))
        if self.evolution_bounds and hasattr(self.evolution, "free_g2"):
            p = self.evolution.free_g2
            if p.requires_grad and len(self.evolution_bounds) > 0:
                lo, hi = self.evolution_bounds[0]
                result.append((p, lo, hi))
        return result

    def clamp_parameters_to_bounds(self) -> None:
        """
        Clamp all trainable parameters to their allowed intervals.
        Call this after optimizer.step() to enforce bounds.
        """
        for param, low, high in self.get_trainable_bounds():
            if low is not None and high is not None:
                with torch.no_grad():
                    param.data.clamp_(low, high)

    def _compute_zeta(self, Q: torch.Tensor) -> torch.Tensor:
        """Compute rapidity scale zeta from hard scale Q."""
        return Q**2

    def get_evolution(self, b: torch.Tensor, Q: torch.Tensor) -> torch.Tensor:
        """
        Compute and return the non-perturbative evolution factor.

        Args:
            b (torch.Tensor): Impact parameter (2D: [n_events, n_b])
            Q (torch.Tensor): Hard scale Q in GeV (1D: [n_events])

        Returns:
            torch.Tensor: Evolution factor S_NP(ζ, b_T) with shape [n_events, n_b]
        """
        zeta = self._compute_zeta(Q)
        return self.evolution(b, zeta)

    def forward_pdf(
        self,
        x: torch.Tensor,
        b: torch.Tensor,
        flavors: Optional[List[str]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Evaluate TMD PDFs for specified flavors in bare form (without evolution factor).

        Args:
            x (torch.Tensor): Bjorken x values (1D: [n_events])
            b (torch.Tensor): Impact parameter values (2D: [n_events, n_b])
            Q (torch.Tensor): Hard scale Q in GeV (1D: [n_events])
            flavors (Optional[List[str]]): List of PDF flavors to evaluate

        Returns:
            Dict[str, torch.Tensor]: Bare PDF values for each requested flavor.
                                    Shape: [n_events, n_b] for each flavor
                                    Note: Evolution factor must be applied separately by caller.
        """
        if flavors is None:
            flavors = self.pdf_flavor_keys

        outputs = {}
        for flavor in flavors:
            if flavor in self.pdf_modules:
                # Return bare PDF (no evolution applied)
                base_result = self.pdf_modules[flavor](x, b, 0)
                outputs[flavor] = base_result
            else:
                raise ValueError(f"Unknown PDF flavor: {flavor}")

        return outputs

    def forward_ff(
        self,
        z: torch.Tensor,
        b: torch.Tensor,
        flavors: Optional[List[str]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Evaluate TMD FFs for specified flavors in bare form (without evolution factor).

        Args:
            z (torch.Tensor): Energy fraction z values (1D: [n_events])
            b (torch.Tensor): Impact parameter values (2D: [n_events, n_b])
            Q (torch.Tensor): Hard scale Q in GeV (1D: [n_events])
            flavors (Optional[List[str]]): List of FF flavors to evaluate

        Returns:
            Dict[str, torch.Tensor]: Bare FF values for each requested flavor.
                                    Shape: [n_events, n_b] for each flavor
                                    Note: Evolution factor must be applied separately by caller.
        """
        if flavors is None:
            flavors = self.ff_flavor_keys

        outputs = {}
        for flavor in flavors:
            if flavor in self.ff_modules:
                # Return bare FF (no evolution applied)
                base_result = self.ff_modules[flavor](z, b, 0)
                outputs[flavor] = base_result
            else:
                raise ValueError(f"Unknown FF flavor: {flavor}")

        return outputs

    def forward_sivers(
        self,
        x: torch.Tensor,
        b: torch.Tensor,
    ) -> torch.Tensor:
        """
        Evaluate Sivers function in bare form (without evolution factor).

        Note: Sivers function support is not yet implemented in the flexible model.
        This method returns zeros with the correct shape as a placeholder.

        Args:
            x (torch.Tensor): Bjorken x values (1D: [n_events])
            b (torch.Tensor): Impact parameter values (2D: [n_events, n_b])

        Returns:
            torch.Tensor: Bare Sivers function values (currently zeros as placeholder).
                         Shape: [n_events, n_b] (same as b)
                         Note: Evolution factor must be applied separately by caller.
        """
        # Ensure x can broadcast with b
        if b.dim() > x.dim():
            x = x.unsqueeze(-1)

        # TODO: Implement Sivers function with parameter linking support
        # For now, return zeros with correct shape (bare, no evolution)
        return torch.zeros_like(b)

    def forward(
        self,
        x: torch.Tensor,
        z: torch.Tensor,
        b: torch.Tensor,
        Q: torch.Tensor,
        pdf_flavors: Optional[List[str]] = None,
        ff_flavors: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate both TMD PDFs and FFs simultaneously.

        Args:
            x (torch.Tensor): Bjorken x values (1D: [n_events])
            z (torch.Tensor): Energy fraction z values (1D: [n_events])
            b (torch.Tensor): Impact parameter values (2D: [n_events, n_b])
            Q (torch.Tensor): Hard scale Q in GeV (1D: [n_events])
            pdf_flavors (Optional[List[str]]): List of PDF flavors to evaluate
            ff_flavors (Optional[List[str]]): List of FF flavors to evaluate

        Returns:
            Dict containing:
                - 'pdfs': Dict of bare PDF values for each flavor
                - 'ffs': Dict of bare FF values for each flavor
                - 'evolution': Evolution factor tensor (computed once)
            Note:
                Evolution must be applied separately by caller: bare * evolution
        """
        x = x.unsqueeze(-1)
        z = z.unsqueeze(-1)

        # Compute evolution once
        evolution = self.get_evolution(b, Q)

        return {
            "pdfs": self.forward_pdf(x, b, pdf_flavors),
            "ffs": self.forward_ff(z, b, ff_flavors),
            "evolution": evolution,
        }
