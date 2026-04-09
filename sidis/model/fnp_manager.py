"""
Unified fNP manager and construction entry points.

This file is the single place to inspect the non-perturbative manager flow:
- combo selection and defaults
- per-flavor module wiring (PDF/FF/Sivers/Qiu-Sterman)
- parameter linking dependency checks
- parameter bounds bookkeeping
"""

from typing import Dict, Any, List, Optional, Tuple
import pathlib

import torch
import torch.nn as nn
from omegaconf import OmegaConf

try:
    from ..utilities.colors import tcolors
except ImportError:
    try:
        from utilities.colors import tcolors
    except ImportError:
        from sidis.utilities.colors import tcolors

from .fnp_config import (
    ParameterRegistry,
    ExpressionEvaluator,
    DependencyResolver,
    parse_bound,
    FLAVORS,
)
from .fnp.tmdpdf import TMDPDFFlexible, TMDPDFSimple
from .fnp.tmdff import TMDFFFlexible, TMDFFSimple
from .fnp.sivers import Sivers, SiversAV
from .fnp.qiu_sterman import QiuSterman, QiuStermanAV
SUPPORTED_COMBOS = {"simple", "flavor_dep", "flavor_blind", "flexible", "flexible_new", "flexible AV"}


class fNP_evolution(nn.Module):
    """Shared non-perturbative evolution factor."""

    def __init__(self, init_g2: float, free_mask: List[bool]):
        super().__init__()
        if len(free_mask) != 1:
            raise ValueError(
                f"[fnp_manager.py] {tcolors.FAIL}Evolution free_mask must have exactly 1 element, got {len(free_mask)}{tcolors.ENDC}"
            )

        self.register_buffer("Q0_squared", torch.tensor(1.0, dtype=torch.float32))
        mask = torch.tensor(free_mask, dtype=torch.float32)
        self.register_buffer("g2_mask", mask)

        init_tensor = torch.tensor([init_g2], dtype=torch.float32)
        self.register_buffer("fixed_g2", init_tensor * (1 - mask))
        self.free_g2 = nn.Parameter(init_tensor * mask)
        self.free_g2.register_hook(lambda grad: grad * self.g2_mask)

    @property
    def g2(self):
        return self.fixed_g2 + self.free_g2

    def forward(self, b: torch.Tensor, zeta: torch.Tensor) -> torch.Tensor:
        if b.dim() > zeta.dim():
            zeta = zeta.unsqueeze(-1)
        return torch.exp(-(self.g2**2) * (b**2) * torch.log(zeta / self.Q0_squared) / 4.0)


MAP22_DEFAULT_PDF_PARAMS = {
    "init_params": [
        0.28516,
        0.29755,
        0.17293,
        0.39432,
        0.28516,
        0.28516,
        0.39432,
        0.29755,
        0.29755,
        0.17293,
        0.17293,
    ],
    "free_mask": [True] * 11,
}

MAP22_DEFAULT_FF_PARAMS = {
    "init_params": [
        0.21012,
        2.12062,
        0.093554,
        0.25246,
        5.2915,
        0.033798,
        2.1012,
        0.093554,
        0.25246,
    ],
    "free_mask": [True] * 9,
}

DEFAULT_SIVERS_PARAMS = {"init_params": [0.045], "free_mask": [True]}

DEFAULT_QIU_STERMAN_PARAMS = {
    "init_params": [0.045, 0.031, 0.011],
    "free_mask": [True] * 3,
}


class fNPManager(nn.Module):
    """Unified fNP manager used by all cards."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.hadron = config.get("hadron", "proton")
        self.combo = config.get("combo", "flexible_new")
        self.flavor_keys = list(FLAVORS)
        self.pdf_flavor_keys = self.flavor_keys
        self.ff_flavor_keys = self.flavor_keys
        self.sivers_flavor_keys = self.flavor_keys
        self.qiu_sterman_flavor_keys = self.flavor_keys

        print(
            f"{tcolors.BLUE}\n[fNPManager] Initializing unified fNP manager\n"
            f"  combo: {self.combo}\n"
            f"  hadron: {self.hadron}\n"
            f"  flavors: {len(self.flavor_keys)}\n{tcolors.ENDC}"
        )

        evolution_config = config.get("evolution", {})
        self.evolution = fNP_evolution(
            init_g2=evolution_config.get("init_g2", 0.12840),
            free_mask=evolution_config.get("free_mask", [True]),
        )

        self.registry = ParameterRegistry()
        self.evaluator = ExpressionEvaluator(self.registry)
        self.sivers_flag = config.get("polarization", "unpolarized") == "transverse"
        self.qiu_sterman_flag = self.sivers_flag

        # Validate linking dependency graphs before module creation.
        pdf_graph = DependencyResolver.build_dependency_graph(
            config, "pdfs", self.pdf_flavor_keys
        )
        ff_graph = DependencyResolver.build_dependency_graph(
            config, "ffs", self.ff_flavor_keys
        )
        DependencyResolver.resolve_circular_dependencies(pdf_graph)
        DependencyResolver.resolve_circular_dependencies(ff_graph)
        if self.sivers_flag:
            sivers_graph = DependencyResolver.build_dependency_graph(
                config, "sivers", self.sivers_flavor_keys
            )
            DependencyResolver.resolve_circular_dependencies(sivers_graph)
        if self.qiu_sterman_flag:
            qiu_graph = DependencyResolver.build_dependency_graph(
                config, "qiu_sterman", self.qiu_sterman_flavor_keys
            )
            DependencyResolver.resolve_circular_dependencies(qiu_graph)

        pdf_param_classes = {"flexible": TMDPDFFlexible, "simple": TMDPDFSimple, "flexible AV": TMDPDFFlexible}
        ff_param_classes = {"flexible": TMDFFFlexible, "simple": TMDFFSimple, "flexible AV": TMDFFFlexible}
        if self.sivers_flag:
            sivers_param_classes = {"flexible": Sivers, "flexible AV": SiversAV}
        if self.qiu_sterman_flag:
            qiu_sterman_param_classes = {"flexible": QiuSterman, "flexible AV": QiuStermanAV}
        default_parametrization = self.combo if "flavor" not in self.combo else "flexible"
 
        # Build modules first.
        self.pdf_modules = self._build_module(
            np_type="pdfs",
            config=config,
            flavor_keys=self.pdf_flavor_keys,
            class_map=pdf_param_classes,
            default_cfg=MAP22_DEFAULT_PDF_PARAMS,
            default_parametrization=default_parametrization,
            include_bounds=True,
        )
        self.ff_modules = self._build_module(
            np_type="ffs",
            config=config,
            flavor_keys=self.ff_flavor_keys,
            class_map=ff_param_classes,
            default_cfg=MAP22_DEFAULT_FF_PARAMS,
            default_parametrization=default_parametrization,
            include_bounds=True,
        )
        if self.sivers_flag:
            self.sivers_modules = self._build_module(
                np_type="sivers",
                config=config,
                flavor_keys=self.sivers_flavor_keys,
                class_map=sivers_param_classes,
                default_cfg=DEFAULT_SIVERS_PARAMS,
                default_parametrization=default_parametrization,
                include_bounds=False,
            )
        else:
            self.sivers_modules = None

        if self.qiu_sterman_flag:
            self.qiu_sterman_modules = self._build_module(
                np_type="qiu_sterman",
                config=config,
                flavor_keys=self.qiu_sterman_flavor_keys,
                class_map=qiu_sterman_param_classes,
                default_cfg=DEFAULT_QIU_STERMAN_PARAMS,
                default_parametrization=default_parametrization,
                include_bounds=False,
            )
        else:
            self.qiu_sterman_modules = None

        # Then compute bounds.
        self.param_bounds = self._collect_param_bounds(config)
        self.evolution_bounds = self._collect_evolution_bounds(config)
        self._propagate_param_bounds_map()

    def _build_module(
        self,
        np_type: str,
        config: Dict[str, Any],
        flavor_keys: List[str],
        class_map: Dict[str, type],
        default_cfg: Dict[str, Any],
        default_parametrization: str,
        include_bounds: bool,
    ) -> nn.ModuleDict:
        modules = {}
        type_config = config.get(np_type, {})
        for flavor in flavor_keys:
            flavor_cfg = type_config.get(flavor, None) or default_cfg.copy()
            parametrization = flavor_cfg.get("parametrization", default_parametrization)
            if parametrization not in class_map:
                available = ", ".join(sorted(class_map.keys()))
                raise ValueError(
                    f"Unknown {np_type} parametrization '{parametrization}' for flavor '{flavor}'. "
                    f"Available: {available}"
                )
            cls = class_map[parametrization]
            kwargs = dict(
                flavor=flavor,
                init_params=flavor_cfg.get("init_params", default_cfg["init_params"]),
                free_mask=flavor_cfg.get("free_mask", default_cfg["free_mask"]),
                registry=self.registry,
                evaluator=self.evaluator,
                param_type=np_type,
            )
            if include_bounds:
                kwargs["param_bounds"] = flavor_cfg.get("param_bounds")
                kwargs["param_bounds_map"] = {}
            modules[flavor] = cls(**kwargs)
        return nn.ModuleDict(modules)

    def _collect_param_bounds(self, config: Dict[str, Any]) -> Dict[Tuple[str, str, int], Tuple[float, float]]:
        out: Dict[Tuple[str, str, int], Tuple[float, float]] = {}
        param_types = ["pdfs", "ffs"]
        if self.sivers_flag:
            param_types.append("sivers")
        if self.qiu_sterman_flag:
            param_types.append("qiu_sterman")
        for param_type in param_types:
            type_config = config.get(param_type, {})
            flavor_keys = (
                self.pdf_flavor_keys
                if param_type == "pdfs"
                else self.ff_flavor_keys
                if param_type == "ffs"
                else self.sivers_flavor_keys
                if param_type == "sivers"
                else self.qiu_sterman_flavor_keys
            )
            for flavor in flavor_keys:
                flavor_cfg = type_config.get(flavor, {})
                if flavor_cfg is None:
                    continue
                bounds_list = flavor_cfg.get("param_bounds") if hasattr(flavor_cfg, "get") else None
                if bounds_list is None:
                    continue
                n_params = len(flavor_cfg.get("init_params", []))
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
                    key = (param_type, flavor, idx)
                    if key in self.registry.registry and key not in getattr(
                        self.registry, "shared_groups", {}
                    ):
                        out[key] = parsed
        return out

    def _collect_evolution_bounds(self, config: Dict[str, Any]) -> List[Tuple[float, float]]:
        out: List[Tuple[float, float]] = []
        ev_bounds = config.get("evolution", {}).get("param_bounds")
        if ev_bounds is None:
            return out
        try:
            for b in ev_bounds:
                parsed = parse_bound(b)
                if parsed is not None:
                    out.append(parsed)
        except TypeError:
            pass
        return out

    def _propagate_param_bounds_map(self) -> None:
        """Attach global bounds map to all modules that can use linked bounds."""
        module_groups: List[nn.Module] = [*self.pdf_modules.values(), *self.ff_modules.values()]
        if self.sivers_modules is not None:
            module_groups.extend(self.sivers_modules.values())
        if self.qiu_sterman_modules is not None:
            module_groups.extend(self.qiu_sterman_modules.values())
        for module in module_groups:
            if hasattr(module, "param_bounds_map"):
                module.param_bounds_map = self.param_bounds

    def get_trainable_bounds(
        self,
    ) -> List[Tuple[nn.Parameter, Optional[float], Optional[float]]]:
        result: List[Tuple[nn.Parameter, Optional[float], Optional[float]]] = []
        # PDF/FF bounded parameters are optimized in theta-space and mapped through
        # sigmoid to physical bounds inside each module. Clamping theta to physical
        # bounds is incorrect and hurts fit quality. Keep post-step clamping only for
        # evolution parameters that are represented directly in physical space.
        if self.evolution_bounds and hasattr(self.evolution, "free_g2"):
            p = self.evolution.free_g2
            if p.requires_grad and len(self.evolution_bounds) > 0:
                lo, hi = self.evolution_bounds[0]
                result.append((p, lo, hi))
        return result

    def clamp_parameters_to_bounds(self) -> None:
        for param, low, high in self.get_trainable_bounds():
            if low is not None and high is not None:
                with torch.no_grad():
                    param.data.clamp_(low, high)

    def randomize_params_in_bounds(self, seed: int = 42) -> None:
        """
        Randomize bounded trainable parameters in a reproducible way.

        - For bounded PDF/FF params, modules optimize unbounded ``theta`` and map
          physical values through ``sigmoid(theta)``. We therefore sample
          ``u ~ Uniform(0, 1)`` and set ``theta = logit(u)``.
        - For unbounded params, no randomization is applied.
        - Evolution g2 is randomized directly in physical space using its bounds.
        """
        try:
            dev = next(self.parameters()).device
        except StopIteration:
            dev = torch.device("cpu")

        gen = torch.Generator(device=dev)
        gen.manual_seed(seed)

        # Avoid re-randomizing shared-reference parameters.
        seen: set = set()

        # Include polarized modules when enabled.
        module_groups: List[nn.Module] = [*self.pdf_modules.values(), *self.ff_modules.values()]
        if self.sivers_modules is not None:
            module_groups.extend(self.sivers_modules.values())
        if self.qiu_sterman_modules is not None:
            module_groups.extend(self.qiu_sterman_modules.values())

        for module in module_groups:
            if not hasattr(module, "free_params_list") or not hasattr(module, "param_configs"):
                continue
            for param_idx, param in module.free_params_list:
                if id(param) in seen:
                    continue
                config = module.param_configs[param_idx]
                parsed = config.get("parsed", {})
                bounds = config.get("bounds")

                # Reference params may inherit bounds from their source.
                if bounds is None and parsed.get("type") == "reference":
                    ref = parsed["value"]
                    ref_type = ref["type"] if ref["type"] else getattr(module, "param_type", None)
                    ref_key = (ref_type, ref["flavor"], ref["param_idx"])
                    bounds = self.param_bounds.get(ref_key)

                if bounds is not None:
                    seen.add(id(param))
                    u = torch.rand(1, generator=gen, device=param.device, dtype=param.dtype)
                    u = u.clamp(1e-6, 1 - 1e-6)
                    theta = torch.logit(u)
                    with torch.no_grad():
                        param.data.copy_(theta)

        # Evolution g2 is not sigmoid-reparameterized; randomize directly.
        if self.evolution_bounds and hasattr(self.evolution, "free_g2"):
            p = self.evolution.free_g2
            if p.requires_grad and len(self.evolution_bounds) > 0:
                lo, hi = self.evolution_bounds[0]
                u = torch.rand(1, generator=gen, device=p.device, dtype=p.dtype)
                with torch.no_grad():
                    p.data.copy_(lo + (hi - lo) * u)

    def _compute_zeta(self, Q: torch.Tensor) -> torch.Tensor:
        return Q**2

    def forward_evolution(self, b: torch.Tensor, Q: torch.Tensor) -> torch.Tensor:
        return self.evolution(b, self._compute_zeta(Q))

    def forward_pdf(
        self, x: torch.Tensor, b: torch.Tensor, flavors: Optional[List[str]] = None
    ) -> Dict[str, torch.Tensor]:
        flavors = self.pdf_flavor_keys if flavors is None else flavors
        return {fl: self.pdf_modules[fl](x, b, 0) for fl in flavors}

    def forward_ff(
        self, z: torch.Tensor, b: torch.Tensor, flavors: Optional[List[str]] = None
    ) -> Dict[str, torch.Tensor]:
        flavors = self.ff_flavor_keys if flavors is None else flavors
        return {fl: self.ff_modules[fl](z, b, 0) for fl in flavors}

    def forward_sivers(
        self, x: torch.Tensor, b: torch.Tensor, flavors: Optional[List[str]] = None
    ) -> Dict[str, torch.Tensor]:
        if not self.sivers_flag or self.sivers_modules is None:
            raise RuntimeError(
                "Sivers modules are not initialized. Set polarization='transverse' to enable Sivers."
            )
        flavors = self.sivers_flavor_keys if flavors is None else flavors
        return {fl: self.sivers_modules[fl](x, b) for fl in flavors}

    def forward_qiu_sterman(
        self, x: torch.Tensor, b: torch.Tensor, flavors: Optional[List[str]] = None
    ) -> Dict[str, torch.Tensor]:
        if not self.qiu_sterman_flag or self.qiu_sterman_modules is None:
            raise RuntimeError(
                "Qiu-Sterman modules are not initialized. Set polarization='transverse' to enable Qiu-Sterman."
            )
        flavors = self.qiu_sterman_flavor_keys if flavors is None else flavors
        return {fl: self.qiu_sterman_modules[fl](x, b) for fl in flavors}

    def forward(
        self,
        x: torch.Tensor,
        z: torch.Tensor,
        b: torch.Tensor,
        Q: torch.Tensor,
        pdf_flavors: Optional[List[str]] = None,
        ff_flavors: Optional[List[str]] = None,
        sivers_flavors: Optional[List[str]] = None,
        qiu_sterman_flavors: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        x = x.unsqueeze(-1)
        z = z.unsqueeze(-1)
        out = {
            "pdfs": self.forward_pdf(x, b, pdf_flavors),
            "ffs": self.forward_ff(z, b, ff_flavors),
            "evolution": self.forward_evolution(b, Q),
        }
        out["sivers"] = self.forward_sivers(x, b, sivers_flavors) if self.sivers_flag else None
        out["qiu_sterman"] = (
            self.forward_qiu_sterman(x, b, qiu_sterman_flavors) if self.qiu_sterman_flag else None
        )
        return out


def _load_config(config_path: str = None, config_dict: Dict[str, Any] = None) -> Dict[str, Any]:
    if config_dict is not None:
        return config_dict
    if config_path is None:
        raise ValueError("Either config_path or config_dict must be provided")
    config_path_obj = pathlib.Path(config_path)
    if not config_path_obj.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    return OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)


def create_fnp_manager(config_path: str = None, config_dict: Dict[str, Any] = None):
    config = _load_config(config_path=config_path, config_dict=config_dict)
    combo_name = config.get("combo", "flexible_new")
    if combo_name not in SUPPORTED_COMBOS:
        available = ", ".join(sorted(SUPPORTED_COMBOS))
        raise ValueError(f"Unknown combo '{combo_name}'. Available combos: {available}")
    return fNPManager(config)


def create_fnp_manager_from_dict(config_dict: Dict[str, Any]):
    return create_fnp_manager(config_dict=config_dict)


def create_fnp_manager_from_file(config_path: str):
    return create_fnp_manager(config_path=config_path)
