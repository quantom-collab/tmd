"""
Shared linked-parameter wiring for unpolarized TMD PDF/FF ``nn.Module`` objects.

Builds bounded/unbounded trainable parameters from ``free_mask``, registers them
on the module, and materializes a flat parameter tensor for ``forward``.
This is not a selectable fNP parametrisation; TMD model classes in
``sidis/model/fnp/`` call these helpers from ``__init__`` and ``get_params_tensor``.
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

import torch
import torch.nn as nn

# tcolors: same fallbacks as fnp_config and fnp_manager 
# when imported outside package.
try:
    from .utilities.colors import tcolors
except ImportError:
    try:
        from utilities.colors import tcolors
    except ImportError:
        from sidis.utilities.colors import tcolors

from .fnp_config import ParameterLinkParser, ParameterRegistry, parse_bound


def build_bounds_list(
    param_bounds: Optional[List[Any]],
    n_params: int,
    param_type: str,
    flavor: str,
    warn_tag: str,
) -> List[Optional[Tuple[float, float]]]:
    """
    Parse ``param_bounds`` into a list of ``(lo, hi)`` or ``None``, padded to
    ``n_params`` entries. Mirrors the historical try/warn/parse loop in tmdpdf/tmdff.

    ``warn_tag`` is inserted into the length-mismatch message (e.g. ``"[fnp/tmdpdf.py]"``).
    """
    bounds_list: List[Optional[Tuple[float, float]]] = []
    if param_bounds is not None:
        try:
            if len(param_bounds) != n_params:
                print(
                    f"{tcolors.WARNING}{warn_tag} {param_type}.{flavor}: "
                    f"param_bounds has {len(param_bounds)} entries for {n_params} parameters. "
                    f"Missing entries are treated as unbounded; extra entries are ignored.{tcolors.ENDC}"
                )
            for idx in range(min(len(param_bounds), n_params)):
                b = parse_bound(
                    param_bounds[idx] if idx < len(param_bounds) else None
                )
                bounds_list.append(b)
        except (TypeError, KeyError):
            pass
    while len(bounds_list) < n_params:
        bounds_list.append(None)
    return bounds_list


def populate_linked_params(
    module: nn.Module,
    *,
    flavor: str,
    param_type: str,
    init_params: List[float],
    free_mask: List[Any],
    registry: ParameterRegistry,
    bounds_list: List[Optional[Tuple[float, float]]],
) -> None:
    """
    Mutates ``module`` with ``param_configs``, ``fixed_params``, ``free_params_list``,
    and registers fixed buffers / free ``nn.Parameter`` submodules.

    The module must already set ``evaluator`` on itself (used by
    ``get_params_tensor_from_state`` for expression-type entries).

    Required later by ``get_params_tensor_from_state`` and by ``fNPManager``
    (e.g. ``randomize_params_in_bounds``), which expect ``param_configs`` and
    ``free_params_list`` on each PDF/FF module.
    """
    parser = ParameterLinkParser()
    module.param_configs = []
    module.fixed_params = []
    module.free_params_list = []

    n_params = len(init_params)
    for param_idx, (init_val, entry) in enumerate(zip(init_params, free_mask)):
        parsed = parser.parse_entry(entry, param_type, flavor)
        bounds = bounds_list[param_idx] if param_idx < len(bounds_list) else None
        module.param_configs.append(
            {
                "idx": param_idx,
                "init_val": init_val,
                "parsed": parsed,
                "bounds": bounds,
            }
        )

        if parsed["is_fixed"]:
            module.fixed_params.append((param_idx, init_val))
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
            module.free_params_list.append((param_idx, param))
            registry.register_parameter(
                param_type, flavor, param_idx, param, bounds=bounds
            )
        elif parsed["type"] == "reference":
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
            module.free_params_list.append((param_idx, shared_param))
            registry.register_parameter(
                param_type,
                flavor,
                param_idx,
                shared_param,
                source=(ref_type, ref["flavor"], ref["param_idx"]),
                bounds=bounds,
            )
        elif parsed["type"] == "expression":
            param = nn.Parameter(torch.tensor([init_val], dtype=torch.float32))
            module.free_params_list.append((param_idx, param))
            registry.register_parameter(
                param_type, flavor, param_idx, param, bounds=bounds
            )
            parsed["expression"] = parsed["value"]

    for param_idx, val in module.fixed_params:
        module.register_buffer(
            f"fixed_param_{param_idx}", torch.tensor([val], dtype=torch.float32)
        )
    for param_idx, param in module.free_params_list:
        module.register_parameter(f"free_param_{param_idx}", param)


def get_params_tensor_from_state(module: nn.Module) -> torch.Tensor:
    """
    Assemble the length-``n_params`` parameter vector from fixed values, bounded
    sigmoid-mapped params, and dynamically evaluated expressions.
    """
    try:
        dev = next(module.parameters()).device
    except StopIteration:
        try:
            dev = next(module.buffers()).device
        except StopIteration:
            dev = torch.device("cpu")

    n_params = module.n_params
    param_vals: List[Optional[torch.Tensor]] = [None] * n_params

    for param_idx, val in module.fixed_params:
        param_vals[param_idx] = torch.tensor(
            [float(val)], dtype=torch.float32, device=dev
        )

    for param_idx, param in module.free_params_list:
        config = module.param_configs[param_idx]
        parsed = config["parsed"]
        if parsed["type"] == "boolean" or parsed["type"] == "reference":
            bounds = config.get("bounds")
            if bounds is None and parsed["type"] == "reference":
                ref = parsed["value"]
                ref_type = ref["type"] if ref["type"] else module.param_type
                key = (ref_type, ref["flavor"], ref["param_idx"])
                bounds = module.param_bounds_map.get(key)
            if bounds is not None:
                lo, hi = bounds
                raw = torch.sigmoid(param)
                val_t = lo + (hi - lo) * raw.flatten()[0]
                param_vals[param_idx] = val_t.unsqueeze(0)
            else:
                p = param.flatten()[0]
                param_vals[param_idx] = p.unsqueeze(0)
        elif parsed["type"] == "expression":
            expr_value = module.evaluator.evaluate(
                parsed["expression"], module.param_type, module.flavor
            )
            param_vals[param_idx] = expr_value
            param.data = expr_value.detach()

    vals = [
        (
            param_vals[i]
            if param_vals[i] is not None
            else torch.tensor([0.0], dtype=torch.float32, device=dev)
        )
        for i in range(n_params)
    ]
    return torch.cat([v.flatten()[:1] for v in vals])
