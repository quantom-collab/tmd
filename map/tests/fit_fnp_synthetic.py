#!/usr/bin/env python3
"""
Synthetic fNP parameter fit sanity check using PyTorch autograd.

This script can run in two modes:
- Full backend: uses map.sidis_crossect_torch (requires LHAPDF/APFEL and Python 3.10)
- Toy backend: if full backend import fails, uses fNP model + toy luminosity (no external deps)

It auto-discovers the repo root and the 'map' folder, so you can run it from map/tests/.
"""

import os
import sys
import argparse
import yaml
import torch
import numpy as np
from typing import Dict, Any, Tuple, Optional


def _ensure_map_on_syspath() -> Tuple[str, str]:
    start = os.path.abspath(os.path.dirname(__file__))
    cur = start
    while True:
        candidate_map = os.path.join(cur, "map")
        if os.path.isdir(candidate_map) and os.path.isdir(
            os.path.join(candidate_map, "modules")
        ):
            repo_root = cur
            if repo_root not in sys.path:
                sys.path.insert(0, repo_root)
            if candidate_map not in sys.path:
                sys.path.insert(0, candidate_map)
            return repo_root, candidate_map
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent
    # Fallback for unusual layouts
    fallback_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if fallback_root not in sys.path:
        sys.path.insert(0, fallback_root)
    return fallback_root, os.path.join(fallback_root, "map")


REPO_ROOT, MAP_DIR = _ensure_map_on_syspath()


def load_kinematics(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError("Invalid kinematics YAML format")
    return data


def tensors_from_kinematics(
    context, data: Dict[str, Any], use_toy: bool
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    kd = {}
    for key in ["x", "Q2", "z", "PhT"]:
        if use_toy:
            dtype, device = context
            kd[key] = torch.tensor(data["data"][key], dtype=dtype, device=device)
        else:
            comp = context
            kd[key] = torch.tensor(
                data["data"][key], dtype=comp.dtype, device=comp.device
            )
    return kd["x"], kd["Q2"], kd["z"], kd["PhT"]


def forward_cross_sections_torch(
    comp, kin: Dict[str, Any], max_points: Optional[int] = None
) -> torch.Tensor:
    header = kin.get("header", {})
    Vs = header.get("Vs")
    targetiso = header.get("target_isoscalarity", 0.0)
    if Vs is None:
        raise ValueError("Kinematics header missing 'Vs'")

    comp.setup_isoscalar_tmds(Vs, targetiso)

    x, Q2, z, PhT = tensors_from_kinematics(comp, kin, use_toy=False)
    if max_points is not None:
        x = x[:max_points]
        Q2 = Q2[:max_points]
        z = z[:max_points]
        PhT = PhT[:max_points]

    Q = torch.sqrt(Q2).to(comp.device).to(comp.integration_dtype)
    qT = (PhT / z).to(comp.device).to(comp.integration_dtype)

    theo = torch.zeros_like(qT, dtype=comp.dtype, device=comp.device)

    for i in range(qT.shape[0]):
        qTm = float(qT[i].detach().cpu().numpy())
        Qm = float(Q[i].detach().cpu().numpy())
        xm = float(x[i].detach().cpu().numpy())
        zm = float(z[i].detach().cpu().numpy())
        if qTm > comp.qToQcut * Qm:
            continue

        Yp = 1 + (1 - (Qm / Vs) ** 2 / xm) ** 2
        L_b = comp._precompute_luminosity_constants(xm, zm, Qm, Yp)

        b = comp._b_nodes_torch
        x_t = x[i].to(comp.device).to(comp.integration_dtype)
        z_t = z[i].to(comp.device).to(comp.integration_dtype)
        qT_t = qT[i].to(comp.device).to(comp.integration_dtype)
        Q_t = Q[i].to(comp.device).to(comp.integration_dtype)

        pdf_flavor = "u"
        ff_flavor = "u"
        fnp_pdf = comp.compute_fnp_pytorch(
            x_t.expand_as(b), b.to(comp.dtype), pdf_flavor
        ).to(comp.integration_dtype)
        fnp_ff = comp.compute_fnp_pytorch(
            z_t.expand_as(b), b.to(comp.dtype), ff_flavor
        ).to(comp.integration_dtype)

        J0 = comp._bessel_j0_torch(qT_t * b)
        integrand = b * J0 * fnp_pdf * fnp_ff * L_b
        xs = torch.trapz(integrand, b)

        differential_xsec = (
            torch.tensor(4.0 * np.pi, device=comp.device, dtype=comp.integration_dtype)
            * qT_t
            * xs
            / (2.0 * Q_t)
            / z_t
        )
        theo[i] = differential_xsec.to(theo.dtype)

    return theo


def _bessel_j0_torch(x: torch.Tensor, n_terms: int = 30) -> torch.Tensor:
    try:
        return torch.special.bessel_j0(x)
    except AttributeError:
        y = (x * x) / 4.0
        term = torch.ones_like(x)
        s = term.clone()
        for k in range(1, n_terms + 1):
            term = term * (-y) / (k * k)
            s = s + term
        return s


def forward_cross_sections_toy(
    model: torch.nn.Module,
    kin: Dict[str, Any],
    device: torch.device,
    dtype: torch.dtype,
    max_points: Optional[int] = None,
) -> torch.Tensor:
    header = kin.get("header", {})
    Vs = header.get("Vs")
    if Vs is None:
        raise ValueError("Kinematics header missing 'Vs'")

    x, Q2, z, PhT = tensors_from_kinematics((dtype, device), kin, use_toy=True)
    if max_points is not None:
        x = x[:max_points]
        Q2 = Q2[:max_points]
        z = z[:max_points]
        PhT = PhT[:max_points]

    # Choose integration dtype: prefer float64 except on MPS (no float64 support)
    use_float64 = not (device.type == "mps")
    integ_dtype = torch.float64 if use_float64 else torch.float32

    Q = torch.sqrt(Q2).to(device).to(integ_dtype)
    qT = (PhT / z).to(device).to(integ_dtype)
    theo = torch.zeros_like(qT, dtype=dtype, device=device)

    b = torch.logspace(
        np.log10(1e-2), np.log10(2.0), steps=256, device=device, dtype=integ_dtype
    )

    for i in range(qT.shape[0]):
        qTm = float(qT[i].detach().cpu().numpy())
        Qm = float(Q[i].detach().cpu().numpy())
        xm = float(x[i].detach().cpu().numpy())
        zm = float(z[i].detach().cpu().numpy())
        if qTm > 0.3 * Qm:
            continue

        beta = 0.05 + 0.02 * np.log(1.0 + Qm) + 0.01 / max(zm, 1e-3)
        L_b = torch.exp(-torch.tensor(beta, dtype=integ_dtype, device=device) * b * b)

        x_t = x[i].to(device).to(integ_dtype)
        z_t = z[i].to(device).to(integ_dtype)
        qT_t = qT[i]
        Q_t = Q[i]

        pdf_flavor = "u"
        ff_flavor = "u"
        out_pdf = model(
            x_t.expand_as(b).to(model.zeta.device),
            b.to(model.zeta.device),
            flavors=[pdf_flavor],
        )
        out_ff = model(
            z_t.expand_as(b).to(model.zeta.device),
            b.to(model.zeta.device),
            flavors=[ff_flavor],
        )
        fnp_pdf = out_pdf[pdf_flavor].to(integ_dtype)
        fnp_ff = out_ff[ff_flavor].to(integ_dtype)

        J0 = _bessel_j0_torch(qT_t * b)
        integrand = b * J0 * fnp_pdf * fnp_ff * L_b
        xs = torch.trapz(integrand, b)

        differential_xsec = (
            torch.tensor(4.0 * np.pi, device=device, dtype=integ_dtype)
            * qT_t
            * xs
            / (2.0 * Q_t)
            / z_t
        )
        theo[i] = differential_xsec.to(theo.dtype)

    return theo


def get_trainable_vector(model: torch.nn.Module) -> torch.Tensor:
    vec = []
    for _, p in model.named_parameters():
        if p.requires_grad:
            vec.append(p.view(-1))
    if not vec:
        return torch.tensor([])
    return torch.cat(vec)


def randomize_fnp_parameters(model: torch.nn.Module, scale: float = 0.2, seed: int = 0):
    torch.manual_seed(seed)
    with torch.no_grad():
        if hasattr(model, "NPevolution") and hasattr(model.NPevolution, "free_g2"):
            if model.NPevolution.free_g2.requires_grad:
                model.NPevolution.free_g2.add_(
                    torch.randn_like(model.NPevolution.free_g2) * scale
                )
        if hasattr(model, "flavors"):
            for mod in model.flavors.values():
                if hasattr(mod, "free_params") and mod.free_params.requires_grad:
                    mod.free_params.add_(torch.randn_like(mod.free_params) * scale)


def main():
    # Version hint for LHAPDF/APFEL
    if not (sys.version_info.major == 3 and sys.version_info.minor == 10):
        print(
            f"[fit_fnp_synthetic] Note: LHAPDF/APFEL typically require Python 3.10; current is {sys.version_info.major}.{sys.version_info.minor}"
        )

    parser = argparse.ArgumentParser(
        description="Fit fNP parameters on synthetic cross sections to test autograd"
    )
    parser.add_argument(
        "config", default=os.path.join(MAP_DIR, "inputs", "config.yaml"), nargs="?"
    )
    parser.add_argument(
        "kinematics",
        default=os.path.join(MAP_DIR, "inputs", "kinematics.yaml"),
        nargs="?",
    )
    parser.add_argument(
        "fnp_config",
        default=os.path.join(MAP_DIR, "inputs", "fNPconfig.yaml"),
        nargs="?",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--points", type=int, default=20, help="Max number of kinematic points to use"
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=5e-2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    kin = load_kinematics(args.kinematics)

    # Try full backend; fallback to toy if imports fail
    use_toy = False
    try:
        from map.sidis_crossect_torch import SIDISComputationPyTorch  # type: ignore

        comp = SIDISComputationPyTorch(args.config, args.fnp_config, device=args.device)
        if comp.model_fNP is None:
            raise RuntimeError(
                "fNP model failed to load. Check fnp_config path and contents."
            )
        comp.model_fNP.train()

        with torch.no_grad():
            y_true = forward_cross_sections_torch(
                comp, kin, max_points=args.points
            ).detach()
        true_vec = get_trainable_vector(comp.model_fNP).detach().clone()
        randomize_fnp_parameters(comp.model_fNP, scale=0.3, seed=args.seed)
        opt = torch.optim.Adam(
            [p for p in comp.model_fNP.parameters() if p.requires_grad], lr=args.lr
        )
    except ModuleNotFoundError as e:
        print(f"[fit_fnp_synthetic] Falling back to toy backend: {e}")
        use_toy = True
        if args.device is None:
            device = torch.device(
                "cuda"
                if torch.cuda.is_available()
                else (
                    "mps"
                    if hasattr(torch.backends, "mps")
                    and torch.backends.mps.is_available()
                    else "cpu"
                )
            )
        else:
            device = torch.device(args.device)
        with open(args.fnp_config, "r") as f:
            fnp_cfg = yaml.safe_load(f)
        if not isinstance(fnp_cfg, dict):
            raise TypeError("fNP config must be a mapping (YAML dict)")
        from map.modules.fNP import (
            fNP,
        )  # Local import to avoid top-level dependency when not used

        model = fNP(fnp_cfg).to(device)
        model.train()
        with torch.no_grad():
            y_true = forward_cross_sections_toy(
                model, kin, device=device, dtype=torch.float32, max_points=args.points
            ).detach()
        true_vec = get_trainable_vector(model).detach().clone()
        randomize_fnp_parameters(model, scale=0.3, seed=args.seed)
        opt = torch.optim.Adam(
            [p for p in model.parameters() if p.requires_grad], lr=args.lr
        )

    def chi_square(pred: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
        sigma = torch.clamp(0.1 * truth.abs(), min=1e-3).to(pred.device)
        return torch.sum(((pred - truth) / sigma) ** 2)

    for epoch in range(1, args.epochs + 1):
        opt.zero_grad(set_to_none=True)
        if not use_toy:
            y_pred = forward_cross_sections_torch(comp, kin, max_points=args.points)  # type: ignore[name-defined]
        else:
            y_pred = forward_cross_sections_toy(model, kin, device=device, dtype=torch.float32, max_points=args.points)  # type: ignore[name-defined]
        loss = chi_square(y_pred, y_true)
        loss.backward()
        opt.step()
        if epoch % max(1, args.epochs // 10) == 0:
            print(f"Epoch {epoch:4d}/{args.epochs}  chi2 = {loss.item():.4e}")

    if not use_toy:
        fit_vec = get_trainable_vector(comp.model_fNP).detach()  # type: ignore[name-defined]
    else:
        fit_vec = get_trainable_vector(model).detach()  # type: ignore[name-defined]
    delta = torch.norm(fit_vec - true_vec).item()
    rel = delta / (torch.norm(true_vec).item() + 1e-12)
    print(f"Param L2 diff: {delta:.4e} (rel {rel:.4e})")

    if rel < 1e-2:
        print("SUCCESS: Parameters converged close to ground truth.")
    else:
        print(
            "WARNING: Parameters did not fully converge. Consider more epochs, LR tuning, or point count."
        )


if __name__ == "__main__":
    main()
