#!/usr/bin/env python3
"""
Synthetic fNP parameter fit sanity check using PyTorch autograd.

This script:
- Loads kinematic points from a YAML (map/inputs/kinematics.yaml).
- Builds SIDISComputationPyTorch with APFEL objects and a PyTorch fNP model.
- Generates synthetic "truth" cross sections using the default fNP parameters.
- Randomizes the fNP trainable parameters and runs a simple chi-square minimization
  to recover the truth, verifying that .backward() reaches fNP parameters.

Note: This uses the Torch-native b-integral implemented inside SIDISComputationPyTorch
so gradients propagate through the fNP model only (APFEL luminosity is treated as constants).
"""

import os
import sys
import argparse
import yaml
import torch
import numpy as np
from typing import Dict, Any, Tuple, Optional

# Ensure repo root is on sys.path so we can import from the 'map' folder
THIS_DIR = os.path.dirname(__file__)
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
if REPO_ROOT not in sys.path:
"""
    sys.path.insert(0, REPO_ROOT)

from map.sidis_crossect_torch import SIDISComputationPyTorch


def load_kinematics(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError("Invalid kinematics YAML format")
    return data


def tensors_from_kinematics(comp: SIDISComputationPyTorch, data: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    kd = {}
from map.modules.fNP import fNP
        kd[key] = torch.tensor(data["data"][key], dtype=comp.dtype, device=comp.device)
    return kd["x"], kd["Q2"], kd["z"], kd["PhT"]


def forward_cross_sections(comp: SIDISComputationPyTorch, kin: Dict[str, Any], max_points: Optional[int] = None) -> torch.Tensor:
    """
    Forward compute differential cross sections as a Torch tensor for the given comp.model_fNP.
    Mirrors compute_sidis_cross_section_pytorch but returns a tensor to allow autograd.
    """
    header = kin.get("header", {})
    Vs = header.get("Vs")
    targetiso = header.get("target_isoscalarity", 0.0)
    if Vs is None:
        raise ValueError("Kinematics header missing 'Vs'")

    comp.setup_isoscalar_tmds(Vs, targetiso)

    x, Q2, z, PhT = tensors_from_kinematics(comp, kin)
    if max_points is not None:
        x = x[:max_points]
        Q2 = Q2[:max_points]
        z = z[:max_points]
        PhT = PhT[:max_points]

    Q = torch.sqrt(Q2).to(comp.device).to(comp.integration_dtype)
    qT = (PhT / z).to(comp.device).to(comp.integration_dtype)
    def tensors_from_kinematics(dtype: torch.dtype, device: torch.device, data: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    theo = torch.zeros_like(qT, dtype=comp.dtype, device=comp.device)

            kd[key] = torch.tensor(data["data"][key], dtype=dtype, device=device)
        qTm = float(qT[i].detach().cpu().numpy())
        Qm = float(Q[i].detach().cpu().numpy())
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


    def forward_cross_sections(model: fNP, kin: Dict[str, Any], device: torch.device, dtype: torch.dtype, max_points: Optional[int] = None) -> torch.Tensor:
        zm = float(z[i].detach().cpu().numpy())
        Forward compute synthetic differential cross sections as a Torch tensor using the fNP model directly.
        Uses a toy luminosity factor to avoid external dependencies; preserves gradients through fNP.

        Yp = 1 + (1 - (Qm / Vs) ** 2 / xm) ** 2
        L_b = comp._precompute_luminosity_constants(xm, zm, Qm, Yp)  # constant wrt fNP

        b = comp._b_nodes_torch
        x_t = x[i].to(comp.device).to(comp.integration_dtype)
        z_t = z[i].to(comp.device).to(comp.integration_dtype)
        model.setup_isoscalar_tmds(Vs, targetiso)
        Q_t = Q[i].to(comp.device).to(comp.integration_dtype)
        x, Q2, z, PhT = tensors_from_kinematics(dtype, device, kin)
        # Evaluate fNP for a representative flavor for both PDF and FF pieces
        pdf_flavor = "u"
        ff_flavor = "u"
        fnp_pdf = comp.compute_fnp_pytorch(x_t.expand_as(b), b.to(comp.dtype), pdf_flavor).to(comp.integration_dtype)
        fnp_ff = comp.compute_fnp_pytorch(z_t.expand_as(b), b.to(comp.dtype), ff_flavor).to(comp.integration_dtype)

        Q = torch.sqrt(Q2).to(device).to(torch.float64)
        qT = (PhT / z).to(device).to(torch.float64)
        xs = torch.trapz(integrand, b)
        theo = torch.zeros_like(qT, dtype=dtype, device=device)

        differential_xsec = (
            torch.tensor(1.0 * 4.0 * np.pi, device=comp.device, dtype=comp.integration_dtype)  # ap.constants.ConvFact*FourPi ~ const
            * qT_t
            * xs
            / (2.0 * Q_t)
            if qTm > 0.3 * Qm:
        )
        theo[i] = differential_xsec.to(theo.dtype)
            vec.append(p.view(-1))

            # Toy luminosity factor (constant wrt model params): exponential damping with Q and z
            beta = 0.05 + 0.02 * np.log(1.0 + Qm) + 0.01 / max(zm, 1e-3)
            L_b = torch.exp(-torch.tensor(beta, dtype=torch.float64, device=device) * b * b)
        return torch.tensor([])
            x_t = x[i].to(device).to(torch.float64)
            z_t = z[i].to(device).to(torch.float64)
            qT_t = qT[i]
            Q_t = Q[i]
    return torch.cat(vec)
            # Evaluate fNP for a representative flavor for both PDF and FF pieces (keep gradients)


    with torch.no_grad():
            out_pdf = model(x_t.expand_as(b).to(model.zeta.device), b.to(model.zeta.device), flavors=[pdf_flavor])
            out_ff = model(z_t.expand_as(b).to(model.zeta.device), b.to(model.zeta.device), flavors=[ff_flavor])
            fnp_pdf = out_pdf[pdf_flavor].to(torch.float64)
            fnp_ff = out_ff[ff_flavor].to(torch.float64)
        if hasattr(model, "NPevolution"):
            J0 = _bessel_j0_torch(qT_t * b)
            if model.NPevolution.free_g2.requires_grad:
                noise = torch.randn_like(model.NPevolution.free_g2) * scale
                model.NPevolution.free_g2.add_(noise)
        # Flavor params
                torch.tensor(4.0 * np.pi, device=device, dtype=torch.float64)
            if hasattr(mod, "free_params") and mod.free_params.requires_grad:
                noise = torch.randn_like(mod.free_params) * scale
                mod.free_params.add_(noise)


def main():
    parser = argparse.ArgumentParser(description="Fit fNP parameters on synthetic cross sections to test autograd")
    parser.add_argument("config", default=os.path.join(REPO_ROOT, "map/inputs/config.yaml"), nargs="?")
    parser.add_argument("kinematics", default=os.path.join(REPO_ROOT, "map/inputs/kinematics.yaml"), nargs="?")
    parser.add_argument("fnp_config", default=os.path.join(REPO_ROOT, "map/inputs/fNPconfig.yaml"), nargs="?")
    parser.add_argument("--device", default=None)
    parser.add_argument("--points", type=int, default=20, help="Max number of kinematic points to use")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=5e-2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Device selection
    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"))
    else:
        device = torch.device(args.device)

    # Load fNP model
    with open(args.fnp_config, "r") as f:
        fnp_cfg = yaml.safe_load(f)
    model = fNP(fnp_cfg).to(device)
    model.train()

    kin = load_kinematics(args.kinematics)

    # Ground-truth cross sections with default parameters
    with torch.no_grad():
        y_true = forward_cross_sections(model, kin, device=device, dtype=torch.float32, max_points=args.points).detach()

    # Snapshot true trainable vector for comparison
    true_vec = get_trainable_vector(model).detach().clone()

    # Randomize parameters to create initial guess
    randomize_fnp_parameters(model, scale=0.3, seed=args.seed)

    # Optimizer on model parameters
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    kin = load_kinematics(args.kinematics)

    if comp.model_fNP is None:
        raise RuntimeError("fNP model failed to load. Check fnp_config path and contents.")
    comp.model_fNP.train()

    # Ground-truth cross sections with default parameters
    with torch.no_grad():
        y_true = forward_cross_sections(comp, kin, max_points=args.points).detach()

    # Snapshot true trainable vector for comparison
    true_vec = get_trainable_vector(comp.model_fNP).detach().clone()

    # Randomize parameters to create initial guess
    randomize_fnp_parameters(comp.model_fNP, scale=0.3, seed=args.seed)

    # Optimizer on model parameters
    opt = torch.optim.Adam([p for p in comp.model_fNP.parameters() if p.requires_grad], lr=args.lr)

    def chi_square(pred: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
        sigma = torch.clamp(0.1 * truth.abs(), min=1e-3).to(pred.device)
        return torch.sum(((pred - truth) / sigma) ** 2)

    for epoch in range(1, args.epochs + 1):
        opt.zero_grad(set_to_none=True)
    y_pred = forward_cross_sections(model, kin, device=device, dtype=torch.float32, max_points=args.points)
        loss = chi_square(y_pred, y_true)
        loss.backward()
        opt.step()
        if epoch % max(1, args.epochs // 10) == 0:
            print(f"Epoch {epoch:4d}/{args.epochs}  chi2 = {loss.item():.4e}")

    fit_vec = get_trainable_vector(model).detach()
    delta = torch.norm(fit_vec - true_vec).item()
    rel = delta / (torch.norm(true_vec).item() + 1e-12)
    print(f"Param L2 diff: {delta:.4e} (rel {rel:.4e})")

    # Basic success heuristic
    if rel < 1e-2:
        print("SUCCESS: Parameters converged close to ground truth.")
    else:
        print("WARNING: Parameters did not fully converge. Consider more epochs, LR tuning, or point count.")


if __name__ == "__main__":
    main()
