#!/usr/bin/env python3
"""
Synthetic fNP parameter fit sanity check using PyTorch autograd.

This script uses the full SIDIS computation machinery with APFEL++ and LHAPDF.
REQUIRES: Python 3.10 and LHAPDF/APFEL++ packages.

It auto-discovers the repo root and the 'map' folder.
"""

import os
import sys
import argparse
import yaml
import torch
import numpy as np
from typing import Dict, Any, Tuple, Optional

# Initialize repository paths and set up import system
# First import the utilities module using a fallback method
try:
    # Try direct import if we're already in the right place
    from modules.utilities import ensure_repo_on_syspath
except ImportError:
    # Fallback: manually add map to path. Assumes the current
    # file is inside `map/<something>/this_file.py`
    script_dir = os.path.dirname(os.path.abspath(__file__))
    map_dir = os.path.dirname(script_dir)
    if map_dir not in sys.path:
        sys.path.insert(0, map_dir)
    from modules.utilities import ensure_repo_on_syspath

# Now use the centralized function to set up paths
REPO_ROOT, MAP_DIR = ensure_repo_on_syspath()

# Local imports
from modules.utilities import check_python_version, load_and_validate_kinematics


def forward_cross_sections_torch(
    comp, kin: Dict[str, Any], max_points: Optional[int] = None
) -> torch.Tensor:
    """
    Compute SIDIS cross sections using the full TMD machinery with PyTorch autograd.

    This is the core physics computation that evaluates the SIDIS differential cross section
    dsigma(x, Q, z, PhT) using TMD factorization with APFEL++ evolution and PyTorch integration.
    The computation preserves gradients for automatic differentiation.

    Args:
        comp: SIDISComputationPyTorch instance containing:
              - TMD evolution setup (APFEL++)
              - fNP model parameters (PyTorch)
              - Integration configuration (b-grid, devices, dtypes)

        kin (Dict[str, Any]): Kinematic data from load_and_validate_kinematics() containing:
                             - "header": {"Vs": sqrt(s), "target_isoscalarity": float}
                             - "data": {"x": [...], "Q2": [...], "z": [...], "PhT": [...]}

        max_points (Optional[int]): Limit computation to first N kinematic points.
                                   None means use all points. Default: None.

    Returns:
        torch.Tensor: Differential cross sections dsigma/dxdQdzdPhT in pb/GeV.
                     Shape: [N] where N is number of computed points.
                     Gradients preserved w.r.t. fNP parameters for optimization.
    """
    # Extract and validate header information
    header = kin.get("header", {})
    Vs = header.get("Vs")  # Center-of-mass energy √s
    targetiso = header.get("target_isoscalarity", 0.0)  # Target nuclear isoscalarity

    # Validate kinematic variables in the header. This part is not checked in the
    # loading function, so we need to ensure they are present.
    if Vs is None:
        raise ValueError(
            f"\033[33m[forward_cross_sections_torch] Kinematics header missing 'Vs'\033[0m"
        )
    if targetiso is None:
        raise ValueError(
            f"\033[33m[forward_cross_sections_torch] Kinematics header missing 'target_isoscalarity'\033[0m"
        )

    # Initialize TMD setup. This configures APFEL++.
    # The function comp.setup_isoscalar_tmds() is defined in the
    # SIDISComputationPyTorch class.
    comp.setup_isoscalar_tmds(Vs, targetiso)

    # Extract x, Q2, z, PhT from YAML data and convert to PyTorch tensors
    x = torch.tensor(kin["data"]["x"]).to(comp.device).to(comp.integration_dtype)
    Q2 = torch.tensor(kin["data"]["Q2"]).to(comp.device).to(comp.integration_dtype)
    z = torch.tensor(kin["data"]["z"]).to(comp.device).to(comp.integration_dtype)
    PhT = torch.tensor(kin["data"]["PhT"]).to(comp.device).to(comp.integration_dtype)

    # Optionally limit to first N points for quick testing
    if max_points is not None:
        x = x[:max_points]
        Q2 = Q2[:max_points]
        z = z[:max_points]
        PhT = PhT[:max_points]

    # Compute derived kinematic variables. .to(comp.device) moves the tensor
    # to a target device (CPU/GPU/MPS). .to(comp.integration_dtype) converts
    # the tensor to a target dtype (precision)
    Q = torch.sqrt(Q2).to(comp.device).to(comp.integration_dtype)
    qT = (PhT / z).to(comp.device).to(comp.integration_dtype)

    # Initialize result tensor
    # theo will store the computed cross sections for each kinematic point
    theo = torch.zeros_like(qT, dtype=comp.dtype, device=comp.device)

    # Loop over each kinematic point
    for i in range(qT.shape[0]):

        # Convert tensors to floats for APFEL++ (C++ interface requirement)
        qTm = float(qT[i].detach().cpu().numpy())  # qT value for point i
        Qm = float(Q[i].detach().cpu().numpy())  # Q value for point i
        xm = float(x[i].detach().cpu().numpy())  # x value for point i
        zm = float(z[i].detach().cpu().numpy())  # z value for point i

        # Apply kinematic cuts
        # Skip points where qT is too large relative to Q (outside TMD validity)
        if qTm > comp.qToQcut * Qm:
            continue  # Leave theo[i] = 0 for invalid kinematic points

        # SIDIS cross section prefactor
        Yp = 1 + (1 - (Qm / Vs) ** 2 / xm) ** 2

        # Pre-compute APFEL++ luminosity constants
        # L_b contains all the hard factors, PDFs, FFs, and Sudakov factors
        # evaluated on the b-grid. This is NOT differentiable (APFEL++ is in C++)
        L_b = comp._precompute_luminosity_constants(xm, zm, Qm, Yp)

        # Set up tensor variables for current kinematic point
        # Get the b-grid as PyTorch tensor
        b = comp._b_nodes_torch  # Shape: [Nb] where Nb is number of b-points

        # Convert current kinematic values to tensors with proper precision/device
        x_t = x[i].to(comp.device).to(comp.integration_dtype)
        z_t = z[i].to(comp.device).to(comp.integration_dtype)
        qT_t = qT[i].to(comp.device).to(comp.integration_dtype)
        Q_t = Q[i].to(comp.device).to(comp.integration_dtype)

        # ===== Evaluate fNP functions (THE DIFFERENTIABLE PART) =====
        # These are the only parts that maintain gradients for optimization

        # Properly sum over all quark flavors instead of just 'u'
        # This implements the missing flavor sum: Σ_q e_q^2 * f_q(x,b) * D_q(z,b)
        try:
            # Use the new flavor sum function that handles all active flavors
            flavor_sum = comp.compute_flavor_sum_pytorch(
                x_t.expand_as(b), z_t.expand_as(b), b.to(comp.dtype), Q_t
            ).to(comp.integration_dtype)
        except Exception as e:
            # Fallback to single flavor if the new function fails
            print(
                f"\033[93mWarning: Flavor sum failed, using single 'u' flavor: {e}\033[0m"
            )
            fnp_pdf = comp.compute_fnp_pytorch(
                x_t.expand_as(b), b.to(comp.dtype), "u"
            ).to(comp.integration_dtype)
            fnp_ff = comp.compute_fnp_pytorch(
                z_t.expand_as(b), b.to(comp.dtype), "u"
            ).to(comp.integration_dtype)
            flavor_sum = fnp_pdf * fnp_ff  # Single flavor approximation

        # ===== Compute Fourier-Bessel transform kernel =====
        # Zeroth-order Bessel function
        J0 = comp._bessel_j0_torch(qT_t * b)

        # Build the full integrand
        # L_b already contains the APFEL++ TMD evolution factors for all flavors
        integrand = b * J0 * flavor_sum * L_b

        # Perform numerical integration over b
        xs = torch.trapz(integrand, b)

        # Convert the integrated result to physical cross section units (pb/GeV²)
        # The factors come from the SIDIS cross section formula
        differential_xsec = (
            torch.tensor(4.0 * np.pi, device=comp.device, dtype=comp.integration_dtype)
            * qT_t  # qT factor from phase space
            * xs  # Integrated TMD result
            / (2.0 * Q_t)  # DIS normalization
            / z_t  # Fragmentation normalization
        )

        # Store result with proper dtype. Convert from integration
        # precision back to model precision
        theo[i] = differential_xsec.to(theo.dtype)

    return theo


def get_trainable_vector(model: torch.nn.Module) -> torch.Tensor:
    """
    Extract all trainable parameters from a PyTorch model into a single flat vector.

    This function is used for parameter difference analysis during optimization.
    It concatenates all parameters that have requires_grad=True into one tensor.

    Args:
        model (torch.nn.Module): PyTorch model (e.g., fNP model) containing parameters.
                                Parameters with requires_grad=False are ignored.

    Returns:
        torch.Tensor: Flattened 1D tensor containing all trainable parameters.
                     Shape: [total_params] where total_params is sum of all parameter sizes.
                     Empty tensor if no trainable parameters found.
    """
    vec = []
    for _, p in model.named_parameters():
        if p.requires_grad:
            vec.append(p.view(-1))
    if not vec:
        return torch.tensor([])
    return torch.cat(vec)


def randomize_fnp_parameters(model: torch.nn.Module, scale: float = 0.2, seed: int = 0):
    """
    Add random noise to fNP model parameters to create a fitting challenge.

    This function perturbs the fNP parameters away from their initial values to simulate
    the situation where we need to fit parameters to data. It's essential for testing
    that the optimization actually works and that gradients are computed correctly.

    Args:
        model (torch.nn.Module): fNP model whose parameters will be randomized.
                                Only parameters with requires_grad=True are modified.

        scale (float): Standard deviation of Gaussian noise added to parameters.
                      Larger values create bigger perturbations. Default: 0.2.

        seed (int): Random seed for reproducible parameter perturbations.
                   Same seed gives same randomization. Default: 0.

    Modifies:
        The input model's parameters are modified in-place:
        - model.NPevolution.free_g2: g2 evolution parameters (if present)
        - model.flavors[flavor].free_params: Flavor-specific parameters (if present)
    """
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
    """
    Main function to run the fitting procedure.
    """
    # Check Python version requirement
    check_python_version()

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
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=5e-2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    kin = load_and_validate_kinematics(args.kinematics)

    # Initialize SIDIS computation with proper error handling
    try:
        from modules.sidis import SIDISComputationPyTorch

        comp = SIDISComputationPyTorch(args.config, args.fnp_config, device=args.device)
        if comp.model_fNP is None:
            raise RuntimeError(
                f"\033[33m[main] fNP model failed to load. Check fnp_config path and contents.\033[0m"
            )
        comp.model_fNP.train()

        # Generate synthetic targets
        with torch.no_grad():
            y_true = forward_cross_sections_torch(
                comp, kin, max_points=args.points
            ).detach()

        # Initialize optimization
        true_vec = get_trainable_vector(comp.model_fNP).detach().clone()
        randomize_fnp_parameters(comp.model_fNP, scale=0.3, seed=args.seed)
        opt = torch.optim.Adam(
            [p for p in comp.model_fNP.parameters() if p.requires_grad], lr=args.lr
        )

    except ModuleNotFoundError as e:
        print(f"ERROR: Required dependencies not found: {e}")
        print("This script requires LHAPDF and APFEL++ packages.")
        print(
            "Make sure you're running with Python 3.10 and have these packages installed."
        )
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: Failed to initialize SIDIS computation: {e}")
        print("Check your configuration files and dependencies.")
        sys.exit(1)

    def chi_square(pred: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
        """Compute chi-squared loss with relative uncertainties."""
        sigma = torch.clamp(0.1 * truth.abs(), min=1e-3).to(pred.device)
        return torch.sum(((pred - truth) / sigma) ** 2)

    # Training loop
    for epoch in range(1, args.epochs + 1):
        opt.zero_grad(set_to_none=True)
        y_pred = forward_cross_sections_torch(comp, kin, max_points=args.points)
        loss = chi_square(y_pred, y_true)
        loss.backward()
        opt.step()
        if epoch % max(1, args.epochs // 10) == 0:
            print(f"Epoch {epoch:4d}/{args.epochs}  chi2 = {loss.item():.4e}")

    # Analyze results
    fit_vec = get_trainable_vector(comp.model_fNP).detach()
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
