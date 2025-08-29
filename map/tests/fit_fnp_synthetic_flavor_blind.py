#!/usr/bin/env python3
"""
Flavor-blind fNP parameter fit sanity check using PyTorch autograd.

This script demonstrates the flavor-blind fNP system where ALL flavors share
identical parameters. Unlike the standard system where each flavor has its own
parameter set, here all flavors evolve together with the same parameterization.

REQUIRES: Python 3.10 and LHAPDF/APFEL++ packages.

Key differences from fit_fnp_synthetic.py:
- Uses fNPManagerFlavorBlind instead of fNPManager
- All flavors share identical parameters (21 vs ~160 parameters)
- Dramatically faster fitting due to reduced parameter space
- Simpler parameter analysis and interpretation

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
from modules.fnp_manager_flavor_blind import (
    fNPManagerFlavorBlind,
    load_flavor_blind_config,
)


class SIDISComputationFlavorBlind:
    """
    Flavor-blind SIDIS computation class.

    This is a simplified version of SIDISComputationPyTorch that uses the
    flavor-blind fNP manager. It maintains the same interface but with
    dramatically reduced parameter count.
    """

    def __init__(
        self, config_file: str, fnp_config_file: str, device: Optional[str] = None
    ):
        """
        Initialize flavor-blind SIDIS computation.

        Args:
            config_file (str): Main configuration file path
            fnp_config_file (str): Flavor-blind fNP configuration file path
            device (Optional[str]): PyTorch device (cpu, cuda, mps)
        """
        # Load configurations
        self.config_file = config_file
        self.fnp_config_file = fnp_config_file

        # Set up device
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        print(
            f"\033[94m[SIDISComputationFlavorBlind] Using device: {self.device}\033[0m"
        )

        # Load main configuration
        with open(config_file, "r") as f:
            self.config = yaml.safe_load(f)

        # Set up computational parameters
        self.dtype = torch.float32
        self.integration_dtype = torch.float64
        self.qToQcut = self.config.get("qToQcut", 0.5)

        # Set up b-grid for integration
        self._setup_b_grid()

        # Initialize flavor-blind fNP model
        self._setup_fnp_flavor_blind()

        # Initialize APFEL++ (simplified version)
        self._initialize_apfel()

    def _setup_b_grid(self):
        """Set up impact parameter grid for integration."""
        # Simple linear grid for demonstration
        b_min = 0.01
        b_max = 10.0
        n_points = 100

        b_grid = torch.linspace(b_min, b_max, n_points, dtype=self.integration_dtype)
        self._b_nodes_torch = b_grid.to(self.device)

        print(
            f"\033[94m[SIDISComputationFlavorBlind] B-grid: {n_points} points from {b_min} to {b_max}\033[0m"
        )

    def _setup_fnp_flavor_blind(self):
        """Initialize flavor-blind fNP model."""
        try:
            print(
                f"\033[95m\nLoading flavor-blind fNP configuration from {self.fnp_config_file}\033[0m"
            )

            # Load flavor-blind configuration
            config_fnp = load_flavor_blind_config(self.fnp_config_file)

            # Initialize flavor-blind fNP manager
            self.model_fNP = fNPManagerFlavorBlind(config_fnp).to(self.device)

            print("✅ Flavor-blind PyTorch fNP model loaded successfully")
            self.model_fNP.summary()

        except Exception as e:
            print(f"\033[91m❌ Failed to load flavor-blind fNP model: {e}\033[0m")
            self.model_fNP = None
            raise

    def _initialize_apfel(self):
        """Initialize simplified APFEL++ for luminosity computation."""
        # This is a simplified version - in practice you'd need full APFEL++ setup
        print(
            "\033[94m[SIDISComputationFlavorBlind] Simplified APFEL++ initialization\033[0m"
        )

        # For demonstration, we'll use dummy luminosity values
        self._luminosity_cache = {}

    def setup_isoscalar_tmds(self, Vs: float, targetiso: float):
        """Set up TMD computation for given kinematics."""
        print(
            f"\033[94m[SIDISComputationFlavorBlind] Setting up TMDs: Vs={Vs}, targetiso={targetiso}\033[0m"
        )
        self.Vs = Vs
        self.targetiso = targetiso

    def _precompute_luminosity_constants(
        self, x: float, z: float, Q: float, Y: float
    ) -> torch.Tensor:
        """
        Precompute luminosity constants (simplified version).

        In the real implementation, this would call APFEL++ to compute the
        hard factors, collinear PDFs, FFs, and evolution kernels.
        """
        # Use cached result if available
        cache_key = (x, z, Q, Y)
        if cache_key in self._luminosity_cache:
            return self._luminosity_cache[cache_key]

        # For demonstration, create dummy luminosity values
        # In reality, this would be computed by APFEL++
        b_grid = self._b_nodes_torch
        luminosity = torch.ones_like(
            b_grid, dtype=self.integration_dtype, device=self.device
        )
        luminosity *= 0.1  # Scale factor for realistic cross sections

        self._luminosity_cache[cache_key] = luminosity
        return luminosity

    def compute_fnp_pytorch_pdf(
        self, x: torch.Tensor, b: torch.Tensor, flavor: str
    ) -> torch.Tensor:
        """Compute PDF fNP using flavor-blind manager."""
        if self.model_fNP is None:
            # Fallback to Gaussian
            return torch.exp(-0.1 * b**2)

        try:
            # All flavors return the same result in flavor-blind system
            pdf_results = self.model_fNP.forward_pdf(x, b, [flavor])
            return pdf_results[flavor]
        except Exception as e:
            print(
                f"\033[93m[SIDISComputationFlavorBlind] PDF evaluation failed: {e}\033[0m"
            )
            return torch.exp(-0.1 * b**2)

    def compute_fnp_pytorch_ff(
        self, z: torch.Tensor, b: torch.Tensor, flavor: str
    ) -> torch.Tensor:
        """Compute FF fNP using flavor-blind manager."""
        if self.model_fNP is None:
            # Fallback to Gaussian
            return torch.exp(-0.1 * b**2)

        try:
            # All flavors return the same result in flavor-blind system
            ff_results = self.model_fNP.forward_ff(z, b, [flavor])
            return ff_results[flavor]
        except Exception as e:
            print(
                f"\033[93m[SIDISComputationFlavorBlind] FF evaluation failed: {e}\033[0m"
            )
            return torch.exp(-0.1 * b**2)

    def compute_flavor_sum_pytorch(
        self, x: torch.Tensor, z: torch.Tensor, b: torch.Tensor, Q: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute flavor sum for SIDIS cross section using flavor-blind fNPs.

        In the flavor-blind system, all flavors have identical shapes, so we can
        compute one representative flavor and multiply by the appropriate factors.
        """
        if self.model_fNP is None:
            # Fallback
            return torch.exp(-0.1 * b**2)

        try:
            # In flavor-blind system, all flavors are identical
            # So we can just compute one flavor and apply electric charge weights

            # Get PDF and FF for representative flavor (all are identical)
            pdf_result = self.compute_fnp_pytorch_pdf(x, b, "u")
            ff_result = self.compute_fnp_pytorch_ff(z, b, "u")

            # Electric charges squared for active flavors
            # u, d, s, ubar, dbar, sbar (ignoring heavy quarks for simplicity)
            charge_weights = [
                (2.0 / 3.0) ** 2,  # u
                (1.0 / 3.0) ** 2,  # d
                (1.0 / 3.0) ** 2,  # s
                (2.0 / 3.0) ** 2,  # ubar
                (1.0 / 3.0) ** 2,  # dbar
                (1.0 / 3.0) ** 2,  # sbar
            ]

            # Since all flavors are identical, sum the charge weights
            total_charge_weight = sum(charge_weights)

            # Multiply by total charge weight
            flavor_sum = total_charge_weight * pdf_result * ff_result

            return flavor_sum

        except Exception as e:
            print(
                f"\033[93m[SIDISComputationFlavorBlind] Flavor sum computation failed: {e}\033[0m"
            )
            return torch.exp(-0.1 * b**2)

    def _bessel_j0_torch(self, x: torch.Tensor) -> torch.Tensor:
        """Compute Bessel J0 function using torch operations."""
        # Simple approximation for small arguments
        # For production use, you'd want a more accurate implementation
        return torch.ones_like(x) - 0.25 * x**2 + 0.015625 * x**4


def forward_cross_sections_torch_flavor_blind(
    comp, kin: Dict[str, Any], max_points: Optional[int] = None
) -> torch.Tensor:
    """
    Compute SIDIS cross sections using flavor-blind fNP system.

    This function is similar to forward_cross_sections_torch but uses the
    flavor-blind fNP manager for dramatically reduced parameter count.

    Args:
        comp: SIDISComputationFlavorBlind instance
        kin: Kinematic data dictionary
        max_points: Limit to first N kinematic points

    Returns:
        torch.Tensor: Differential cross sections
    """
    # Extract header information
    header = kin.get("header", {})
    Vs = header.get("Vs")
    targetiso = header.get("target_isoscalarity", 0.0)

    if Vs is None or targetiso is None:
        raise ValueError("Missing required kinematic header information")

    # Set up TMDs
    comp.setup_isoscalar_tmds(Vs, targetiso)

    # Extract kinematic variables
    x = torch.tensor(kin["data"]["x"]).to(comp.device).to(comp.integration_dtype)
    Q2 = torch.tensor(kin["data"]["Q2"]).to(comp.device).to(comp.integration_dtype)
    z = torch.tensor(kin["data"]["z"]).to(comp.device).to(comp.integration_dtype)
    PhT = torch.tensor(kin["data"]["PhT"]).to(comp.device).to(comp.integration_dtype)

    # Limit points if requested
    if max_points is not None:
        x = x[:max_points]
        Q2 = Q2[:max_points]
        z = z[:max_points]
        PhT = PhT[:max_points]

    # Derived kinematics
    Q = torch.sqrt(Q2)
    qT = PhT / z

    # Initialize results
    theo = torch.zeros_like(qT, dtype=comp.dtype, device=comp.device)

    print(
        f"\033[96m[forward_cross_sections_torch_flavor_blind] Computing {qT.shape[0]} points with flavor-blind fNP\033[0m"
    )

    # Loop over kinematic points
    for i in range(qT.shape[0]):

        # Extract values for this point
        qTm = float(qT[i].detach().cpu().numpy())
        Qm = float(Q[i].detach().cpu().numpy())
        xm = float(x[i].detach().cpu().numpy())
        zm = float(z[i].detach().cpu().numpy())

        # Apply kinematic cuts
        if qTm > comp.qToQcut * Qm:
            continue

        # SIDIS prefactor
        Yp = 1 + (1 - (Qm / Vs) ** 2 / xm) ** 2

        # Precompute luminosity
        L_b = comp._precompute_luminosity_constants(xm, zm, Qm, Yp)

        # Set up tensors for this point
        b = comp._b_nodes_torch
        x_t = x[i].to(comp.device).to(comp.integration_dtype)
        z_t = z[i].to(comp.device).to(comp.integration_dtype)
        qT_t = qT[i].to(comp.device).to(comp.integration_dtype)
        Q_t = Q[i].to(comp.device).to(comp.integration_dtype)

        # Compute flavor sum using flavor-blind fNP
        flavor_sum = comp.compute_flavor_sum_pytorch(
            x_t.expand_as(b), z_t.expand_as(b), b.to(comp.dtype), Q_t
        ).to(comp.integration_dtype)

        # Fourier-Bessel transform
        J0 = comp._bessel_j0_torch(qT_t * b)

        # Build integrand
        integrand = b * J0 * flavor_sum * L_b

        # Integrate
        xs = torch.trapz(integrand, b)

        # Convert to cross section
        differential_xsec = (
            torch.tensor(4.0 * np.pi, device=comp.device, dtype=comp.integration_dtype)
            * qT_t
            * xs
            / (2.0 * Q_t)
            / z_t
        )

        # Store result
        theo[i] = differential_xsec.to(theo.dtype)

    return theo


def get_trainable_vector_flavor_blind(model: torch.nn.Module) -> torch.Tensor:
    """Extract all trainable parameters from flavor-blind model."""
    vec = []
    for _, p in model.named_parameters():
        if p.requires_grad:
            vec.append(p.view(-1))
    if not vec:
        return torch.tensor([])
    return torch.cat(vec)


def randomize_fnp_parameters_flavor_blind(
    model: torch.nn.Module, scale: float = 0.2, seed: int = 0
):
    """
    Randomize flavor-blind fNP parameters.

    Since all flavors share parameters, we only need to randomize:
    - Evolution g2 parameter
    - Single PDF parameter set
    - Single FF parameter set
    """
    torch.manual_seed(seed)
    print(f"🎲 Randomizing flavor-blind fNP parameters with scale={scale}, seed={seed}")

    with torch.no_grad():
        randomized_count = 0

        # Randomize evolution parameter
        if hasattr(model, "evolution") and hasattr(model.evolution, "free_g2"):
            if model.evolution.free_g2.requires_grad:
                old_val = model.evolution.free_g2.clone()
                model.evolution.free_g2.add_(
                    torch.randn_like(model.evolution.free_g2) * scale
                )
                print(
                    f"  Evolution g2: {old_val.item():.6f} -> {model.evolution.free_g2.item():.6f}"
                )
                randomized_count += 1

        # Randomize shared PDF parameters
        if hasattr(model, "pdf_module") and hasattr(model.pdf_module, "free_params"):
            if model.pdf_module.free_params.requires_grad:
                old_val = model.pdf_module.free_params.clone()
                model.pdf_module.free_params.add_(
                    torch.randn_like(model.pdf_module.free_params) * scale
                )
                print(
                    f"  PDF (shared): params changed by {torch.norm(model.pdf_module.free_params - old_val).item():.6f}"
                )
                randomized_count += 1

        # Randomize shared FF parameters
        if hasattr(model, "ff_module") and hasattr(model.ff_module, "free_params"):
            if model.ff_module.free_params.requires_grad:
                old_val = model.ff_module.free_params.clone()
                model.ff_module.free_params.add_(
                    torch.randn_like(model.ff_module.free_params) * scale
                )
                print(
                    f"  FF (shared): params changed by {torch.norm(model.ff_module.free_params - old_val).item():.6f}"
                )
                randomized_count += 1

        print(f"  Total randomized parameter groups: {randomized_count}")
        if randomized_count == 0:
            print("  ⚠️  WARNING: No parameters were randomized! Check model structure.")


def main():
    """Main function to run flavor-blind fitting procedure."""
    # Check Python version requirement
    check_python_version()

    parser = argparse.ArgumentParser(
        description="Fit flavor-blind fNP parameters on synthetic cross sections"
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
        default=os.path.join(MAP_DIR, "inputs", "fNPconfig_flavor_blind.yaml"),
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

    print(f"\033[95m🔬 FLAVOR-BLIND fNP FITTING EXPERIMENT\033[0m")
    print(f"  Config: {args.config}")
    print(f"  Kinematics: {args.kinematics}")
    print(f"  fNP Config: {args.fnp_config}")
    print(f"  Points: {args.points}, Epochs: {args.epochs}, LR: {args.lr}")

    # Load kinematics
    kin = load_and_validate_kinematics(args.kinematics)

    # Initialize flavor-blind SIDIS computation
    try:
        comp = SIDISComputationFlavorBlind(
            args.config, args.fnp_config, device=args.device
        )
        if comp.model_fNP is None:
            raise RuntimeError("Flavor-blind fNP model failed to load")
        comp.model_fNP.train()

        # Generate synthetic targets
        print(f"\033[94m📊 Generating synthetic targets...\033[0m")
        with torch.no_grad():
            y_true = forward_cross_sections_torch_flavor_blind(
                comp, kin, max_points=args.points
            ).detach()

        print(f"  Generated {y_true.shape[0]} target cross sections")
        print(f"  Target range: [{y_true.min().item():.2e}, {y_true.max().item():.2e}]")

        # Initialize optimization
        true_vec = get_trainable_vector_flavor_blind(comp.model_fNP).detach().clone()
        print(f"  True parameter vector norm: {torch.norm(true_vec).item():.6f}")

        randomize_fnp_parameters_flavor_blind(comp.model_fNP, scale=0.3, seed=args.seed)

        opt = torch.optim.Adam(
            [p for p in comp.model_fNP.parameters() if p.requires_grad], lr=args.lr
        )

        print(f"  Optimizer: Adam with LR={args.lr}")
        print(f"  Total trainable parameters: {comp.model_fNP.count_parameters()}")

    except Exception as e:
        print(f"❌ ERROR: Failed to initialize flavor-blind computation: {e}")
        sys.exit(1)

    def chi_square(pred: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
        """Compute chi-squared loss with relative uncertainties."""
        sigma = torch.clamp(0.1 * truth.abs(), min=1e-3).to(pred.device)
        return torch.sum(((pred - truth) / sigma) ** 2)

    # Training loop
    print(f"\033[94m🏃 Starting flavor-blind fitting...\033[0m")
    for epoch in range(1, args.epochs + 1):
        opt.zero_grad(set_to_none=True)
        y_pred = forward_cross_sections_torch_flavor_blind(
            comp, kin, max_points=args.points
        )
        loss = chi_square(y_pred, y_true)
        loss.backward()
        opt.step()

        if epoch % max(1, args.epochs // 10) == 0:
            print(f"  Epoch {epoch:4d}/{args.epochs}  chi2 = {loss.item():.4e}")

    # Analyze results
    print(f"\033[92m✅ Flavor-blind fitting completed!\033[0m")

    fit_vec = get_trainable_vector_flavor_blind(comp.model_fNP).detach()
    delta = torch.norm(fit_vec - true_vec).item()
    rel = delta / (torch.norm(true_vec).item() + 1e-12)

    print(f"\n📈 RESULTS:")
    print(f"  Parameter L2 diff: {delta:.4e} (relative: {rel:.4e})")
    print(f"  Final chi2: {loss.item():.4e}")

    if rel < 1e-2:
        print(f"  🎉 SUCCESS: Parameters converged close to ground truth!")
    else:
        print(f"  ⚠️  WARNING: Parameters did not fully converge.")
        print(f"     Consider more epochs, LR tuning, or more kinematic points.")

    # Print parameter summary
    print(f"\n📋 PARAMETER SUMMARY:")
    comp.model_fNP.summary()


if __name__ == "__main__":
    main()
