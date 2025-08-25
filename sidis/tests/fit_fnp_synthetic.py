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


def check_python_version():
    """
    Check that we're running Python 3.10 as required by LHAPDF.
    If it's not running with python3.10, print an error message and exit.
    """
    if not (sys.version_info.major == 3 and sys.version_info.minor == 10):
        print(f"\033[91mERROR: This script MUST be run with Python 3.10")
        print(
            f"Current Python version: {sys.version_info.major}.{sys.version_info.minor}"
        )
        print(f"LHAPDF requires Python 3.10 for proper compatibility.")
        print(f"Please switch to Python 3.10 and try again.\033[0m")

        # Terminate the program with exit status 1
        sys.exit(1)


def _ensure_rootdir_on_syspath() -> Tuple[str, str]:
    """
    Auto-discover the repository root and ensure the 'map' modules are importable.

    This function walks up the directory tree from the current script location
    to find the TMD repository root (identified by the presence of a 'map'
    directory containing a 'modules' subdirectory). Once found, it adds both
    the repo root and the 'map' directory to sys.path for imports.

    Returns:
        Tuple[str, str]: (repo_root_path, map_directory_path)

    Algorithm:
        1. Start from the directory containing this script
        2. Walk up the directory tree looking for 'map/modules/'
        3. When found, add both repo root and map dir to sys.path
        4. If not found, fall back to assuming standard layout (../../map)
    """
    # Start from the directory containing this script file
    start = os.path.abspath(os.path.dirname(__file__))
    cur = start

    # Walk up the directory tree to find the repo root.
    # Starts an infinite loop: the condition is the literal boolean
    # True, so it’s always “true.” The loop only stops when code
    # inside it returns or breaks.
    while True:
        # Check if current directory contains a 'map' folder with 'modules' subfolder
        candidate_map = os.path.join(cur, "map")
        if os.path.isdir(candidate_map) and os.path.isdir(
            os.path.join(candidate_map, "modules")
        ):
            # Found the repo root! Set up import paths
            repo_root = cur

            # Add repo root to sys.path if not already present
            # This allows imports like 'from map.modules import ...'
            if repo_root not in sys.path:
                # Puts repo_root at index 0 (highest priority),
                # ahead of the script dir, site-packages, etc.
                sys.path.insert(0, repo_root)

            # Add map directory to sys.path if not already present
            # This allows imports like 'from sidis_crossect_torch import ...'
            if candidate_map not in sys.path:
                sys.path.insert(0, candidate_map)

            return repo_root, candidate_map

        # Move up one directory level
        parent = os.path.dirname(cur)

        # Check if we've reached the filesystem root (can't go higher)
        if parent == cur:
            break
        cur = parent

    # Fallback: assume script is in repo_root/map/tests/ (standard layout)
    # Go up two levels: tests/ -> map/ -> repo_root/
    fallback_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

    # Add fallback paths to sys.path
    if fallback_root not in sys.path:
        sys.path.insert(0, fallback_root)

    return fallback_root, os.path.join(fallback_root, "map")


# Initialize repository paths and set up import system
# This must be done before any local imports (like 'from map.sidis_crossect_torch import ...')
REPO_ROOT, MAP_DIR = _ensure_rootdir_on_syspath()


def load_kinematics(path: str) -> Dict[str, Any]:
    """
    Load kinematic data from a YAML file and validate its structure.
    This function performs comprehensive validation to ensure the YAML file
    contains the expected structure for SIDIS kinematic data.

    Args:
        path (str): Absolute or relative path to the YAML kinematics file.
                   Expected format: {"header": {...}, "data": {"x": [...], "Q2": [...], ...}}

    Returns:
        Dict[str, Any]: Parsed YAML data containing:
                       - "header": metadata (Vs, target_isoscalarity, etc.)
                       - "data": kinematic arrays (x, Q2, z, PhT as Python lists)

    Raises:
        FileNotFoundError: If the YAML file doesn't exist
        yaml.YAMLError: If the file contains invalid YAML syntax
        ValueError: If the loaded data has wrong structure, including:
                   - Not a dictionary at top level (e.g., list, string, number)
                   - Missing required keys ("header" or "data")
                   - Missing required kinematic variables (x, Q2, z, PhT)
                   - Kinematic variables are not lists/arrays
    """
    with open(path, "r") as f:
        data = yaml.safe_load(f)

    # Check 1: Must be a dictionary at the top level
    # YAML can return lists, strings, numbers, None - we need a dict
    if not isinstance(data, dict):
        raise ValueError(
            f"\033[33m[load_kinematics] Invalid kinematics YAML format: top level must be a dictionary, "
            f"got {type(data).__name__}. "
            f"Expected: {{header: {{...}}, data: {{...}}}}\033[0m"
        )

    # Check 2: Must have required top-level keys
    required_keys = ["header", "data"]
    for key in required_keys:
        if key not in data:
            raise ValueError(
                f"\033[33m[load_kinematics] Invalid kinematics YAML format: missing required key '{key}'. "
                f"Expected structure: {{header: {{...}}, data: {{x: [...], Q2: [...], z: [...], PhT: [...]}}}}\033[0m"
            )

    # Check 3: Data section must contain required kinematic variables
    required_kinematics = ["x", "Q2", "z", "PhT"]
    data_section = data["data"]
    if not isinstance(data_section, dict):
        raise ValueError(
            f"\033[33m[load_kinematics] Invalid kinematics YAML format: 'data' must be a dictionary, "
            f"got {type(data_section).__name__}\033[0m"
        )

    for var in required_kinematics:
        if var not in data_section:
            raise ValueError(
                f"\033[33m[load_kinematics] Invalid kinematics YAML format: missing kinematic variable '{var}' in data section. "
                f"Required variables: {required_kinematics}\033[0m"
            )

        # Check that each kinematic variable is a list/array (not a single value)
        if not isinstance(data_section[var], (list, tuple)):
            raise ValueError(
                f"\033[33m[load_kinematics] Invalid kinematics YAML format: kinematic variable '{var}' must be a list/array, "
                f"got {type(data_section[var]).__name__}. "
                f"Example: {var}: [0.1, 0.2, 0.3]\033[0m"
            )

    return data


def tensors_from_kinematics(
    comp, data: Dict[str, Any]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Convert kinematic data from Python lists to PyTorch tensors for computation.

    This function extracts the kinematic variables (x, Q2, z, PhT) from the loaded
    YAML data and converts them to PyTorch tensors with the correct dtype and device
    configuration for the SIDIS computation.

    Args:
        comp: SIDISComputationPyTorch instance containing:
              - comp.dtype: Target tensor dtype (usually torch.float32)
              - comp.device: Computation device ('cpu', 'cuda', or 'mps')
              This object is created later in main() and contains all TMD computation setup.

        data (Dict[str, Any]): Kinematic data dictionary from load_kinematics().
                              A Dict (dictionary) is Python's key-value mapping type: {"key": value}.
                              Expected structure:
                              {
                                "header": {"Vs": float, "target_isoscalarity": float, ...},
                                "data": {
                                  "x": [float, float, ...],      # Bjorken x values
                                  "Q2": [float, float, ...],     # Virtuality squared (GeV²)
                                  "z": [float, float, ...],      # Fragmentation fraction
                                  "PhT": [float, float, ...]     # Transverse momentum (GeV)
                                }
                              }

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            Four tensors (x, Q2, z, PhT) configured with:
            - Same dtype as comp.dtype (for memory efficiency)
            - Same device as comp.device (for GPU acceleration)
            - Shape: [N] where N is the number of kinematic points
    """
    # Create empty object which will hold the kinematic tensors
    kd = {}

    # Print message
    print("\033[92m---Converting x, Q2, z, PhT to tensors...\033[0m")

    # For each kinematic variable, convert the Python list to a PyTorch tensor
    for key in ["x", "Q2", "z", "PhT"]:
        kd[key] = torch.tensor(data["data"][key], dtype=comp.dtype, device=comp.device)
    return kd["x"], kd["Q2"], kd["z"], kd["PhT"]


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

        kin (Dict[str, Any]): Kinematic data from load_kinematics() containing:
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

    # Convert kinematic data to tensors
    # Extract x, Q2, z, PhT from YAML data and convert to PyTorch tensors
    x, Q2, z, PhT = tensors_from_kinematics(comp, kin)

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
        # TODO: figure out how to put all flavors
        pdf_flavor = "u"  # PDF flavor (up quark for simplicity)
        ff_flavor = "u"  # FF flavor (up quark for simplicity)

        # Evaluate fNP for PDF: fNP(x, b) for the PDF sector
        # x_t.expand_as(b) creates tensor [x, x, x, ...] matching b-grid length
        fnp_pdf = comp.compute_fnp_pytorch(
            x_t.expand_as(b), b.to(comp.dtype), pdf_flavor
        ).to(comp.integration_dtype)

        # Evaluate fNP for FF: fNP(z, b) for the fragmentation function sector
        fnp_ff = comp.compute_fnp_pytorch(
            z_t.expand_as(b), b.to(comp.dtype), ff_flavor
        ).to(comp.integration_dtype)

        # ===== Compute Fourier-Bessel transform kernel =====
        # Zeroth-order Bessel function
        J0 = comp._bessel_j0_torch(qT_t * b)

        # Build the full integrand
        integrand = b * J0 * fnp_pdf * fnp_ff * L_b

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
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=5e-2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    kin = load_kinematics(args.kinematics)

    # Initialize SIDIS computation with proper error handling
    try:
        from map.sidis_crossect_torch import SIDISComputationPyTorch

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
