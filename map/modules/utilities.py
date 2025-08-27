"""
utilities.py
author: Chiara Bissolotti (cbissolotti@anl.gov)

This file contains utility functions that are used in the map module
"""

import sys
import yaml
import torch
from typing import Dict, Tuple, Any, Optional
import os
import matplotlib.pyplot as plt
from IPython.display import display, Latex


###############################################################################
# Repository path setup for local imports
###############################################################################
def ensure_repo_on_syspath(start_dir: Optional[str] = None) -> Tuple[str, str]:
    """
    Ensure the repository root and 'map' directory are present in sys.path.

    Walks up from start_dir (or current file/cwd) to locate a directory that
    contains a 'map' folder with a 'modules' subfolder. When found, it inserts
    both the repo root and the 'map' directory at the front of sys.path.

    Args:
        start_dir: Optional starting directory. If None, uses the directory
                   of the caller's file if available, otherwise the cwd.

    Returns:
        (repo_root_path, map_directory_path)

    Raises:
        RuntimeError if the repository structure cannot be located.
    """
    # Determine a reasonable starting directory
    if start_dir is None:
        # Try to infer the caller's directory from the call stack; fallback to cwd
        try:
            # __file__ is not always defined (e.g., in some REPL contexts)
            start_dir = os.path.abspath(os.path.dirname(__file__))
        except Exception:
            start_dir = os.getcwd()

    current = os.path.abspath(start_dir)

    # Walk up the directory tree to find repo root containing map/modules
    while True:
        candidate_map = os.path.join(current, "map")
        if os.path.isdir(candidate_map) and os.path.isdir(
            os.path.join(candidate_map, "modules")
        ):
            # Found repository root
            repo_root = current
            # Prepend to sys.path to prioritize local code
            if repo_root not in sys.path:
                sys.path.insert(0, repo_root)
            if candidate_map not in sys.path:
                sys.path.insert(0, candidate_map)
            return repo_root, candidate_map

        parent = os.path.dirname(current)
        if parent == current:
            break
        current = parent

    # If not found, raise a clear error
    raise RuntimeError(
        "Could not locate repository root containing 'map/modules'. "
        "Start the search from a path inside the repository or set PYTHONPATH to the repo root."
    )


###############################################################################
# Check Python Version
###############################################################################
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


###############################################################################
# YAML Configuration Loader
###############################################################################
def load_yaml_file(yaml_file_path: str) -> dict:
    """
    Load a YAML configuration file. This function load any yaml file, e.g.,
    config.yaml and fNPconfig.yaml and kinematics.yaml. It does not validate
    the structure of the file, it just loads it.
    """

    with open(yaml_file_path, "r") as f:
        config = yaml.safe_load(f)
    return config


###############################################################################
# Load and validate Kinematics
###############################################################################
def load_and_validate_kinematics(path: str) -> Dict[str, Any]:
    """
    Load kinematic data from a YAML file and validate its structure. This function
    is meant to be used to load the kinematics.yaml file.

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
    data = load_yaml_file(path)

    # Check 1: Must be a dictionary at the top level
    # YAML can return lists, strings, numbers, None - we need a dict
    if not isinstance(data, dict):
        raise ValueError(
            f"\033[33m[load_and_validate_kinematics] Invalid kinematics YAML format: top level must be a dictionary, "
            f"got {type(data).__name__}. "
            f"Expected: {header: {...}, data: {...}}\033[0m"
        )

    # Check 2: Must have required top-level keys
    required_keys = ["header", "data"]
    for key in required_keys:
        if key not in data:
            raise ValueError(
                f"\033[33m[load_and_validate_kinematics] Invalid kinematics YAML format: missing required key '{key}'. "
                f"Expected structure: {header: {...}, data: {x: [...], Q2: [...], z: [...], PhT: [...]}}\033[0m"
            )

    # Check 3: Data section must contain required kinematic variables
    required_kinematics = ["x", "Q2", "z", "PhT"]
    data_section = data["data"]
    if not isinstance(data_section, dict):
        raise ValueError(
            f"\033[33m[load_and_validate_kinematics] Invalid kinematics YAML format: 'data' must be a dictionary, "
            f"got {type(data_section).__name__}\033[0m"
        )

    for var in required_kinematics:
        if var not in data_section:
            raise ValueError(
                f"\033[33m[load_and_validate_kinematics] Invalid kinematics YAML format: missing kinematic variable '{var}' in data section. "
                f"Required variables: {required_kinematics}\033[0m"
            )

        # Check that each kinematic variable is a list/array (not a single value)
        if not isinstance(data_section[var], (list, tuple)):
            raise ValueError(
                f"\033[33m[load_and_validate_kinematics] Invalid kinematics YAML format: kinematic variable '{var}' must be a list/array, "
                f"got {type(data_section[var]).__name__}. "
                f"Example: {var}: [0.1, 0.2, 0.3]\033[0m"
            )

    return data


###############################################################################
# Plot fNP
###############################################################################
def plot_fNP(model_fNP, x, flavors=None, plot_type="pdf"):
    """
    Evaluate and plot the fNP values for a given fNP model over a range of b values.

    Parameters:
      - model_fNP (nn.Module): An instance of your fNP module (modular system).
      - x (torch.Tensor): The input tensor for x (typically a scalar tensor).
      - flavors (list, optional): A list of flavor keys to evaluate.
                                  If None, uses model_fNP.pdf_flavor_keys.
      - plot_type (str): Type of fNP to plot - 'pdf' (fnp1), 'ff' (fnp2), or 'both'.

    The function evaluates the model for each b in b_range and collects the fNP
    output for each flavor. It then produces a plot of fNP versus b for all flavors.

    Note: fnp1 = PDF, fnp2 = FF as specified for cross section computation.
    """
    # If no specific flavors are provided, use all available flavors.
    if flavors is None:
        if hasattr(model_fNP, "pdf_flavor_keys"):
            flavors = model_fNP.pdf_flavor_keys
        else:
            # Fallback for backward compatibility
            flavors = ["u", "d", "s"]

    # Create a range of b values from 0 to 10.
    b_values = torch.linspace(0, 10, steps=100)

    if plot_type == "both":
        # Create dictionaries for both PDF and FF results
        pdf_results = {flavor: [] for flavor in flavors}
        ff_results = {flavor: [] for flavor in flavors}

        # Loop over the b values, evaluating both PDF and FF
        for b in b_values:
            # Evaluate PDF fNP (fnp1)
            pdf_outputs = model_fNP.forward_pdf(x, b, flavors=flavors)
            # Evaluate FF fNP (fnp2) - use x as z for fragmentation
            ff_outputs = model_fNP.forward_ff(x, b, flavors=flavors)

            for flavor in flavors:
                pdf_results[flavor].append(pdf_outputs[flavor].item())
                ff_results[flavor].append(ff_outputs[flavor].item())

        # Plot both PDF and FF on the same figure
        plt.figure(figsize=(12, 8))
        for flavor in flavors:
            plt.plot(
                b_values.numpy(),
                pdf_results[flavor],
                label=f"{flavor} PDF (fnp1)",
                linestyle="-",
            )
            plt.plot(
                b_values.numpy(),
                ff_results[flavor],
                label=f"{flavor} FF (fnp2)",
                linestyle="--",
            )

        plt.xlabel("b")
        plt.ylabel("fNP")
        plt.title("fNP vs. b: PDF (fnp1) and FF (fnp2) for all flavors")
        plt.legend()
        plt.show()

    else:
        # Create a dictionary to store the computed fNP for each flavor.
        results_dict = {flavor: [] for flavor in flavors}

        # Loop over the b values, evaluating the model at each b.
        for b in b_values:
            if plot_type == "pdf":
                # Evaluate PDF fNP (fnp1)
                outputs = model_fNP.forward_pdf(x, b, flavors=flavors)
            elif plot_type == "ff":
                # Evaluate FF fNP (fnp2) - use x as z for fragmentation
                outputs = model_fNP.forward_ff(x, b, flavors=flavors)
            else:
                raise ValueError(
                    f"Invalid plot_type: {plot_type}. Use 'pdf', 'ff', or 'both'."
                )

            for flavor in flavors:
                # We assume that each output is a scalar tensor.
                results_dict[flavor].append(outputs[flavor].item())

        # Plot all flavors on the same figure.
        plt.figure(figsize=(10, 6))
        for flavor, values in results_dict.items():
            plt.plot(b_values.numpy(), values, label=flavor)
        plt.xlabel("b")

        if plot_type == "pdf":
            plt.ylabel("fNP PDF (fnp1)")
            plt.title("fNP PDF (fnp1) vs. b for all flavors")
        else:
            plt.ylabel("fNP FF (fnp2)")
            plt.title("fNP FF (fnp2) vs. b for all flavors")

        plt.legend()
        plt.show()


###############################################################################
# Show Latex Formulas in Jupyter Notebooks
###############################################################################
def show_latex_formula(latex_formula: str):
    """
    Automatically render the LaTeX formula in a Jupyter notebook.
    """
    display(Latex(latex_formula))
