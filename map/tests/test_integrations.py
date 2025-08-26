#!/usr/bin/env python3
"""
Quick test script to verify both integration methods work.

This script can be run from any directory within the repository and will automatically
discover the correct paths for imports and file locations.
"""

import sys
import os
import yaml
import numpy as np
from typing import Tuple


def discover_repository_structure() -> Tuple[str, str, str]:
    """
    Auto-discover the repository structure and return the necessary paths.

    This function walks up the directory tree to find:
    1. The repository root (containing the 'map' directory)
    2. The map directory (containing the main modules)
    3. The tests directory (where this script is located)

    Returns:
        Tuple[str, str, str]: (repo_root, map_dir, tests_dir)
    """
    # Start from the directory containing this script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Walk up the directory tree to find the repository root
    current_dir = script_dir
    while current_dir != os.path.dirname(current_dir):  # Stop at filesystem root
        # Check if this directory contains a 'map' folder with 'modules' subfolder
        potential_map = os.path.join(current_dir, "map")
        if os.path.isdir(potential_map) and os.path.isdir(
            os.path.join(potential_map, "modules")
        ):
            repo_root = current_dir
            map_dir = potential_map
            tests_dir = script_dir
            return repo_root, map_dir, tests_dir

        # Move up one directory
        current_dir = os.path.dirname(current_dir)

    # If we can't find the structure, try to work with current directory
    # This handles the case where the script is run from within the map directory
    if os.path.isdir("modules"):
        # We're already in the map directory
        repo_root = os.path.dirname(os.getcwd())
        map_dir = os.getcwd()
        tests_dir = os.path.join(map_dir, "tests")
    else:
        raise RuntimeError(
            "Could not find repository structure. "
            "Please run this script from within the repository."
        )

    return repo_root, map_dir, tests_dir


def setup_import_paths(repo_root: str, map_dir: str) -> None:
    """
    Set up the Python import paths to allow importing the required modules.

    Args:
        repo_root: Path to the repository root
        map_dir: Path to the map directory
    """
    # Add repository root to sys.path for relative imports
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    # Add map directory to sys.path for direct imports
    if map_dir not in sys.path:
        sys.path.insert(0, map_dir)


def create_output_directory(tests_dir: str) -> str:
    """
    Create the tests/output directory if it doesn't exist.

    Args:
        tests_dir: Path to the tests directory

    Returns:
        str: Path to the output directory
    """
    output_dir = os.path.join(tests_dir, "output")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def test_integration_methods():
    """Test both PyTorch and Ogata integration methods."""

    # Discover repository structure
    try:
        repo_root, map_dir, tests_dir = discover_repository_structure()
        print(f"\033[94m📁 Repository root: {repo_root}\033[0m")
        print(f"\033[94m📁 Map directory: {map_dir}\033[0m")
        print(f"\033[94m📁 Tests directory: {tests_dir}\033[0m")
    except RuntimeError as e:
        print(f"\033[91m❌ {e}\033[0m")
        return

    # Set up import paths
    setup_import_paths(repo_root, map_dir)

    # Import the required module
    try:
        import sidis_crossect_torch

        print(f"\033[94m✅ Successfully imported sidis_crossect_torch\033[0m")
    except ImportError as e:
        print(f"\033[91m❌ Failed to import sidis_crossect_torch: {e}\033[0m")
        return

    # Create output directory
    output_dir = create_output_directory(tests_dir)
    print(f"\033[94m📁 Output directory: {output_dir}\033[0m")

    # Create a minimal kinematic data file
    test_data = {
        "header": {"Vs": 5.0, "target_isoscalarity": 1.0},
        "data": {"x": [0.1], "Q2": [4.0], "z": [0.5], "PhT": [0.5]},
    }

    # Write test data to output directory
    test_kinematics_file = os.path.join(output_dir, "test_kinematics.yaml")
    with open(test_kinematics_file, "w") as f:
        yaml.dump(test_data, f)

    print(
        "\033[94m🧪 Testing SIDIS computation with both integration methods...\033[0m"
    )

    # Initialize computation object
    config_file = os.path.join(map_dir, "inputs", "config.yaml")
    fnp_config_file = os.path.join(map_dir, "inputs", "fNPconfig.yaml")

    comp = sidis_crossect_torch.SIDISComputationPyTorch(config_file, fnp_config_file)

    print("\n\033[94m1️⃣ Testing PyTorch integration (differentiable)...\033[0m")
    pytorch_result_file = os.path.join(output_dir, "test_pytorch_result.yaml")
    try:
        comp.compute_sidis_cross_section_pytorch(
            test_kinematics_file,
            pytorch_result_file,
            use_ogata=False,
        )
        print("\033[92m✅ PyTorch integration successful!\033[0m")

        # Check if result file was created and contains non-zero results
        if os.path.exists(pytorch_result_file):
            with open(pytorch_result_file, "r") as f:
                result_data = yaml.safe_load(f)
                if result_data and "Kinematics" in result_data:
                    cross_sections = [
                        point.get("cross_section", 0.0)
                        for point in result_data["Kinematics"]
                    ]
                    if any(cs > 0.0 for cs in cross_sections):
                        print(
                            "\033[92m✅ PyTorch integration produced non-zero results\033[0m"
                        )
                    else:
                        print(
                            "\033[93m⚠️  PyTorch integration produced zero results\033[0m"
                        )
                else:
                    print(
                        "\033[93m⚠️  PyTorch integration result file is empty or malformed\033[0m"
                    )
        else:
            print("\033[93m⚠️  PyTorch integration result file was not created\033[0m")

    except Exception as e:
        print(f"\033[91m❌ PyTorch integration failed: {e}\033[0m")

    print("\n\033[94m2️⃣ Testing Ogata integration (high accuracy)...\033[0m")
    ogata_result_file = os.path.join(output_dir, "test_ogata_result.yaml")
    try:
        comp.compute_sidis_cross_section_pytorch(
            test_kinematics_file, ogata_result_file, use_ogata=True
        )
        print("\033[92m✅ Ogata integration completed!\033[0m")

        # Check if result file was created and contains non-zero results
        if os.path.exists(ogata_result_file):
            with open(ogata_result_file, "r") as f:
                result_data = yaml.safe_load(f)
                if result_data and "Kinematics" in result_data:
                    cross_sections = [
                        point.get("cross_section", 0.0)
                        for point in result_data["Kinematics"]
                    ]
                    if any(cs > 0.0 for cs in cross_sections):
                        print(
                            "\033[92m✅ Ogata integration produced non-zero results\033[0m"
                        )
                    else:
                        print(
                            "\033[91m❌ Ogata integration failed: produced zero results\033[0m"
                        )
                        print(
                            "\033[91m   This indicates the Ogata quadrature integration failed\033[0m"
                        )
                else:
                    print(
                        "\033[91m❌ Ogata integration failed: result file is empty or malformed\033[0m"
                    )
        else:
            print(
                "\033[91m❌ Ogata integration failed: result file was not created\033[0m"
            )

    except Exception as e:
        print(f"\033[91m❌ Ogata integration failed: {e}\033[0m")

    # Clean up test files (optional - comment out if you want to keep them for inspection)
    # cleanup_files = [
    #     test_kinematics_file,
    #     pytorch_result_file,
    #     ogata_result_file,
    # ]

    # print(f"\n\033[94m🧹 Cleaning up test files...\033[0m")
    # for f in cleanup_files:
    #     if os.path.exists(f):
    #         os.remove(f)
    #         print(f"\033[94m   Removed: {f}\033[0m")

    print("\n\033[92m🎉 Integration method testing completed!\033[0m")
    print(f"\033[94m   All outputs were saved to: {output_dir}\033[0m")


if __name__ == "__main__":
    test_integration_methods()
