#!/usr/bin/env python3
"""
Quick test script to verify both integration methods work.
"""

import sys
import os
import yaml
import numpy as np

# Add sidis to path
sys.path.insert(0, "sidis")
import sidis_crossect_torch


def create_output_directory():
    """Create the sidis/tests/output directory if it doesn't exist."""
    output_dir = "sidis/tests/output"
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def test_integration_methods():
    """Test both PyTorch and Ogata integration methods."""

    # Create output directory
    output_dir = create_output_directory()

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
    comp = sidis_crossect_torch.SIDISComputationPyTorch(
        "sidis/inputs/config.yaml", "sidis/inputs/fNPconfig.yaml"
    )

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
