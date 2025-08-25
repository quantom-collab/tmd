#!/usr/bin/env python3
"""
Quick test script to verify both integration methods work.
"""

import sys
import os
import yaml

# Add map to path
sys.path.insert(0, "map")
import sidis_crossect_torch


def test_integration_methods():
    """Test both PyTorch and Ogata integration methods."""

    # Create a minimal kinematic data file
    test_data = {
        "header": {"Vs": 5.0, "target_isoscalarity": 1.0},
        "data": {"x": [0.1], "Q2": [4.0], "z": [0.5], "PhT": [0.5]},
    }

    # Write test data
    with open("test_kinematics.yaml", "w") as f:
        yaml.dump(test_data, f)

    print("🧪 Testing SIDIS computation with both integration methods...")

    # Initialize computation object
    comp = sidis_crossect_torch.SIDISComputationPyTorch(
        "map/inputs/config.yaml", "map/inputs/fNPconfig.yaml"
    )

    print("\n1️⃣ Testing PyTorch integration (differentiable)...")
    try:
        comp.compute_sidis_cross_section_pytorch(
            "test_kinematics.yaml", "test_pytorch_result.yaml", use_ogata=False
        )
        print("✅ PyTorch integration successful!")
    except Exception as e:
        print(f"❌ PyTorch integration failed: {e}")

    print("\n2️⃣ Testing Ogata integration (high accuracy)...")
    try:
        comp.compute_sidis_cross_section_pytorch(
            "test_kinematics.yaml", "test_ogata_result.yaml", use_ogata=True
        )
        print("✅ Ogata integration successful!")
    except Exception as e:
        print(f"❌ Ogata integration failed: {e}")

    # Clean up
    for f in [
        "test_kinematics.yaml",
        "test_pytorch_result.yaml",
        "test_ogata_result.yaml",
    ]:
        if os.path.exists(f):
            os.remove(f)

    print("\n🎉 Integration method testing completed!")


if __name__ == "__main__":
    test_integration_methods()
