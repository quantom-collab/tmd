#!/usr/bin/env python3
"""
Test script for SIDIS cross section computation

This script runs the SIDIS computation with the mock data and configuration files.
"""

import os
import sys
from sidis_computation import SIDISComputation


def main():
    """Run SIDIS computation test"""

    # Check Python version
    print(f"Python version: {sys.version}")

    # File paths
    config_file = "input/config.yaml"
    data_file = "input/kinematics.yaml"
    output_file = "sidis_results.yaml"

    # Check if files exist
    if not os.path.exists(config_file):
        print(f"Error: Configuration file {config_file} not found!")
        return 1

    if not os.path.exists(data_file):
        print(f"Error: Data file {data_file} not found!")
        return 1

    try:
        print("=" * 60)
        print("SIDIS Cross Section Computation Test")
        print("=" * 60)

        # Initialize computation
        print("Initializing SIDIS computation...")
        sidis_comp = SIDISComputation(config_file)

        # Run computation
        print("Running SIDIS cross section computation...")
        sidis_comp.compute_sidis_cross_section(data_file, output_file)

        print("=" * 60)
        print("Test completed successfully!")
        print(f"Results saved to: {output_file}")
        print("=" * 60)

        return 0

    except Exception as e:
        print(f"Error during computation: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
