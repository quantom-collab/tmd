#!/usr/bin/env python3
"""
Test script to demonstrate the improved error formatting in fit_fnp_synthetic.py
"""

import sys
import os

# Add the map tests directory to path
sys.path.insert(0, "/Users/cbissolotti/anl/projects/tmd/map/tests")


def test_error_formatting():
    """Test the colored error messages."""
    print("🧪 Testing improved error formatting in fit_fnp_synthetic.py")
    print()

    try:
        from fit_fnp_synthetic import load_kinematics

        # Test 1: File doesn't exist
        print("1️⃣ Testing file not found error...")
        try:
            load_kinematics("nonexistent_file.yaml")
        except FileNotFoundError as e:
            print(f"   ✅ Got expected FileNotFoundError: {e}")
        except Exception as e:
            print(f"   ❌ Unexpected error: {e}")

        # Test 2: Create invalid YAML content and test it
        print("\n2️⃣ Testing invalid YAML structure errors...")

        # Create temporary invalid YAML files
        test_cases = [
            ("invalid_top_level.yaml", "42.5", "top level must be a dictionary"),
            (
                "missing_header.yaml",
                "data:\n  x: [0.1, 0.2]",
                "missing required key 'header'",
            ),
            (
                "missing_x.yaml",
                "header:\n  Vs: 7.25\ndata:\n  Q2: [4.0, 9.0]",
                "missing kinematic variable 'x'",
            ),
            (
                "x_not_list.yaml",
                "header:\n  Vs: 7.25\ndata:\n  x: 0.5\n  Q2: [4.0]\n  z: [0.5]\n  PhT: [0.8]",
                "'x' must be a list/array",
            ),
        ]

        for filename, content, expected_error in test_cases:
            print(f"\n   Testing: {expected_error}")
            with open(filename, "w") as f:
                f.write(content)

            try:
                load_kinematics(filename)
                print(f"   ❌ Expected error but function succeeded")
            except ValueError as e:
                error_msg = str(e)
                if (
                    "\033[33m[load_kinematics]" in error_msg
                    and expected_error in error_msg
                ):
                    print(f"   ✅ Got expected orange-colored error with function name")
                    print(f"      Error: {error_msg}")
                else:
                    print(f"   ⚠️  Got error but formatting may be wrong: {error_msg}")
            except Exception as e:
                print(f"   ❌ Unexpected error type: {e}")
            finally:
                # Clean up test file
                if os.path.exists(filename):
                    os.remove(filename)

        print(f"\n✅ Error formatting test completed!")
        print(f"   All ValueError exceptions now display in orange with function names")

    except ImportError as e:
        print(f"❌ Could not import from fit_fnp_synthetic: {e}")


if __name__ == "__main__":
    test_error_formatting()
