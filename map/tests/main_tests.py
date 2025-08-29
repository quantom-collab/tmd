#!/usr/bin/env python3
"""
Main Test Runner for fNP System

This script runs all Python test files in the map/tests/ directory.
It provides colored output to clearly show which tests are running
and their results.

Author: Chiara Bissolotti (cbissolotti@anl.gov)
"""

import os
import sys
import subprocess
import importlib.util
import argparse
from pathlib import Path
from typing import List, Tuple, Dict
import time
from datetime import datetime


def get_test_files() -> List[str]:
    """
    Get all Python test files in the current directory.

    Returns:
        List[str]: List of Python test file names (excluding main_tests.py)
    """
    current_dir = Path(__file__).parent
    test_files = []

    for file_path in current_dir.glob("*.py"):
        if file_path.name != "main_tests.py" and not file_path.name.startswith("_"):
            test_files.append(file_path.name)

    return sorted(test_files)


def run_test_verbose(test_file: str) -> Tuple[bool, str]:
    """
    Run a test file with real-time output visible.

    Args:
        test_file (str): Name of the test file to run

    Returns:
        Tuple[bool, str]: (success, summary_message)
    """
    try:
        print_colored(f"▶️  Starting {test_file}...", "yellow")
        print_colored("-" * 60, "yellow")

        # Run as subprocess without capturing output - shows in real-time
        result = subprocess.run(
            [sys.executable, test_file],
            cwd=Path(__file__).parent,
            timeout=300,
        )

        print_colored("-" * 60, "yellow")

        if result.returncode == 0:
            return True, f"✅ {test_file} completed successfully"
        else:
            return False, f"❌ {test_file} failed with return code {result.returncode}"

    except subprocess.TimeoutExpired:
        return False, f"❌ {test_file} timed out after 5 minutes"
    except Exception as e:
        return False, f"❌ {test_file} failed with error: {str(e)}"


def run_test_as_module(test_file: str) -> Tuple[bool, str]:
    """
    Run a test file as a Python module.

    Args:
        test_file (str): Name of the test file to run

    Returns:
        Tuple[bool, str]: (success, output/error_message)
    """
    try:
        # Add the parent directory to sys.path to ensure imports work
        test_dir = Path(__file__).parent
        parent_dir = test_dir.parent

        if str(parent_dir) not in sys.path:
            sys.path.insert(0, str(parent_dir))

        # Import and run the test module
        spec = importlib.util.spec_from_file_location(
            test_file.replace(".py", ""), test_dir / test_file
        )
        module = importlib.util.module_from_spec(spec)

        # Capture stdout/stderr
        import io
        from contextlib import redirect_stdout, redirect_stderr

        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()

        with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
            spec.loader.exec_module(module)

        output = stdout_capture.getvalue()
        error_output = stderr_capture.getvalue()

        if error_output:
            return False, f"STDERR:\n{error_output}\nSTDOUT:\n{output}"
        else:
            return True, output

    except Exception as e:
        return False, f"Import/execution error: {str(e)}"


def run_test_as_subprocess(test_file: str) -> Tuple[bool, str]:
    """
    Run a test file as a subprocess (fallback method).

    Args:
        test_file (str): Name of the test file to run

    Returns:
        Tuple[bool, str]: (success, output/error_message)
    """
    try:
        # Run the test file as a subprocess
        result = subprocess.run(
            [sys.executable, test_file],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent,
            timeout=300,  # 5 minute timeout
        )

        if result.returncode == 0:
            return True, result.stdout
        else:
            error_msg = f"Return code: {result.returncode}\n"
            if result.stderr:
                error_msg += f"STDERR:\n{result.stderr}\n"
            if result.stdout:
                error_msg += f"STDOUT:\n{result.stdout}"
            return False, error_msg

    except subprocess.TimeoutExpired:
        return False, "Test timed out after 5 minutes"
    except Exception as e:
        return False, f"Subprocess error: {str(e)}"


def run_test(test_file: str) -> Tuple[bool, str]:
    """
    Run a test file using the best available method.

    Args:
        test_file (str): Name of the test file to run

    Returns:
        Tuple[bool, str]: (success, output/error_message)
    """
    # Try module import first, fallback to subprocess
    try:
        return run_test_as_module(test_file)
    except Exception:
        return run_test_as_subprocess(test_file)


def print_colored(text: str, color: str = "white"):
    """
    Print colored text to terminal.

    Args:
        text (str): Text to print
        color (str): Color name (blue, red, green, yellow, white)
    """
    colors = {
        "blue": "\033[94m",
        "red": "\033[91m",
        "green": "\033[92m",
        "yellow": "\033[93m",
        "white": "\033[0m",
    }

    color_code = colors.get(color, colors["white"])
    reset_code = colors["white"]

    print(f"{color_code}{text}{reset_code}")


def main():
    """
    Main test runner function.
    """
    # Add argument parsing
    parser = argparse.ArgumentParser(description="Run fNP system tests")
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Show detailed output from each test",
    )
    args = parser.parse_args()

    print_colored("🧪 fNP SYSTEM TEST SUITE", "blue")
    if args.verbose:
        print_colored("📋 VERBOSE MODE: Detailed test output enabled", "yellow")
    print_colored("=" * 50, "blue")
    print()

    # Print the current date and time
    print_colored(
        f"Current date and time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "white",
    )

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
    from modules.utilities import check_python_version

    # Check Python version
    check_python_version()

    # Get all test files
    test_files = get_test_files()

    if not test_files:
        print_colored("❌ No test files found in current directory", "red")
        return

    print_colored(f"Found {len(test_files)} test files:", "white")
    for test_file in test_files:
        print(f"  • {test_file}")
    print()

    # Run tests
    results: Dict[str, Tuple[bool, str]] = {}
    start_time = time.time()

    for i, test_file in enumerate(test_files, 1):
        print_colored(f"[{i}/{len(test_files)}] Running: {test_file}", "blue")
        if not args.verbose:
            print_colored("-" * 60, "blue")

        test_start_time = time.time()

        if args.verbose:
            # Show output in real-time
            success, output = run_test_verbose(test_file)
        else:
            # Capture output (current behavior)
            success, output = run_test(test_file)

        test_duration = time.time() - test_start_time

        results[test_file] = (success, output)

        if success:
            print_colored(f"✅ {test_file} PASSED ({test_duration:.2f}s)", "green")
        else:
            print_colored(f"❌ {test_file} FAILED ({test_duration:.2f}s)", "red")
            if not args.verbose:  # Only show error details in non-verbose mode
                print_colored("Error details:", "red")
                print_colored(output, "red")

        print()

    # Summary
    total_duration = time.time() - start_time
    passed = sum(1 for success, _ in results.values() if success)
    failed = len(results) - passed

    print_colored("📊 TEST SUMMARY", "blue")
    print_colored("=" * 50, "blue")
    print_colored(f"Total tests: {len(test_files)}", "white")
    print_colored(f"Passed: {passed}", "green")
    print_colored(f"Failed: {failed}", "white")
    print_colored(f"Total time: {total_duration:.2f}s", "white")
    print()

    # Detailed results
    if failed > 0:
        print_colored("❌ FAILED TESTS:", "red")
        for test_file, (success, output) in results.items():
            if not success:
                print_colored(f"  • {test_file}", "red")
        print()

    if passed > 0:
        print_colored("✅ PASSED TESTS:", "green")
        for test_file, (success, _) in results.items():
            if success:
                print_colored(f"  • {test_file}", "green")
        print()

    # Final status
    if failed == 0:
        print_colored("🎉 ALL TESTS PASSED!", "green")
        return 0
    else:
        print_colored(f"⚠️  {failed} TEST(S) FAILED", "red")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
