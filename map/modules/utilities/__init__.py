"""
Utilities package for SIDIS framework.

This package contains utility modules for the SIDIS TMD cross-section computation.
"""

from .colors import tcolors
from .utilities import (
    ensure_repo_on_syspath,
    check_python_version,
    load_yaml_file,
    load_and_validate_kinematics,
    plot_fNP,
    show_latex_formula,
)

__all__ = [
    "tcolors",
    "ensure_repo_on_syspath",
    "check_python_version",
    "load_yaml_file",
    "load_and_validate_kinematics",
    "plot_fNP",
    "show_latex_formula",
]
