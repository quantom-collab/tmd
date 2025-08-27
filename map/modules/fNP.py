"""
New modular fNP system for TMD PDFs and FFs.

This module provides the main interface for the reorganized fNP system:
- Unified management of TMD PDFs and FFs
- MAP22 parameterization from NangaParbat
- Shared evolution factor
- Simultaneous optimization framework

Author: Chiara Bissolotti (cbissolotti@anl.gov)
"""

# Re-export the main classes and functions from the modular system
from .fnp_base import (
    fNP_evolution,
    TMDPDFBase,
    TMDFFBase,
    MAP22_DEFAULT_EVOLUTION,
    MAP22_DEFAULT_PDF_PARAMS,
    MAP22_DEFAULT_FF_PARAMS,
)

from .fnp_manager import fNPManager, fNP

# Make the main interface easily accessible
__all__ = [
    "fNP",  # Main unified interface (alias to fNPManager)
    "fNPManager",  # Explicit manager class
    "fNP_evolution",  # Evolution factor module
    "TMDPDFBase",  # TMD PDF base class
    "TMDFFBase",  # TMD FF base class
    "MAP22_DEFAULT_EVOLUTION",  # Default evolution parameters
    "MAP22_DEFAULT_PDF_PARAMS",  # Default PDF parameters
    "MAP22_DEFAULT_FF_PARAMS",  # Default FF parameters
]

# Version info
__version__ = "2.0.0"
__author__ = "Chiara Bissolotti"
__email__ = "cbissolotti@anl.gov"
