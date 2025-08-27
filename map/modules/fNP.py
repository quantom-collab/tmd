"""
Main Interface Module for the Modular fNP System

WHAT THIS MODULE DOES:
======================
This module serves as the primary entry point and clean interface for the modular fNP system.
It re-exports all the key components from the underlying modules to provide a unified API.

WHY IS THIS NEEDED?
==================
1. **Clean Import Interface**: Users can simply import from 'fnp' rather than knowing the internals
   Example: `from modules.fnp import fNP` instead of `from modules.fnp_manager import fNPManager`

2. **API Abstraction**: Hides the internal organization from users
   - fnp_base.py: Core building blocks (evolution, PDF/FF base classes)
   - fnp_manager.py: Orchestration logic that combines everything
   - fnp.py: Clean public interface

3. **Backward Compatibility**: Provides stable import paths even if internal structure changes

4. **Package Organization**: Follows Python packaging best practices for complex modules

WHAT'S AVAILABLE:
================
- fNP: Main unified interface (alias to fNPManager) - this is what most users want
- fNPManager: Explicit manager class for advanced users
- Base classes: fNP_evolution, TMDPDFBase, TMDFFBase
- Default parameters: MAP22_DEFAULT_* constants

TYPICAL USAGE:
=============
```python
from modules.fnp import fNP
import yaml

with open('fNPconfig.yaml', 'r') as f:
    config = yaml.safe_load(f)

model = fNP(config)  # Creates the unified PDF+FF system

# Use for PDFs (fnp1)
pdf_results = model.forward_pdf(x, b, flavors=['u', 'd'])

# Use for FFs (fnp2)
ff_results = model.forward_ff(z, b, flavors=['u', 'd'])
```

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
