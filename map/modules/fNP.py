"""
Unified Public Interface for fNP System (Standard and Flavor-Blind)

WHAT THIS MODULE DOES:
======================
This module serves as the primary entry point and clean interface for the complete fNP system,
including both standard flavor-dependent and flavor-blind implementations. It re-exports all
key components from the underlying modules to provide a unified, stable API.

WHY IS THIS NEEDED?
==================
1. **Clean Import Interface**: Users can import from 'fnp' without knowing internal structure
   Example: `from modules.fnp import fNPManager, fNPManagerFlavorBlind`

2. **API Abstraction**: Hides internal organization from users
   - fnp_base.py: Standard core building blocks (evolution, PDF/FF base classes)
   - fnp_base_flavor_blind.py: Flavor-blind core building blocks
   - fnp_manager.py: Standard orchestration logic
   - fnp_manager_flavor_blind.py: Flavor-blind orchestration logic
   - fnp.py: Clean unified public interface

3. **Backward Compatibility**: Provides stable import paths even if internal structure changes

WHAT'S AVAILABLE:
================
Standard (Flavor-Dependent) Classes:
- fNPManager: Main unified interface for standard flavor-dependent system
- fNP_evolution, TMDPDFBase, TMDFFBase: Standard base classes
- MAP22_DEFAULT_* constants: Standard default parameters

Flavor-Blind Classes:
- fNPManagerFlavorBlind: Main unified interface for flavor-blind system
- TMDPDFFlavorBlind, TMDFFFlavorBlind: Flavor-blind base classes
- load_flavor_blind_config: Configuration loader for flavor-blind system
- MAP22_DEFAULT_*_FLAVOR_BLIND constants: Flavor-blind default parameters

TYPICAL USAGE:
=============
```python
# Standard flavor-dependent system
from modules.fnp import fNPManager
config = yaml.safe_load(open('fNPconfig.yaml'))
model = fNPManager(config)

# Flavor-blind system (all flavors share parameters)
from modules.fnp import fNPManagerFlavorBlind, load_flavor_blind_config
config = load_flavor_blind_config('fNPconfig_flavor_blind.yaml')
model_blind = fNPManagerFlavorBlind(config)

# Both support the same interface
pdf_results = model.forward_pdf(x, b, flavors=['u', 'd'])
ff_results = model.forward_ff(z, b, flavors=['u', 'd'])
```

Author: Chiara Bissolotti (cbissolotti@anl.gov)
"""

# Re-export the main classes and functions from the Base fNP classes
from .fnp_base import (
    fNP_evolution,
    TMDPDFBase,
    TMDFFBase,
    MAP22_DEFAULT_EVOLUTION,
    MAP22_DEFAULT_PDF_PARAMS,
    MAP22_DEFAULT_FF_PARAMS,
)

from .fnp_manager import fNPManager

# Re-export flavor-blind classes for unified public interface
from .fnp_base_flavor_blind import (
    fNP_evolution as fNP_evolution_flavor_blind,  # Same evolution class
    TMDPDFFlavorBlind,
    TMDFFFlavorBlind,
    MAP22_DEFAULT_EVOLUTION_FLAVOR_BLIND,
    MAP22_DEFAULT_PDF_PARAMS_FLAVOR_BLIND,
    MAP22_DEFAULT_FF_PARAMS_FLAVOR_BLIND,
)

from .fnp_manager_flavor_blind import (
    fNPManagerFlavorBlind,
    load_flavor_blind_config,
)

# Make the main interface easily accessible
__all__ = [
    # Standard fNP classes
    "fNPManager",  # Main unified interface for standard flavor-dependent system
    "fNP_evolution",  # Evolution factor module
    "TMDPDFBase",  # TMD PDF base class
    "TMDFFBase",  # TMD FF base class
    "MAP22_DEFAULT_EVOLUTION",  # Default evolution parameters
    "MAP22_DEFAULT_PDF_PARAMS",  # Default PDF parameters
    "MAP22_DEFAULT_FF_PARAMS",  # Default FF parameters
    # Flavor-blind fNP classes
    "fNPManagerFlavorBlind",  # Main unified interface for flavor-blind system
    "fNP_evolution_flavor_blind",  # Same evolution class (re-exported for clarity)
    "TMDPDFFlavorBlind",  # Flavor-blind TMD PDF base class
    "TMDFFFlavorBlind",  # Flavor-blind TMD FF base class
    "load_flavor_blind_config",  # Configuration loader for flavor-blind system
    "MAP22_DEFAULT_EVOLUTION_FLAVOR_BLIND",  # Default evolution parameters (flavor-blind)
    "MAP22_DEFAULT_PDF_PARAMS_FLAVOR_BLIND",  # Default PDF parameters (flavor-blind)
    "MAP22_DEFAULT_FF_PARAMS_FLAVOR_BLIND",  # Default FF parameters (flavor-blind)
]

# Version info
__version__ = "1.0.0"
__author__ = "Chiara Bissolotti"
__email__ = "cbissolotti@anl.gov"
