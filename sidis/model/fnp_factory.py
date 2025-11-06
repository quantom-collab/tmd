"""
Factory for creating fNP Manager instances.

This module provides a unified entry point for creating fNP managers based on
configuration files. It automatically selects the appropriate combo implementation
and instantiates the manager from that combo module.

Author: Chiara Bissolotti (cbissolotti@anl.gov)
"""

from typing import Dict, Any
from omegaconf import OmegaConf
import pathlib
import importlib


# Mapping from combo names to module paths (relative imports)
COMBO_MODULES = {
    "standard": ".fnp_base",
    "flavor_blind": ".fnp_base_flavor_blind",
}


def create_fnp_manager(config_path: str = None, config_dict: Dict[str, Any] = None):
    """
    Create an fNP manager instance based on configuration.

    This is the main entry point for creating fNP managers. It reads the config,
    determines which combo to use, and instantiates the manager from that combo module.

    Args:
        config_path: Path to YAML configuration file (optional if config_dict provided)
        config_dict: Configuration dictionary (optional if config_path provided)

    Returns:
        fNP manager instance

    Raises:
        ValueError: If config is invalid or combo not found
        FileNotFoundError: If config_path doesn't exist

    Example:
        >>> from sidis.model.fnp_factory import create_fnp_manager
        >>> manager = create_fnp_manager("fNPconfig_flavor_blind.yaml")
        >>> result = manager.forward(x, z, b, Q)
    """
    # Load configuration
    if config_dict is not None:
        config = config_dict
    elif config_path is not None:
        config_path_obj = pathlib.Path(config_path)
        if not config_path_obj.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        config = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    else:
        raise ValueError("Either config_path or config_dict must be provided")

    # Get combo name from config
    combo_name = config.get("combo", "flavor_blind")  # Default to flavor_blind

    if combo_name not in COMBO_MODULES:
        available = ", ".join(COMBO_MODULES.keys())
        raise ValueError(f"Unknown combo '{combo_name}'. Available combos: {available}")

    # Import the combo module and get the manager class
    module_path = COMBO_MODULES[combo_name]
    try:
        # Use relative import from current package
        module = importlib.import_module(module_path, package=__package__)
    except ImportError as e:
        raise ValueError(f"Failed to import combo module '{module_path}': {e}")

    if not hasattr(module, "fNPManager"):
        raise ValueError(
            f"Combo module '{module_path}' does not export 'fNPManager' class"
        )

    manager_class = module.fNPManager

    # Create and return manager instance
    manager = manager_class(config)

    return manager


def create_fnp_manager_from_dict(config_dict: Dict[str, Any]):
    """Convenience function to create manager from dictionary."""
    return create_fnp_manager(config_dict=config_dict)


def create_fnp_manager_from_file(config_path: str):
    """Convenience function to create manager from file."""
    return create_fnp_manager(config_path=config_path)
