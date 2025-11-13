"""
Configuration loader for SIDIS calculations

This module loads configuration from config.yaml, providing a single source of truth
for all physics and computational settings across the SIDIS codebase.

This replaces the old cfg.py approach, ensuring all modules use consistent settings
without having to modify multiple files.

Usage:
    from qcdlib import config_loader as cfg
    print(cfg.Q20)
    print(cfg.tmd_order)
"""

import yaml
import os

# Load configuration from YAML file
_config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.yaml')

try:
    with open(_config_path, 'r') as f:
        _config = yaml.safe_load(f)
except FileNotFoundError:
    raise FileNotFoundError(
        f"Configuration file not found: {_config_path}\n"
        "Make sure config.yaml exists in the sidis/ directory."
    )
except yaml.YAMLError as e:
    raise ValueError(f"Error parsing config.yaml: {e}")

# Extract core physics/QCD settings with the same names as the old cfg.py
alphaS_order = _config['alphaS_order']
dglap_order = _config['dglap_order']
idis_order = _config['idis_order']
tmd_order = _config['tmd_order']
tmd_resummation_order = _config['tmd_resummation_order']
Q20 = _config['Q20']

# Keep the full config dict accessible for other uses (e.g., grid paths)
config = _config

def get_grid_path(pdf_or_ff, hadron, flavor):
    """
    Helper function to get grid file paths from config.
    
    Args:
        pdf_or_ff: 'pdf' or 'ff'
        hadron: 'p' for proton (pdf) or 'pi_plus' for pion (ff)
        flavor: 'u', 'd', 's', 'c', 'ub', 'db', 'sb', 'cb'
    
    Returns:
        Path to grid file
    """
    return _config['ope']['grid_files'][pdf_or_ff][hadron][flavor]

