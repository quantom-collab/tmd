'''
utilities.py
author: Chiara Bissolotti (cbissolotti@anl.gov)

This file contains utility functions that are used in the map module
'''

import yaml


###############################################################################
# YAML Configuration Loader
###############################################################################
def load_yaml_config(yaml_file_path: str) -> dict:
    """
    Load a YAML configuration file.
    
    The file should contain entries for each flavor.
    """
    
    with open(yaml_file_path, 'r') as f:
        config = yaml.safe_load(f)
    return config