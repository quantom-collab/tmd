'''
utilities.py
author: Chiara Bissolotti (cbissolotti@anl.gov)

This file contains utility functions that are used in the map module
'''

import yaml
import torch
import matplotlib.pyplot as plt

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

###############################################################################
# Plot fNP
###############################################################################
def plot_fNP(model_fNP, x, flavors=None):
    """
    Evaluate and plot the fNP values for a given fNP model over a range of b values.

    Parameters:
      - model_fNP (nn.Module): An instance of your fNP module.
      - x (torch.Tensor): The input tensor for x (typically a scalar tensor).
      - b_range (torch.Tensor): A 1D tensor of b values at which to evaluate fNP.
      - flavors (list, optional): A list of flavor keys to evaluate.
                                  If None, uses model_fNP.flavor_keys.
    
    The function evaluates the model for each b in b_range and collects the fNP
    output for each flavor. It then produces a plot of fNP versus b for all flavors.
    """
    # If no specific flavors are provided, use all available flavors.
    if flavors is None:
        flavors = model_fNP.flavor_keys

    # Create a range of b values from 0 to 10.
    b_values = torch.linspace(0, 10, steps=100)

    # Create a dictionary to store the computed fNP for each flavor.
    results_dict = {flavor: [] for flavor in flavors}

    # Loop over the b values, evaluating the model at each b.
    for b in b_values:
        # The model's forward method uses the internally stored zeta.
        outputs = model_fNP(x, b)
        for flavor in flavors:
            # We assume that each output is a scalar tensor.
            results_dict[flavor].append(outputs[flavor].item())

    # Plot all flavors on the same figure.
    plt.figure(figsize=(10, 6))
    for flavor, values in results_dict.items():
        plt.plot(b_values.numpy(), values, label=flavor)
    plt.xlabel('b')
    plt.ylabel('fNP')
    plt.title('fNP vs. b for all flavors')
    plt.legend()
    plt.show()