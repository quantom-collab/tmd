# fNP.py
import torch
import torch.nn as nn
import yaml

###############################################################################
# 1. Base TMD PDF Parameterization for a Single Flavor
###############################################################################
class TMDPDFBase(nn.Module):
    def __init__(self, n_flavors: int, init_params: list, free_mask: list):
        """
        Base TMD PDF parametrization for a specific quark flavor.
        
        The (default) forward function uses a simple model:
        
            evolution = exp(- (g2^2 * b^2 * log(zeta) / 4))
            gaussian  = exp(- a * b^2)
            x_dep     = x^alpha
            result    = evolution * gaussian * x_dep
        
        Here we assume that:
          - p[0] = g2 (controls the evolution),
          - p[1] = a (controls the Gaussian width),
          - p[2] = alpha (controls the x-dependence).
        
        The parameters for this flavor are given by the list `init_params` (for example,
        [0.2, 0.1, 0.1]) and a corresponding boolean list `free_mask` (for example,
        [True, False, True]) indicating which parameters are trainable.
        
        Internally the full parameter vector is stored as:
            params = fixed_params + free_params
        where:
          - fixed_params is a buffer (non-trainable), and
          - free_params is an nn.Parameter (trainable).
        A gradient hook ensures that only the free parameters receive gradient updates.
        
        For a flavor-blind instance (if ever needed), the same vector is replicated to have shape
        (n_flavors, n_params). (Usually each flavor's module is created with n_flavors=1.)
        """
        
         # Call the constructor of the parent class (nn.Module)
        super().__init__()
        
        # Save configuration values to the instance.
        self.n_flavors = n_flavors
        self.n_params = len(init_params)
        
        # Create a tensor from init_params and replicate it: shape = (n_flavors, n_params)
        init_tensor = torch.tensor(init_params, dtype=torch.float32).unsqueeze(0).repeat(n_flavors, 1)
        
        # Create a mask tensor (1 for free, 0 for fixed); shape = (1, n_params)
        mask = torch.tensor(free_mask, dtype=torch.float32).unsqueeze(0)
        self.register_buffer('mask', mask)
        
        # Store the fixed part (non‑trainable)
        fixed_init = init_tensor * (1 - mask)
        self.register_buffer('fixed_params', fixed_init)
        
        # Store the free part (trainable)
        free_init = init_tensor * mask
        self.free_params = nn.Parameter(free_init)
        
        # Ensure that during backpropagation, only free entries get gradients.
        self.free_params.register_hook(lambda grad: grad * mask)
        
    @property
    def get_params_tensor(self):
        """Return the full parameter tensor (shape: n_flavors x n_params)."""
        return self.fixed_params + self.free_params

    def forward(self, x: torch.Tensor, b: torch.Tensor, zeta: torch.Tensor, flavor_idx: int = 0) -> torch.Tensor:
        """
        Forward pass.
        
        For the given flavor (selected by flavor_idx, usually 0), extract the parameters
        and compute:
        
            evolution = exp(- (g2^2 * b^2 * log(zeta) / 4))
            gaussian  = exp(- a * b^2)
            x_dep     = x^alpha
            result    = evolution * gaussian * x_dep
        
        A mask is applied so that if x >= 1, the result is zero.
        
        (This is the default functional form; subclasses may override forward.)
        """
        
        # Extract parameters for the flavor (p has shape (n_params,))
        p = self.get_params_tensor[flavor_idx]
        
        # Default: use the first three parameters.
        g2 = p[0]
        a = p[1]
        alpha = p[2]
        evolution = torch.exp(- (g2 ** 2) * (b ** 2) * torch.log(zeta) / 4)
        gaussian = torch.exp(- a * (b ** 2))
        x_dep = x ** alpha
        result = evolution * gaussian * x_dep
        
        # Use a mask (vectorized) so that result = 0 when x >= 1.
        mask_val = (x < 1).type_as(result)
        return result * mask_val

###############################################################################
# 2. Flavor-Specific Subclasses
###############################################################################
# For example, you might want a different parameterization (even a different number of parameters)
# for each flavor. It is assumed that flavors not explicitly subclassed use the base class.

class TMDPDF_u(TMDPDFBase):
    def __init__(self, n_flavors: int = 1, init_params: list = None, free_mask: list = None):
        # Example for the u quark: use 3 parameters.
        if init_params is None:
            init_params = [0.25, 0.15, 0.12]
        if free_mask is None:
            free_mask = [True, False, True]
        super().__init__(n_flavors, init_params, free_mask)
    # (Inherit the default forward function or override if needed.)

class TMDPDF_d(TMDPDFBase):
    def __init__(self, n_flavors: int = 1, init_params: list = None, free_mask: list = None):
        # Example for the d quark: use 4 parameters.
        if init_params is None:
            init_params = [0.22, 0.12, 0.11, 0.05]
        if free_mask is None:
            free_mask = [True, False, True, True]
        super().__init__(n_flavors, init_params, free_mask)
    
    def forward(self, x: torch.Tensor, b: torch.Tensor, zeta: torch.Tensor, flavor_idx: int = 0) -> torch.Tensor:
        """
        An example forward for the d quark that uses 4 parameters.
        
        Let:
          - p[0] = g2, p[1] = a, p[2] = alpha, p[3] = delta.
        Define a modulation factor, for example:
        
            modulation = 1 + delta * b
        
        Then:
            result = evolution * gaussian * modulation * x^alpha
        """
        p = self.get_params_tensor[flavor_idx]  # shape (4,)
        g2 = p[0]
        a = p[1]
        alpha = p[2]
        delta = p[3]
        evolution = torch.exp(- (g2 ** 2) * (b ** 2) * torch.log(zeta) / 4)
        gaussian = torch.exp(- a * (b ** 2))
        modulation = 1 + delta * b
        x_dep = x ** alpha
        result = evolution * gaussian * modulation * x_dep
        mask_val = (x < 1).type_as(result)
        return result * mask_val

###############################################################################
# 3. Top-Level fNP Module
###############################################################################
class fNP(nn.Module):
    def __init__(self, config: dict):
        """
        Top-level non-perturbative module.
        
        This module instantiates one parameterization for each flavor (u, ubar, d, dbar, c, cbar, s, sbar)
        based on a configuration dictionary (loaded from a YAML file). Each configuration entry specifies:
        
          - init_params: a list of initial parameter values,
          - free_mask: a list of booleans (same length as init_params) indicating which parameters
                       are trainable (True) and which are fixed (False).
        
        For example, the YAML configuration file might look like:
        
            u:
              init_params: [0.25, 0.15, 0.12]
              free_mask: [true, false, true]
            d:
              init_params: [0.22, 0.12, 0.11, 0.05]
              free_mask: [true, false, true, true]
        """
        
        # First, call the constructor of the parent class (nn.Module) 
        # to initialize internal machinery.
        super().__init__()
        
        # Extract global settings.
        self.hadron = config.get("hadron", "not_specified")
        self.zeta = torch.tensor(config.get("zeta", 0.0), dtype=torch.float32) # Convert zeta to a torch.Tensor 
                                                                               # so that torch.log will work properly.
        
        flavor_config = config.get("flavors", {})
       
        # Define the order of the flavors we want to support.
        # The order here is important because later we will update parameters using a tensor
        # where each row corresponds to a flavor in this specific order.
        self.flavor_keys = ['u', 'ubar', 'd', 'dbar', 'c', 'cbar', 's', 'sbar']
        
        # Create a temporary Python dictionary to hold the modules for each flavor.
        flavors = {}
        
        # Loop over each flavor key in our defined order.
        for key in self.flavor_keys:
            # Retrieve the configuration for the current flavor.
            # If it is not provided in the config, use a default configuration.
            cfg = flavor_config.get(key, None)
            
            if cfg is None:
                # If a flavor is not defined in the config, use a default 3-parameter parameterization.
                # Default: 3 parameters with all of them set as free (trainable)
                cfg = {"init_params": [0.2, 0.1, 0.1], "free_mask": [True, True, True]}
            
            # Create the fNP module, with a specific parameterization for each flavor.
            # For demonstration, use specific subclasses for 'u' and 'd'; otherwise use the base class.
            # TMDPDF_u is a subclass of the base class, potentially with its own forward method.
            if key == 'u':
                module = TMDPDF_u(n_flavors = 1, init_params = cfg["init_params"], free_mask = cfg["free_mask"])
            elif key == 'd':
                module = TMDPDF_d(n_flavors = 1, init_params = cfg["init_params"], free_mask = cfg["free_mask"])
            else:
                # For flavors not specifically subclassed, we use the base class.
                module = TMDPDFBase(n_flavors = 1, init_params = cfg["init_params"], free_mask = cfg["free_mask"])
                
            # Add the created module to our temporary dictionary.
            flavors[key] = module
        
        # Wrap the Python dictionary into an nn.ModuleDict.
        # nn.ModuleDict is a container that holds submodules (and registers them properly)
        # so that their parameters are visible to PyTorch (for optimizers, .to(device), saving, etc.).    
        self.flavors = nn.ModuleDict(flavors)
    
    def set_parameters(self, param_tensor: torch.Tensor):
        """
        Update the parameters for each flavor from a tensor.
        
        There are 8 possibile flavors (u, ubar, d, dbar, c, cbar, s, sbar), so       
        the provided param_tensor must have shape (8, max_params), where:
          - each row corresponds to a flavor, in the order of self.flavor_keys;
          - max_params is the maximum number of parameters among all flavors.
        
        For each flavor, only the first n_params (as determined by that flavor's module) are used.
        The update uses the module's free/fixed mask to ensure that only free parameters are updated,
        while fixed parameters remain unchanged. This is done using vectorized tensor operations (i.e. masks).

        """
        
        # Loop over the flavors in the same order as self.flavor_keys.
        for i, key in enumerate(self.flavor_keys):
            
            # Retrieve the module corresponding to the current flavor.
            module = self.flavors[key]
            
            # Check that the module (TMDPDFBase, TMDPDF_d(), etc. etc.) 
            # is correctly registered. module is the parametrization class for
            # the current flavor.
            print(f"key: {key}, module: {module}")
            
            # Get the number of parameters that this flavor's module expects.
            n_params = module.n_params
            
            # Slice out the first n_params elements from the i-th row of the param_tensor.
            new_params = param_tensor[i, :n_params]  # shape: (n_params,)
            
            # Retrieve the mask from the module.
            # The mask has shape (1, n_params): 1 means the parameter is free, 0 means it is fixed.
            mask = module.mask  # This was registered as a buffer in the base class.
            
            # Update the fixed part of the parameters.
            # For fixed parameters, we multiply by (1 - mask) so that only fixed entries are updated.
            # new_params.unsqueeze(0) reshapes the 1D tensor to a row (shape (1, n_params)).
            module.fixed_params.data.copy_( new_params.unsqueeze(0) * (1 - mask) )
            
            # Update the free (trainable) part of the parameters.
            # For free parameters, we multiply by mask so that only free entries are updated.
            module.free_params.data.copy_( new_params.unsqueeze(0) * mask )
    
    def forward(self, x: torch.Tensor, b: torch.Tensor, flavors: list = None):
        """
        Evaluate the TMD PDF for the specified flavors.
        
        Parameters:
          - x, b: Input tensors for the computation.
          - flavors (list, optional): A list of flavor keys to evaluate (e.g. ['u', 'd']).
                                      If None, the module evaluates all flavors (i.e. all keys in self.flavor_keys).
                                    
        This method returns a dictionary mapping each flavor key to its computed TMD PDF.
        Each submodule's forward method is called with flavor_idx = 0 (since each module was instantiated with n_flavors = 1).
        """
        
        # Initialize an empty dictionary to hold the outputs.
        outputs = {}
        
        # If no specific flavors are provided, default to all flavors.
        if flavors is None:
            flavors = self.flavor_keys
            
        # Loop over the requested flavors.    
        for key in flavors:
            # Call the forward method of the module corresponding to the current flavor.
            # The flavor_idx argument is 0 because each flavor module was created with n_flavors = 1.
            # Note that each module (TMDPDFBase, TMDPDF_u, TMDPDF_d) is called with the same zeta value
            # stored in the fNP instance and read from the configuration file.
            outputs[key] = self.flavors[key](x, b, self.zeta, flavor_idx = 0)
        
        # Return the dictionary mapping flavor keys to their computed outputs.    
        return outputs

