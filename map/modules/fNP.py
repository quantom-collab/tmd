'''
module: fNP.py
author: Chiara Bissolotti (cbissolotti@anl.gov)

This file contains the fNP parametrization for the TMDPDFs. 
'''

# fNP.py
import torch
import torch.nn as nn
from IPython.display import display, Latex

###############################################################################
# 1. Base TMD PDF Parameterization for a Single Flavor
###############################################################################
class TMDPDFBase(nn.Module):
    def __init__(self, n_flavors: int, init_params: list, free_mask: list):
        """
        Base TMD PDF parametrization for a specific quark flavor.
        
        The (default) forward function uses a simple model:
        
            evolution = NP_evol
            gaussian  = exp(- N1 * b^2)
            N(x)      = N0 * x^alpha * (1-x)^beta
            f_NP(x, b_T) = evolution * N(x) * gaussian(bT)
           
        NP_evol is the shared evolution factor computed outside this class, 
        from the fNP_evolution module.   
        
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
    
    def forward(self, x: torch.Tensor, b: torch.Tensor, zeta: torch.Tensor, 
            NP_evol: torch.Tensor, flavor_idx: int = 0) -> torch.Tensor:
        """
        Forward pass for the pure Gaussian parametrization with x-dependent normalization.
        This is the default functional form; subclasses may override forward.
        
        For the given flavor (selected by flavor_idx, usually 0), extract the parameters
        and compute fNP(x, b).
        This implementation of the non-perturbative function f_{NP} for the unpolarized TMD PDF 
        assumes the following form:
        
            f_{NP}(x, b_T) = NP_evol * N(x) * exp(-λ * b_T^2),
        
        where the x-dependent normalization is defined as:
        
            N(x) = N_0 * x^(α) * (1-x)^(β),
        
        with the following parameter mapping:
        - p[0] = N_0    : Overall normalization factor.
        - p[1] = alpha  : Exponent governing the behavior at small x.
        - p[2] = beta   : Exponent governing the behavior at large x.
        - p[3] = lambda : Gaussian width parameter controlling the fall-off in b_T space.
        
        The shared evolution factor NP_evol is computed externally (from the fNP_evolution module)
        and is applied to all flavors to ensure a consistent evolution factor across the board.
        
        Parameters:
        x (torch.Tensor)          : The Bjorken x variable.
        b (torch.Tensor)          : The impact parameter (b_T).
        zeta (torch.Tensor)       : The energy scale (included for consistency; not used in this form).
        NP_evol (torch.Tensor)    : The precomputed shared evolution factor.
        flavor_idx (int, optional): The row index for this flavor's parameter vector (default is 0).
        
        Returns:
        torch.Tensor: The computed non-perturbative function f_{NP}(x, b_T) as a tensor.
        
        A mask is applied so that if x ≥ 1 the result is forced to zero.
        """
        # Extract the parameter vector for the given flavor.
        # We assume the parameter vector has 4 elements:
        # [N_0, α, β, λ]
        p = self.get_params_tensor[flavor_idx]
        
        # Unpack the parameters.
        N0    = p[0]  # Overall normalization factor.
        alpha = p[1]  # Exponent for x.
        beta  = p[2]  # Exponent for (1-x).
        lam   = p[3]  # Gaussian width parameter (λ).
        
        # Compute the x-dependent normalization:
        # N(x) = N0 * x^(α) * (1-x)^(β)
        N_x = N0 * torch.pow(x, alpha) * torch.pow((1 - x), beta)
        
        # Compute the Gaussian factor in b_T space:
        # G(b) = exp(-λ * b^2)
        G_b = torch.exp(-lam * (b ** 2))
        
        # Combine the factors with the shared evolution factor.
        # The final non-perturbative function is given by:
        # f_NP(x, b_T) = NP_evol * N(x) * G(b)
        result = NP_evol * N_x * G_b
        
        # Create a mask so that if x >= 1 the output is forced to zero.
        # This is done elementwise: for each element, (x < 1) returns True (1.0) if x < 1, or False (0.0) otherwise.
        mask_val = (x < 1).type_as(result)
        
        # Return the masked result.
        return result * mask_val


###############################################################################
# 2. fNP Evolution Module
###############################################################################
class fNP_evolution(nn.Module):
    def __init__(self, init_g2: float):
        """
        Evolution factor module.
        
        This class computes the evolution factor for the non-perturbative TMD PDF. 
        The evolution factor depends on the parameter g2, b, and the rapidity scale (zeta). 
        Parameters:
            init_g2 (float):
                Initial value for the evolution factor parameter g2. 
                This value is used to initialize a trainable parameter. 
        Attributes:
            g2 (torch.nn.Parameter):
                A trainable parameter initialized with the value of init_g2, 
                representing the non-perturbative evolution factor. This parameter 
                is optimized during training to best fit the TMD PDF model.
        """
                
        # Call the constructor of the parent class (nn.Module) 
        super().__init__()
        
        # g2 is a trainable parameter shared across all flavors.
        self.g2 = nn.Parameter(torch.tensor(init_g2, dtype=torch.float32))
    
    def forward(self, b: torch.Tensor, zeta: torch.Tensor):
        """
        Compute the evolution factor that depends only on g2, b, and zeta.
        This method computes the evolution factor for the non-perturbative TMD PDF.
        The evolution factor is given by:
            NPevol = exp(- (g2^2) * (b^2) * log(zeta) / 4)
        Parameters:
            b (torch.Tensor)    : The impact parameter (b_T).
            zeta (torch.Tensor) : The rapidity scale.
        Returns:
            torch.Tensor: The computed evolution factor as a tensor.
        
        The result of this function (a tensor) is shared across all flavors to 
        ensure a consistent evolution factor, NP_evol.    
        """
        
        # Compute the evolution factor that depends only on g2, b, and zeta.
        return torch.exp(- (self.g2 ** 2) * (b ** 2) * torch.log(zeta) / 4)
    
###############################################################################
# 3. Flavor-Specific Subclasses
###############################################################################
# For example, you might want a different parameterization (even a different number of parameters)
# for each flavor. It is assumed that flavors not explicitly subclassed use the base class.

class TMDPDF_u(TMDPDFBase):
    def __init__(self, n_flavors: int = 1, init_params: list = None, free_mask: list = None):
        """
        u-quark TMD PDF parameterization.
        
        This parameterization implements the following analytic form (MAP22):
        
            Let xhat = 0.1.
            Define:
              g1  = N1  * (x/xhat)^(sigma1)  * ((1-x)/(1-xhat))^(alpha1^2)
              g1B = N1B * (x/xhat)^(sigma2)  * ((1-x)/(1-xhat))^(alpha2^2)
              g1C = N1C * (x/xhat)^(sigma3)  * ((1-x)/(1-xhat))^(alpha3^2)
            
            Then the u-quark TMD PDF is given by:
              result = NPevol * ( g1 * exp(-g1*(b/2)^2)
                        + (lambda)^2 * (g1B)^2 * (1 - g1B*(b/2)^2) * exp(-g1B*(b/2)^2)
                        + g1C * (lambda2)^2 * exp(-g1C*(b/2)^2) )
                       / ( g1 + (lambda)^2 * (g1B)^2 + g1C * (lambda2)^2 )
            
            Here, the parameters (from the u-quark parameter vector) are:
              p[0] = N1
              p[1] = alpha1
              p[2] = sigma1
              p[3] = lambda      (we avoid using the reserved keyword "lambda" by naming it lam)
              p[4] = N1B
              p[5] = N1C
              p[6] = lambda2     (named lam2)
              p[7] = alpha2
              p[8] = alpha3
              p[9] = sigma2
              p[10] = sigma3
        
        The shared evolution factor, NPevol, is provided externally and multiplies the whole expression.
        """
        
        # Set default parameters if none are provided:
        if init_params is None:
            
            # Default values; these can be adjusted as needed.
            init_params = [0.25, 0.15, 0.12, 0.10, 0.20, 0.18, 0.08, 0.14, 0.13, 0.11, 0.09]
            
        if free_mask is None:
            
            # All parameters are set as trainable by default.
            free_mask = [True] * 11
            
        # Initialize the base class with these parameters.
        super().__init__(n_flavors, init_params, free_mask)
    
    def forward(self, x: torch.Tensor, b: torch.Tensor, zeta: torch.Tensor, 
                NP_evol: torch.Tensor, flavor_idx: int = 0) -> torch.Tensor:
        """
        Compute the u-quark TMD PDF using the analytic form of MAP22.
        
        Parameters:
          - x (torch.Tensor)      : The Bjorken x variable.
          - b (torch.Tensor)      : The impact parameter.
          - zeta (torch.Tensor)   : The energy scale (used in the evolution factor inside the integrals).
          - NPevol (torch.Tensor) : The shared evolution factor computed from the evolution module.
          - flavor_idx (int)      : The row index for this flavor's parameters (usually 0).
        
        Returns:
          - result (torch.Tensor): The computed u-quark TMD PDF.
        """
        # Retrieve the parameter vector for this flavor.
        # p is a 1D tensor with 11 elements.
        p = self.get_params_tensor[flavor_idx]
        
        # Define the fixed constant xhat.
        xhat = 0.1
        
        # Unpack the parameters.
        N1     = p[0]
        alpha1 = p[1]
        sigma1 = p[2]
        lam    = p[3]  # 'lam' stands for lambda (we cannot use "lambda")
        N1B    = p[4]
        N1C    = p[5]
        lam2   = p[6]  # lambda2, renamed to lam2
        alpha2 = p[7]
        alpha3 = p[8]
        sigma2 = p[9]
        sigma3 = p[10]
        
        # Compute the three g-functions using the analytic expressions:
        # g1  = N1  * (x/xhat)^(sigma1)  * ((1-x)/(1-xhat))^(alpha1^2)
        # g1B = N1B * (x/xhat)^(sigma2)  * ((1-x)/(1-xhat))^(alpha2^2)
        # g1C = N1C * (x/xhat)^(sigma3)  * ((1-x)/(1-xhat))^(alpha3^2)
        g1  = N1  * torch.pow(x / xhat, sigma1)  * torch.pow((1 - x) / (1 - xhat), alpha1 ** 2)
        g1B = N1B * torch.pow(x / xhat, sigma2)  * torch.pow((1 - x) / (1 - xhat), alpha2 ** 2)
        g1C = N1C * torch.pow(x / xhat, sigma3)  * torch.pow((1 - x) / (1 - xhat), alpha3 ** 2)
        
        # Compute (b/2)^2
        b_half_sq = (b / 2) ** 2
        
        # Compute the numerator according to the formula:
        # Numerator = g1 * exp(-g1 * (b/2)^2)
        #           + (lam^2) * (g1B)^2 * (1 - g1B*(b/2)^2) * exp(-g1B*(b/2)^2)
        #           + g1C * (lam2^2) * exp(-g1C*(b/2)^2)
        num = (g1 * torch.exp(-g1 * b_half_sq)
               + (lam ** 2) * (g1B ** 2) * (1 - g1B * b_half_sq) * torch.exp(-g1B * b_half_sq)
               + g1C * (lam2 ** 2) * torch.exp(-g1C * b_half_sq))
        
        # Compute the denominator according to the formula:
        # Denom = g1 + (lam^2) * (g1B)^2 + g1C * (lam2^2)
        den = g1 + (lam ** 2) * (g1B ** 2) + g1C * (lam2 ** 2)
        
        # Multiply by the shared evolution factor (NPevol) and divide by the denominator.
        result = NP_evol * num / den
        
        # Apply a mask so that if x >= 1 the result is forced to zero.
        mask_val = (x < 1).type_as(result)
        return result * mask_val
    
    @property
    def latex_formula(self):
        """
        Return a LaTeX string representing the analytic form of the u-quark TMD PDF
        parameterization. This formula is based on the C++ code provided.
        
        The formula is:
        
            f(x,b) = N_{Pevol} \cdot \frac{
                g_1 \, \exp\left(-g_1\left(\frac{b}{2}\right)^2\right)
              + \lambda^2 \, g_{1B}^2 \left(1 - g_{1B}\left(\frac{b}{2}\right)^2\right)
                \exp\left(-g_{1B}\left(\frac{b}{2}\right)^2\right)
              + g_{1C} \, \lambda_2^2 \, \exp\left(-g_{1C}\left(\frac{b}{2}\right)^2\right)
            }{
                g_1 + \lambda^2 \, g_{1B}^2 + g_{1C} \, \lambda_2^2
            }
            
        with:
        
            g_1  = N_1 \left(\frac{x}{x_{\text{hat}}}\right)^{\sigma_1}
                     \left(\frac{1-x}{1-x_{\text{hat}}}\right)^{\alpha_1^2},\\[1mm]
            g_{1B} = N_{1B} \left(\frac{x}{x_{\text{hat}}}\right)^{\sigma_2}
                     \left(\frac{1-x}{1-x_{\text{hat}}}\right)^{\alpha_2^2},\\[1mm]
            g_{1C} = N_{1C} \left(\frac{x}{x_{\text{hat}}}\right)^{\sigma_3}
                     \left(\frac{1-x}{1-x_{\text{hat}}}\right)^{\alpha_3^2},\quad
            x_{\text{hat}} = 0.1.
        """
        return r"""
        f(x,b) = N_{Pevol} \cdot \frac{
            g_1 \, \exp\left(-g_1\left(\frac{b}{2}\right)^2\right)
          + \lambda^2 \, g_{1B}^2 \left(1 - g_{1B}\left(\frac{b}{2}\right)^2\right)
            \exp\left(-g_{1B}\left(\frac{b}{2}\right)^2\right)
          + g_{1C} \, \lambda_2^2 \, \exp\left(-g_{1C}\left(\frac{b}{2}\right)^2\right)
        }{
            g_1 + \lambda^2 \, g_{1B}^2 + g_{1C} \, \lambda_2^2
        }
        
        \text{where } 
        g_1 = N_1 \left(\frac{x}{x_{\text{hat}}}\right)^{\sigma_1}
            \left(\frac{1-x}{1-x_{\text{hat}}}\right)^{\alpha_1^2},\\[1mm]
        g_{1B} = N_{1B} \left(\frac{x}{x_{\text{hat}}}\right)^{\sigma_2}
            \left(\frac{1-x}{1-x_{\text{hat}}}\right)^{\alpha_2^2},\\[1mm]
        g_{1C} = N_{1C} \left(\frac{x}{x_{\text{hat}}}\right)^{\sigma_3}
            \left(\frac{1-x}{1-x_{\text{hat}}}\right)^{\alpha_3^2},\quad
        x_{\text{hat}} = 0.1.
        """
        
    @property    
    def show_formula(self):
        """
        Automatically render the LaTeX formula in a Jupyter notebook.
        """
        display(Latex(self.latex_formula))    


class TMDPDF_d(TMDPDFBase):
    """
    d-quark TMD PDF parameterization.
    
    This parameterization implements A DUMB analytic form at the moment. 
    """
    def __init__(self, n_flavors: int = 1, init_params: list = None, free_mask: list = None):
        if init_params is None:
            init_params = [0.22, 0.12, 0.11, 0.05]
        if free_mask is None:
            free_mask = [True, False, True, True]
        super().__init__(n_flavors, init_params, free_mask)
    
    def forward(self, x: torch.Tensor, b: torch.Tensor, zeta: torch.Tensor,
                NP_evol: torch.Tensor, flavor_idx: int = 0) -> torch.Tensor:
        """
        An example forward for the d quark that uses 4 parameters and the shared evolution factor.
        
        Let:
          - shared_g2 = the shared evolution parameter (g2) provided externally,
          - p[1] = a, p[2] = alpha, p[3] = delta.
        Define a modulation factor:
            modulation = 1 + delta * b
        
        Then:
            result = evolution * gaussian * modulation * x^alpha,
        where evolution is computed using the shared g2.
        """
        # Do not extract g2 from p; use the shared one.
        p = self.get_params_tensor[flavor_idx]  
        
        N1 = p[0]
        a = p[1]
        alpha = p[2]
        delta = p[3]
        
        evolution = NP_evol
        gaussian = torch.exp(- a * (b ** 2))
        modulation = N1 + delta * b
        x_dep = x ** alpha
        result = evolution * gaussian * modulation * x_dep
        mask_val = (x < 1).type_as(result)
        return result * mask_val
    
###############################################################################
# 4. Top-Level fNP Module
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
        """
        
        # First, call the constructor of the parent class (nn.Module) 
        # to initialize internal machinery.
        super().__init__()
        
        # Extract global settings.
        self.hadron = config.get("hadron", "not_specified")
        self.zeta = torch.tensor(config.get("zeta", 0.0), dtype=torch.float32) # Convert zeta to a torch.Tensor 
                                                                               # so that torch.log will work properly.
        
        # Read the initial g2 value from the configuration file.
        # If the key "evolution" does not exist, return an empty dictionary.
        # If the key does not exist, set a default value of 0.2.
        init_g2 = config.get("evolution", {}).get("init_g2", 0.2)
        
        # If the "evolution" key exists in the configuration dictionary, 
        # raise an exception with a clear error message.
        if "evolution" not in config:
            raise KeyError("\033[93m" + "Missing 'evolution' key in configuration." + "\033[0m")
        
        # Print a message to indicate that the shared g2 is being initialized.
        print("\033[94m" + f"Initializing shared g2 with value {init_g2}." + "\033[0m")

        # Create the shared evolution module. 
        self.NPevolution = fNP_evolution(init_g2 = init_g2)

        # Extract the flavor-specific configurations.
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
        Update parameters from a tensor for each flavor and for the shared evolution parameter .
        
        The param_tensor MUST have shape (9, max_params), where the first row corresponds to shared g2.
        The remaining 8 rows correspond to the 8 possible flavors (u, ubar, d, dbar, c, cbar, s, sbar)
        in the order of self.flavor_keys.
        
          - the first row corresponds to the shared g2 parameter;
          - each of the following rows corresponds to a flavor, in the order of self.flavor_keys;
          - max_params is the maximum number of parameters among all flavors.
        
        For each flavor, only the first n_params (as determined by that flavor's module) are used.
        The update uses the module's free/fixed mask to ensure that only free parameters are updated,
        while fixed parameters remain unchanged. This is done using vectorized tensor operations (i.e. masks).

        """
        
        # First, update the shared evolution parameter,
        # assuming g2 is a scalar.
        new_g2 = param_tensor[0, 0]  
        self.NPevolution.g2.data.copy_(torch.tensor(new_g2))
        
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
            # Starting from row 1 because of the shared g2.
            new_params = param_tensor[i + 1, :n_params]  # shape: (n_params,)
            
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
            
        # Compute the shared evolution factor for all flavors.
        shared_evol = self.NPevolution(b, self.zeta)
        
        # Loop over the requested flavors.    
        for key in flavors:
            # Call the forward method of the module corresponding to the current flavor.
            # The flavor_idx argument is 0 because each flavor module was created with n_flavors = 1.
            # Note that each module (TMDPDFBase, TMDPDF_u, TMDPDF_d) is called with the same zeta value
            # stored in the fNP instance and read from the configuration file.
            outputs[key] = self.flavors[key](x, b, self.zeta, shared_evol, flavor_idx = 0)
        
        # Return the dictionary mapping flavor keys to their computed outputs.    
        return outputs

