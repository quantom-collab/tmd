"""
module: fNP.py
author: Chiara Bissolotti (cbissolotti@anl.gov)

This file contains the fNP parametrization for the TMDPDFs.
"""

# fNP.py
import torch
import torch.nn as nn
from typing import Optional, Union, List, Dict, Any

# IPython imports - handle gracefully if not available
# Try to import IPython display functionality for notebook environments
try:
    import importlib

    ipython_display = importlib.import_module("IPython.display")
    display = ipython_display.display
    Latex = ipython_display.Latex
    HAS_IPYTHON = True
except ImportError:
    # Fallback for environments without IPython
    HAS_IPYTHON = False

    def display(obj):
        """Fallback display function when IPython is not available."""
        print(obj)

    def Latex(text):
        """Fallback Latex function when IPython is not available."""
        return text


###############################################################################
# 1. Base TMD PDF Parameterization for a Single Flavor
###############################################################################
class TMDPDFBase(nn.Module):
    def __init__(self, n_flavors: int, init_params: List[float], free_mask: List[bool]):
        """
        Base TMD PDF parametrization for a specific quark flavor.

        The (default) forward function uses a simple model.
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
        init_tensor = (
            torch.tensor(init_params, dtype=torch.float32)
            .unsqueeze(0)
            .repeat(n_flavors, 1)
        )

        # Create a mask tensor (1 for free, 0 for fixed); shape = (1, n_params)
        mask = torch.tensor(free_mask, dtype=torch.float32).unsqueeze(0)
        self.register_buffer("mask", mask)

        # Store the fixed part (non‑trainable)
        fixed_init = init_tensor * (1 - mask)
        self.register_buffer("fixed_params", fixed_init)

        # Store the free part (trainable)
        free_init = init_tensor * mask
        self.free_params = nn.Parameter(free_init)

        # Ensure that during backpropagation, only free entries get gradients.
        self.free_params.register_hook(lambda grad: grad * mask)

    @property
    def get_params_tensor(self):
        """Return the full parameter tensor (shape: n_flavors x n_params)."""
        return self.fixed_params + self.free_params

    def forward(
        self,
        x: torch.Tensor,
        b: torch.Tensor,
        zeta: torch.Tensor,
        NP_evol: torch.Tensor,
        flavor_idx: int = 0,
    ) -> torch.Tensor:
        """
        Forward pass for the pure Gaussian parametrization.
        This is the default functional form; subclasses may override forward.

        For the given flavor (selected by flavor_idx, usually 0), extract the parameters
        and compute fNP(x, b).
        This implementation of the non-perturbative function f_{NP} for the unpolarized TMD PDF
        assumes the following form:

            f_{NP}(x, b_T) = NP_evol * exp(-λ * b_T^2) * exp( g1 * ln(1/x)),

        Parameter mapping:
        - p[0] = g1     : Exponent governing the x behavior.
        - p[1] = lambda : Gaussian width parameter controlling the fall-off in b_T space.

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
        # We assume the parameter vector has 2 elements.
        p = self.get_params_tensor[flavor_idx]

        # Unpack the parameters.
        g1 = p[0]  # For x-dependence.
        lam = p[1]  # Gaussian width parameter (λ).

        # Compute the x-dependent width factor:
        # T(x) = g1 * ln(1/x)
        T_x = g1 * torch.log(1 / x)

        # Compute the Gaussian factor in b_T space:
        # G(b) = exp(-λ * b^2)
        G_b = torch.exp(-lam * (b**2) - T_x * (b**2))

        # Combine the factors with the shared evolution factor.
        # The final non-perturbative function is given by:
        # f_NP(x, b_T) = NP_evol * T(x) * G(b)
        result = NP_evol * G_b

        # Create a mask so that if x >= 1 the output is forced to zero.
        # This is done elementwise: for each element, (x < 1) returns True (1.0) if x < 1, or False (0.0) otherwise.
        mask_val = (x < 1).type_as(result)

        # Return the masked result.
        return result * mask_val

    @property
    def latex_formula(self) -> str:
        r"""
        Returns a LaTeX string representing the analytic form of the non-perturbative function f_{NP}(x, b_T)
        used in this parametrization.

        The parametrization implemented in the forward method is given by:

            f_{NP}(x, b_T) = NP_{evol} \cdot \exp\left[-\lambda\, b_T^2\right]
                                       \cdot \exp\left[- g_1\, \ln\frac{1}{x}\, b_T^2\right]

        where:
        - g_1 is the parameter governing the x-dependence,
        - lambda is the Gaussian width parameter,
        - NP_{evol} is the shared evolution factor computed externally.

        Note: This form ensures that at b_T = 0,
            exp[-lambda 0^2 + g_1 ln(1/x)] = exp[g_1 ln(1/x)],
            so if NP_{evol} is normalized to 1 and if one wishes to impose f_{NP}(x,0)=1,
            further normalization may be applied.
        """
        return r"""$$
        f_{NP}(x,b_T) = NP_{evol} \cdot \exp\left[-\lambda\, b_T^2 - g_1\, \ln\frac{1}{x}\, b_T^2\right]\,
        $$
        """

    @property
    def show_latex_formula(self) -> None:
        """
        Automatically render the LaTeX formula in a Jupyter notebook.
        NOTE: this is a property, not a method.
        NOTE: all the classes that inherit from this class will have this property.
        """
        if HAS_IPYTHON:
            display(Latex(self.latex_formula))
        else:
            print("LaTeX formula:")
            print(self.latex_formula)


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
        return torch.exp(-(self.g2**2) * (b**2) * torch.log(zeta) / 4)


###############################################################################
# 3. Flavor-Specific Subclasses
###############################################################################
# For example, you might want a different parameterization (even a different number of parameters)
# for each flavor. It is assumed that flavors not explicitly subclassed use the base class.


class TMDPDF_u(TMDPDFBase):
    def __init__(
        self,
        n_flavors: int = 1,
        init_params: Optional[List[float]] = None,
        free_mask: Optional[List[bool]] = None,
    ):
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
            init_params = [
                0.25,
                0.15,
                0.12,
                0.10,
                0.20,
                0.18,
                0.08,
                0.14,
                0.13,
                0.11,
                0.09,
            ]

        if free_mask is None:

            # All parameters are set as trainable by default.
            free_mask = [True] * 11

        # Initialize the base class with these parameters.
        super().__init__(n_flavors, init_params, free_mask)

    def forward(
        self,
        x: torch.Tensor,
        b: torch.Tensor,
        zeta: torch.Tensor,
        NP_evol: torch.Tensor,
        flavor_idx: int = 0,
    ) -> torch.Tensor:
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
        N1 = p[0]
        alpha1 = p[1]
        sigma1 = p[2]
        lam = p[3]  # 'lam' stands for lambda (we cannot use "lambda")
        N1B = p[4]
        N1C = p[5]
        lam2 = p[6]  # lambda2, renamed to lam2
        alpha2 = p[7]
        alpha3 = p[8]
        sigma2 = p[9]
        sigma3 = p[10]

        # Compute the three g-functions
        g1 = (
            N1
            * torch.pow(x / xhat, sigma1)
            * torch.pow((1 - x) / (1 - xhat), alpha1**2)
        )
        g1B = (
            N1B
            * torch.pow(x / xhat, sigma2)
            * torch.pow((1 - x) / (1 - xhat), alpha2**2)
        )
        g1C = (
            N1C
            * torch.pow(x / xhat, sigma3)
            * torch.pow((1 - x) / (1 - xhat), alpha3**2)
        )

        # Compute (b/2)^2
        b_half_sq = (b / 2) ** 2

        # Compute the numerator
        num = (
            g1 * torch.exp(-g1 * b_half_sq)
            + (lam**2) * (g1B**2) * (1 - g1B * b_half_sq) * torch.exp(-g1B * b_half_sq)
            + g1C * (lam2**2) * torch.exp(-g1C * b_half_sq)
        )

        # Compute the denominator
        den = g1 + (lam**2) * (g1B**2) + g1C * (lam2**2)

        # Multiply by the shared evolution factor (NPevol) and divide by the denominator.
        result = NP_evol * num / den

        # Apply a mask so that if x >= 1 the result is forced to zero.
        mask_val = (x < 1).type_as(result)
        return result * mask_val

    @property
    def latex_formula(self) -> str:
        r"""
        Return a LaTeX string representing the analytic form of the u-quark TMD PDF
        parameterization. This formula is based on the C++ code of MAP22.
        
        $$
        f(x,b) = N_{Pevol} \cdot \frac{
            g_1 \, \exp\left(-g_1\left(\frac{b}{2}\right)^2\right)
          + \lambda^2 \, g_{1B}^2 \left(1 - g_{1B}\left(\frac{b}{2}\right)^2\right)
            \exp\left(-g_{1B}\left(\frac{b}{2}\right)^2\right)
          + g_{1C} \, \lambda_2^2 \, \exp\left(-g_{1C}\left(\frac{b}{2}\right)^2\right)
        }{
            g_1 + \lambda^2 \, g_{1B}^2 + g_{1C} \, \lambda_2^2
        }
        \\
        \text{where } 
        \\
        g_1 = N_1 \left(\frac{x}{x_{\text{hat}}}\right)^{\sigma_1}
            \left(\frac{1-x}{1-x_{\text{hat}}}\right)^{\alpha_1^2},\\[1mm]
        \\    
        g_{1B} = N_{1B} \left(\frac{x}{x_{\text{hat}}}\right)^{\sigma_2}
            \left(\frac{1-x}{1-x_{\text{hat}}}\right)^{\alpha_2^2},\\[1mm]
        \\    
        g_{1C} = N_{1C} \left(\frac{x}{x_{\text{hat}}}\right)^{\sigma_3}
            \left(\frac{1-x}{1-x_{\text{hat}}}\right)^{\alpha_3^2},\quad
        \\    
        x_{\text{hat}} = 0.1.
        $$
        """
        return r"""$$
        f(x,b) = N_{Pevol} \cdot \frac{
            g_1 \, \exp\left(-g_1\left(\frac{b}{2}\right)^2\right)
          + \lambda^2 \, g_{1B}^2 \left(1 - g_{1B}\left(\frac{b}{2}\right)^2\right)
            \exp\left(-g_{1B}\left(\frac{b}{2}\right)^2\right)
          + g_{1C} \, \lambda_2^2 \, \exp\left(-g_{1C}\left(\frac{b}{2}\right)^2\right)
        }{
            g_1 + \lambda^2 \, g_{1B}^2 + g_{1C} \, \lambda_2^2
        }
        \\
        \text{where } 
        \\
        g_1 = N_1 \left(\frac{x}{x_{\text{hat}}}\right)^{\sigma_1}
            \left(\frac{1-x}{1-x_{\text{hat}}}\right)^{\alpha_1^2},\\[1mm]
        \\    
        g_{1B} = N_{1B} \left(\frac{x}{x_{\text{hat}}}\right)^{\sigma_2}
            \left(\frac{1-x}{1-x_{\text{hat}}}\right)^{\alpha_2^2},\\[1mm]
        \\    
        g_{1C} = N_{1C} \left(\frac{x}{x_{\text{hat}}}\right)^{\sigma_3}
            \left(\frac{1-x}{1-x_{\text{hat}}}\right)^{\alpha_3^2},\quad
        \\    
        x_{\text{hat}} = 0.1.
        $$
        """


class TMDPDF_d(TMDPDFBase):
    r"""
    d-quark TMD PDF parameterization using the MAP24 form.

    This parameterization implements the following analytic form (MAP24):

    Let x_hat = 0.1. Define the intermediate functions:

    g_1d = N_1d * (x/x_hat)^sigma_1d * ((1-x)/(1-x_hat))^(alpha_1d^2)
    g_2d = N_2d * (x/x_hat)^sigma_2d * ((1-x)/(1-x_hat))^(alpha_2d^2)
    g_3d = N_3d * (x/x_hat)^sigma_3d * ((1-x)/(1-x_hat))^(alpha_3d^2)

    with sigma_3d taken equal to sigma_2d.

    Then the d-quark non-perturbative function is given by:

    f_NP^d(x,b_T) = N_Pevol * [
        g_1d * exp(-g_1d * (b_T/2)^2)
      + lambda_1d^2 * g_2d^2 * (1 - g_2d * (b_T/2)^2) * exp(-g_2d * (b_T/2)^2)
      + g_3d * lambda_2d^2 * exp(-g_3d * (b_T/2)^2)
    ] / [
        g_1d + lambda_1d^2 * g_2d^2 + g_3d * lambda_2d^2
    ]

    The shared evolution factor N_Pevol is provided externally.

    Parameter mapping (from the d-quark parameter vector p):
      - p[0] = N_1d
      - p[1] = N_2d
      - p[2] = N_3d
      - p[3] = alpha_1d
      - p[4] = alpha_2d
      - p[5] = alpha_3d
      - p[6] = sigma_1d
      - p[7] = sigma_2d  (and sigma_3d = sigma_2d)
      - p[8] = lambda_1d
      - p[9] = lambda_2d

    The forward method receives the shared evolution factor (NP_evol) as an argument and uses it directly.
    """

    def __init__(
        self,
        n_flavors: int = 1,
        init_params: Optional[List[float]] = None,
        free_mask: Optional[List[bool]] = None,
    ):
        # Set default parameters for the d-quark if none are provided.
        # The default parameter vector should have 10 elements as described above.
        if init_params is None:
            init_params = [
                0.21,
                0.015,
                4e-3,
                0.86,
                5.50,
                2.38,
                -0.21,
                9.91,
                0.32,
                0.052,
            ]
        if free_mask is None:
            free_mask = [True] * 10
        super().__init__(n_flavors, init_params, free_mask)

    def forward(
        self,
        x: torch.Tensor,
        b: torch.Tensor,
        zeta: torch.Tensor,
        NP_evol: torch.Tensor,
        flavor_idx: int = 0,
    ) -> torch.Tensor:
        r"""
        Compute the d-quark TMD PDF using the MAP24 parametrization.

        Parameters:
          x (torch.Tensor): The Bjorken x variable.
          b (torch.Tensor): The impact parameter (b_T).
          zeta (torch.Tensor): The energy scale (included for consistency).
          NP_evol (torch.Tensor): The shared evolution factor computed externally.
          flavor_idx (int): The row index for this flavor's parameter vector (default is 0).

        Returns:
          torch.Tensor: The computed d-quark TMD PDF.

        The computation follows:

        1. Define x_hat = 0.1.
        2. Compute:
           g_1d = N_1d * (x/x_hat)^sigma_1d * ((1-x)/(1-x_hat))^(alpha_1d^2)
           g_2d = N_2d * (x/x_hat)^sigma_2d * ((1-x)/(1-x_hat))^(alpha_2d^2)
           g_3d = N_3d * (x/x_hat)^sigma_3d * ((1-x)/(1-x_hat))^(alpha_3d^2)
           with sigma_3d = sigma_2d.

        3. Compute the numerator:
           num = g_1d * exp(-g_1d * (b/2)^2)
               + lambda_1d^2 * g_2d^2 * (1 - g_2d * (b/2)^2) * exp(-g_2d * (b/2)^2)
               + g_3d * lambda_2d^2 * exp(-g_3d * (b/2)^2)

        4. Compute the denominator:
           den = g_1d + lambda_1d^2 * g_2d^2 + g_3d * lambda_2d^2.

        5. The d-quark TMD PDF is then:
           f_NP^d(x,b_T) = NP_evol * num / den.

        A mask is applied so that if x >= 1, the output is forced to zero.
        """
        # Define the fixed constant.
        xhat = 0.1

        # Extract the parameter vector for this flavor.
        # p is a 1D tensor with 10 elements.
        p = self.get_params_tensor[flavor_idx]

        # Unpack the parameters.
        N1d = p[0]
        N2d = p[1]
        N3d = p[2]
        alpha1d = p[3]
        alpha2d = p[4]
        alpha3d = p[5]
        sigma1d = p[6]
        sigma2d = p[7]  # And sigma3d is taken equal to sigma2d.
        lambda1d = p[8]
        lambda2d = p[9]

        # Compute the three g-functions:
        g1d = (
            N1d
            * torch.pow(x / xhat, sigma1d)
            * torch.pow((1 - x) / (1 - xhat), alpha1d**2)
        )
        g2d = (
            N2d
            * torch.pow(x / xhat, sigma2d)
            * torch.pow((1 - x) / (1 - xhat), alpha2d**2)
        )
        g3d = (
            N3d
            * torch.pow(x / xhat, sigma2d)
            * torch.pow((1 - x) / (1 - xhat), alpha3d**2)
        )

        # Compute (b/2)^2.
        b_half_sq = (b / 2) ** 2

        # Compute the numerator:
        num = (
            g1d * torch.exp(-g1d * b_half_sq)
            + (lambda1d**2)
            * (g2d**2)
            * (1 - g2d * b_half_sq)
            * torch.exp(-g2d * b_half_sq)
            + g3d * (lambda2d**2) * torch.exp(-g3d * b_half_sq)
        )

        # Compute the denominator:
        den = g1d + (lambda1d**2) * (g2d**2) + g3d * (lambda2d**2)

        # Multiply by the shared evolution factor NP_evol and divide by the denominator.
        result = NP_evol * num / den

        # Apply a mask so that if x ≥ 1 the output is forced to zero.
        mask_val = (x < 1).type_as(result)
        return result * mask_val

    @property
    def latex_formula(self) -> str:
        r"""
        Returns a LaTeX string representing the analytic form of the d-quark TMD PDF
        parameterization using the MAP24 parametrization.

        The parametrization is given by:

        f_NP^d(x,b_T) = NP_evol * [numerator] / [denominator]

        where numerator and denominator are defined in the LaTeX formula below.
        """
        return r"""$$
        f_{NP}^d(x,b_T) = NP_{evol} \cdot \frac{
            g_{1d}\, \exp\left[-g_{1d}\left(\frac{b_T}{2}\right)^2\right]
        + \lambda_{1d}^2\, g_{2d}^2\, \Bigl(1 - g_{2d}\left(\frac{b_T}{2}\right)^2\Bigr)
            \exp\left[-g_{2d}\left(\frac{b_T}{2}\right)^2\right]
        + g_{3d}\, \lambda_{2d}^2\, \exp\left[-g_{3d}\left(\frac{b_T}{2}\right)^2\right]
        }{
            g_{1d} + \lambda_{1d}^2\, g_{2d}^2 + g_{3d}\, \lambda_{2d}^2
        }
        $$
        """


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

        # Print a message to indicate that the fNP module is being initialized.
        print(
            "\033[94m"
            + "[fNP.py] Initializing fNP module. Reading fNPconfig file"
            + "\033[0m"
        )

        # First, call the constructor of the parent class (nn.Module)
        # to initialize internal machinery.
        super().__init__()

        # Extract global settings.
        self.hadron = config.get("hadron", "not_specified")
        self.zeta = torch.tensor(
            config.get("zeta", 0.0), dtype=torch.float32
        )  # Convert zeta to a torch.Tensor
        # so that torch.log will work properly.

        # Read the initial g2 value from the configuration file.
        # If the key "evolution" does not exist, return an empty dictionary.
        # If the key does not exist, set a default value of 0.2.
        init_g2 = config.get("evolution", {}).get("init_g2", 0.2)

        # If the "evolution" key exists in the configuration dictionary,
        # raise an exception with a clear error message.
        if "evolution" not in config:
            raise KeyError(
                "\033[93m" + "Missing 'evolution' key in configuration." + "\033[0m"
            )

        # Print a message to indicate that the shared g2 is being initialized.
        print(
            "\033[94m"
            + f"[fNP.py] Initializing shared g2 with value {init_g2}."
            + "\033[0m"
        )

        # Create the shared evolution module.
        self.NPevolution = fNP_evolution(init_g2=init_g2)

        # Extract the flavor-specific configurations.
        flavor_config = config.get("flavors", {})

        # Define the order of the flavors we want to support.
        # The order here is important because later we will update parameters using a tensor
        # where each row corresponds to a flavor in this specific order.
        self.flavor_keys = ["u", "ubar", "d", "dbar", "c", "cbar", "s", "sbar"]

        # Create a temporary Python dictionary to hold the modules for each flavor.
        flavors = {}

        # Loop over each flavor key in our defined order.
        for key in self.flavor_keys:
            # Retrieve the configuration for the current flavor.
            # If it is not provided in the config, use a default configuration.
            cfg = flavor_config.get(key, None)

            if cfg is None:
                # If a flavor is not defined in the config, use a default 2-parameter parameterization.
                # Default: 2 parameters with all of them set as free (trainable)
                cfg = {"init_params": [0.1, 0.1], "free_mask": [True, True]}

            # Create the fNP module, with a specific parameterization for each flavor.
            # For demonstration, use specific subclasses for 'u' and 'd'; otherwise use the base class.
            # TMDPDF_u is a subclass of the base class, potentially with its own forward method.
            if key == "u":
                module = TMDPDF_u(
                    n_flavors=1,
                    init_params=cfg["init_params"],
                    free_mask=cfg["free_mask"],
                )
            elif key == "d":
                module = TMDPDF_d(
                    n_flavors=1,
                    init_params=cfg["init_params"],
                    free_mask=cfg["free_mask"],
                )
            else:
                # For flavors not specifically subclassed, we use the base class.
                module = TMDPDFBase(
                    n_flavors=1,
                    init_params=cfg["init_params"],
                    free_mask=cfg["free_mask"],
                )

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
        self.NPevolution.g2.data.copy_(new_g2.clone().detach())

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
            module.fixed_params.data.copy_(new_params.unsqueeze(0) * (1 - mask))

            # Update the free (trainable) part of the parameters.
            # For free parameters, we multiply by mask so that only free entries are updated.
            module.free_params.data.copy_(new_params.unsqueeze(0) * mask)

    def forward(
        self, x: torch.Tensor, b: torch.Tensor, flavors: Optional[List[str]] = None
    ) -> Dict[str, torch.Tensor]:
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
            outputs[key] = self.flavors[key](x, b, self.zeta, shared_evol, flavor_idx=0)

        # Return the dictionary mapping flavor keys to their computed outputs.
        return outputs

    def get_parameter_info(self) -> Dict[str, Any]:
        """
        Get detailed information about model parameters including trainability status.

        Returns:
            Dict containing parameter counts, breakdowns by flavor, and trainability info
        """
        info = {
            "total_parameters": 0,
            "truly_trainable_parameters": 0,
            "pytorch_trainable_parameters": sum(
                p.numel() for p in self.parameters() if p.requires_grad
            ),
            "flavors": {},
            "evolution": {},
        }

        # Evolution parameter (g2)
        if hasattr(self, "NPevolution") and hasattr(self.NPevolution, "g2"):
            info["evolution"] = {"total": 1, "trainable": 1, "fixed": 0}
            info["total_parameters"] += 1
            info["truly_trainable_parameters"] += 1

        # Flavor parameters
        for flavor_key in self.flavor_keys:
            flavor_module = self.flavors[flavor_key]
            n_params = flavor_module.n_params
            info["total_parameters"] += n_params

            if hasattr(flavor_module, "mask"):
                free_count = int(flavor_module.mask.sum().item())
                fixed_count = n_params - free_count
                info["truly_trainable_parameters"] += free_count

                info["flavors"][flavor_key] = {
                    "total": n_params,
                    "trainable": free_count,
                    "fixed": fixed_count,
                    "mask": flavor_module.mask.squeeze().tolist(),
                }
            else:
                info["truly_trainable_parameters"] += n_params
                info["flavors"][flavor_key] = {
                    "total": n_params,
                    "trainable": n_params,
                    "fixed": 0,
                    "mask": [True] * n_params,
                }

        return info

    def print_parameter_summary(self):
        """Print a formatted summary of model parameters and their trainability."""
        info = self.get_parameter_info()

        print("\n" + "=" * 60)
        print("fNP MODEL PARAMETER SUMMARY")
        print("=" * 60)
        print(f"Total parameters: {info['total_parameters']}")
        print(f"Truly trainable parameters: {info['truly_trainable_parameters']}")
        print(f"PyTorch trainable parameters: {info['pytorch_trainable_parameters']}")
        print(
            f"Fixed parameters: {info['total_parameters'] - info['truly_trainable_parameters']}"
        )

        print("\nEvolution parameter:")
        evo = info["evolution"]
        print(
            f"  g2: {evo['total']} total ({evo['trainable']} trainable, {evo['fixed']} fixed)"
        )

        print("\nFlavor parameters:")
        for flavor, data in info["flavors"].items():
            print(
                f"  {flavor}: {data['total']} total ({data['trainable']} trainable, {data['fixed']} fixed)"
            )
            if data["total"] <= 15:  # Show mask for small parameter sets
                mask_str = ["T" if m else "F" for m in data["mask"]]
                print(f"      mask: [{', '.join(mask_str)}]")

        print("=" * 60 + "\n")

    def get_trainable_parameters_dict(self) -> Dict[str, torch.Tensor]:
        """
        Extract only the truly trainable parameters (considering masks) for optimization.

        This method is useful for:
        - Saving only parameters that should be fitted
        - Creating optimizers with only trainable parameters
        - Parameter transfer between models with different fixed/free configurations

        Returns:
            Dict mapping parameter names to tensors containing only trainable values
        """
        trainable_params = {}

        # Evolution parameter (always trainable)
        if hasattr(self, "NPevolution") and hasattr(self.NPevolution, "g2"):
            trainable_params["evolution.g2"] = self.NPevolution.g2.data.clone()

        # Flavor parameters (only free ones according to mask)
        for flavor_key in self.flavor_keys:
            flavor_module = self.flavors[flavor_key]
            if hasattr(flavor_module, "mask"):
                # Get full parameters and apply mask to extract only trainable ones
                full_params = (
                    flavor_module.get_params_tensor.squeeze()
                )  # Remove flavor dimension
                mask = flavor_module.mask.squeeze().bool()  # Convert to boolean mask
                trainable_values = full_params[
                    mask
                ]  # Extract only trainable parameters
                trainable_params[f"flavors.{flavor_key}.trainable"] = (
                    trainable_values.clone()
                )
            else:
                # If no mask, all are trainable
                full_params = flavor_module.get_params_tensor.squeeze()
                trainable_params[f"flavors.{flavor_key}.trainable"] = (
                    full_params.clone()
                )

        return trainable_params

    def set_trainable_parameters_dict(self, trainable_params: Dict[str, torch.Tensor]):
        """
        Update model with trainable parameters only.

        This method allows updating only the trainable parameters while leaving
        fixed parameters unchanged. Useful for loading fitted parameters.

        Args:
            trainable_params: Dict from get_trainable_parameters_dict()
        """
        # Update evolution parameter
        if "evolution.g2" in trainable_params:
            self.NPevolution.g2.data.copy_(trainable_params["evolution.g2"])

        # Update flavor parameters
        for flavor_key in self.flavor_keys:
            param_key = f"flavors.{flavor_key}.trainable"
            if param_key in trainable_params:
                flavor_module = self.flavors[flavor_key]
                trainable_values = trainable_params[param_key]

                if hasattr(flavor_module, "mask"):
                    # Create full parameter tensor and set trainable values
                    full_params = flavor_module.get_params_tensor.squeeze()
                    mask = flavor_module.mask.squeeze().bool()

                    # Update only the trainable parameters
                    full_params[mask] = trainable_values

                    # Update the module's parameters
                    fixed_part = full_params * (~mask).float()
                    free_part = full_params * mask.float()

                    flavor_module.fixed_params.data.copy_(fixed_part.unsqueeze(0))
                    flavor_module.free_params.data.copy_(free_part.unsqueeze(0))
                else:
                    # If no mask, all parameters are trainable
                    full_params = trainable_values
                    flavor_module.free_params.data.copy_(full_params.unsqueeze(0))
