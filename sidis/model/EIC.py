"""
This is the main module for the trainable model for SIDIS and TMDs.
"""

import torch
import pathlib
from omegaconf import OmegaConf
from typing import List


def _safe_structure(func_values: torch.Tensor) -> torch.Tensor:
    """
    Replace NaN and +/-inf values in a structure function tensor with 0.

    This acts as a numerical failsafe so that pathological points in the
    parametric fits do not propagate NaNs into the final cross section.
    """
    return torch.nan_to_num(func_values, nan=0.0, posinf=0.0, neginf=0.0)

# Optional components that may be implemented later. Wrap in a try/except so that
# the core `EICModel` can be imported even if these extra modules are not present yet.
try:
    from .ope import OPE
    from .evolution import PERTURBATIVE_EVOLUTION
    from .ogata import OGATA, OGATA1
    from .fnp_factory import create_fnp_manager
except ImportError:
    OPE = None
    PERTURBATIVE_EVOLUTION = None
    OGATA = None
    OGATA1 = None
    create_fnp_manager = None

from qcdlib.eweak import EWEAK
from qcdlib import params


class EICModel(torch.nn.Module):
    def __init__(
        self, 
        fnp_config: str = None,
        experimental_target_fragmented_hadron: List[str] = [["p","pi_plus"]],
    ):
        """
        Initialize the EIC model for SIDIS TMD cross-section computation.

        The default experimental setup is:
            - e (electron) + p (proton) -> X + pi+
        
        Additional work will be needed to extend this to other experimental setups.

        Args:
            fnp_config (str, optional): Name of the fNP configuration file in cards/ directory.
                                       If None, defaults to 'fNPconfig_base_flavor_blind.yaml'.
        """
        super().__init__()

        # Load configuration from YAML file
        # rootdir is the folder named model
        rootdir = pathlib.Path(__file__).resolve().parent

        # rootdir.joinpath puts everything relative to the model folder
        self.conf = OmegaConf.load(rootdir.joinpath("../config.yaml"))

        self.expt_setup = experimental_target_fragmented_hadron

        #--set up alpha_EM
        self.alpha_em = EWEAK()

    def get_FUU(self, x: torch.Tensor, Q2: torch.Tensor, z: torch.Tensor, PhT: torch.Tensor) -> torch.Tensor:
        """
        Compute the FUU structure function for a given x, Q2, z, and PhT.
        These assume known parameters for now. Will need to update this later.

        Parametric fit for FUU is:
        N * x^(a_0 + s * a_1) * (1-x)^(b_0 + s * b_1) * (1 + (c_0 + s * c_1) * x**0.5 + (d_0 + s * d_1) * x) * z^(e_0 + s * e_1) * (1-z)^(f_0 + s * f_1) * 1 / pi / PhT2_width * exp(-PhT^2 / PhT2_width)
        where s = log(log(Q2/0.4^2) / log(1 / 0.4^2))
        """
        pars = torch.tensor([ 0.93372439,  1.30786141,  0.06134706, -2.34530286, 14.8676692 ,
                            1.78991607,  5.67809053,  0.02277823, -0.57720858,  1.73326726,
                            6.03560319, -1.65404477,  0.5657061 , -0.43674577,  9.51595838,
                            0.15448249])
        N,N_1,N_2,a_0,a_1,b_0,b_1,c_0,c_1,d_0,d_1,e_0,e_1,f_0,f_1,PhT2_width = pars
        s = torch.log(torch.log(Q2 / 0.4**2)/torch.log(torch.tensor(1.0)/0.4**2)) #--this is the Duke and Owens type of log-log evolution

        FUU = N * (N_1 * x**(a_0 + s * a_1) * (1-x)**(b_0 + s * b_1) + N_2 * x**(c_0 + s * c_1) * (1-x)**(d_0 + s * d_1)) * z**(e_0 + s * e_1) * (1-z)**(f_0 + s * f_1) / torch.pi / PhT2_width * torch.exp(-PhT**2 / PhT2_width)
        return _safe_structure(FUU)

    def get_Siv(self, x: torch.Tensor, Q2: torch.Tensor, z: torch.Tensor, PhT: torch.Tensor) -> torch.Tensor:
        """
        Compute the Sivers function for a given x, Q2, z, and PhT.
        These assume known parameters for now. Will need to update this later.
        """
        pars = torch.tensor([ 3.45542226e+00,  7.01278293e-01,  2.48720902e+00, -1.97068163e+02,
                            8.42965633e+02,  3.07802783e+00, -1.20001494e+01,  4.30444249e-01,
                            6.32154696e+00,  1.17658172e-01])
        N,a_0,a_1,b_0,b_1,c_0,c_1,d_0,d_1,PhT2_width = pars
        s = torch.log(torch.log(Q2 / 0.4**2)/torch.log(torch.tensor(1.0)/0.4**2)) #--this is the Duke and Owens type of log-log evolution

        Siv = N * x**(a_0 + s * a_1) * (1-x)**(b_0 + s * b_1) * z**(c_0 + s * c_1) * (1-z)**(d_0 + s * d_1) / torch.pi / PhT2_width * torch.exp(-PhT**2 / PhT2_width) * PhT
        return _safe_structure(Siv)

    def get_Col(self, x: torch.Tensor, Q2: torch.Tensor, z: torch.Tensor, PhT: torch.Tensor) -> torch.Tensor:
        """
        Compute the Collins function for a given x, Q2, z, and PhT.
        These assume known parameters for now. Will need to update this later.
        """
        pars = torch.tensor([ 2.82814598e+01,  1.72271021e+01, -6.63895494e+01,  1.34225316e+02,
                            -5.48699975e+02, -3.65683482e+01,  1.56858301e+02, -2.16146761e+02,
                            9.25982700e+02,  1.91114494e-01])
        N,a_0,a_1,b_0,b_1,c_0,c_1,d_0,d_1,PhT2_width = pars
        s = torch.log(torch.log(Q2 / 0.4**2)/torch.log(torch.tensor(1.0)/0.4**2)) #--this is the Duke and Owens type of log-log evolution

        Col = N * x**(a_0 + s * a_1) * (1-x)**(b_0 + s * b_1) * z**(c_0 + s * c_1) * (1-z)**(d_0 + s * d_1) / torch.pi / PhT2_width * torch.exp(-PhT**2 / PhT2_width) * PhT
        return _safe_structure(Col)

    def forward(self, events_tensor: torch.Tensor, expt_setup: List[str] = ["p","pi_plus"], rs: float = 140.0) -> torch.Tensor:
        """
        Forward pass for batch of events.
        events_tensor: shape (n_events, 6) containing [x, PhT, Q, z, phih, phis] for each event
        This assumes either:
        - an unpolarized SIDIS cross-section is being computed, so phih and phis are set to 0
        - a transversely polarized SIDIS cross-section is being computed, so phih and phis are provided in the events_tensor
        """
        # Unpack the variables from the events tensor
        x = events_tensor[:, 0]  # Bjorken x
        PhT = events_tensor[:, 1]  # Transverse momentum of detected hadron
        Q = events_tensor[:, 2]  # Hard scale
        z = events_tensor[:, 3]  # Energy fraction of hadron relative to struck quark
        if len(events_tensor.shape) == 6:
            phih = events_tensor[:, 4]  # Azimuthal angle of hadron (radians)
        else:
            phih = torch.zeros_like(x)
        if len(events_tensor.shape) == 6:
            phis = events_tensor[:, 5]  # Azimuthal angle of target spin (radians)
        else:
            phis = torch.zeros_like(x)

        # Compute qT
        qT = PhT / z

        # Compute Q2
        Q2 = Q**2

        # Compute the structure functions
        FUU = self.get_FUU(x, Q2, z, PhT) #--unpolarized structure function
        Siv = self.get_Siv(x, Q2, z, PhT) #--Sivers structure function (F_{UT}^{\sin(\phi_h - \phi_s)})
        Col = self.get_Col(x, Q2, z, PhT) #--Collins structure function (F_{UT}^{\sin(\phi_h + \phi_s)})

        print('FUU:', FUU)
        print('Siv:', Siv)
        print('Col:', Col)

        # Compute the sigma0s for the SIDIS cross-section
        alpha_em = torch.tensor([self.alpha_em.get_alpha(_) for _ in Q2.numpy()])
        # Factors taken from Bacchetta, et al. JHEP 02 (2007) 093
        gamma = 2 * params.M2 * x / Q
        y = Q2 / x / (rs**2 - params.M2)
        epsilon = (1 - y - 1/4 * gamma**2 * y**2) / (1 - y + 1/2 * y**2 + 1/4 * gamma**2 * y**2)
        sigma0 = 8 * torch.pi**2 * alpha_em**2 * z**2 * qT / x / Q**3 * y**2 / 2 / (1 - epsilon) * (1 + gamma**2 / 2 / x)

        # Compute the SIDIS cross-section according to the sigma0s and the structure functions
        #--note that if phih and phis are set to 0, then the sin(phih - phis) and sin(phih + phis) terms will be 0, consistent with an unpolarized target
        #--if phih and phis are provided in the events_tensor, then the sin(phih - phis) and sin(phih + phis) terms will be non-zero, consistent with a transversely polarized target
        xsec = sigma0 * (FUU + torch.sin(phih - phis) * Siv + epsilon * torch.sin(phih + phis) * Col)

        return xsec
