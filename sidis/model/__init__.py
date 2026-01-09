"""
This is the main module for the trainable model for SIDIS and TMDs.
"""

import torch
import pathlib
from omegaconf import OmegaConf
from typing import List
from .ope import OPE
from .evolution import PERTURBATIVE_EVOLUTION
from .ogata import OGATA#, OGATA1
from .fnp_factory import create_fnp_manager
from qcdlib.eweak import EWEAK
from qcdlib import params


class TrainableModel(torch.nn.Module):
    def __init__(
        self, 
        fnp_config: str = None,
        experimental_target_fragmented_hadron: List[str] = [["p","pi_plus"]],
    ):
        """
        Initialize the trainable model for SIDIS TMD cross-section computation.

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

        # Load fNP config from cards folder
        # If fnp_config is not provided, use default
        if fnp_config is None:
            fnp_config = "fNPconfig_base_flavor_blind.yaml"

        # Construct full path to config file in cards/ directory
        fnp_config_path = rootdir.joinpath("../cards", fnp_config)

        # Validate that the config file exists
        if not fnp_config_path.exists():
            raise FileNotFoundError(
                f"fNP configuration file not found: {fnp_config_path}\n"
                f"Please ensure the file exists in the cards/ directory."
            )

        self.fnpconf = OmegaConf.load(fnp_config_path)
 
        flavs = ['u','d','s','c','cb','sb','db','ub'] #--have to get this from the config file; need to update this later
        self.flavs = flavs

        self.quark_charges_squared = {'u': 4/9, 'd': 1/9, 's': 1/9, 'c': 4/9, 'cb': 4/9, 'sb': 1/9, 'db': 1/9, 'ub': 4/9}

        self.ope = {}
        self.ope["pdf"] = {} #--this is the unpolarized PDF OPE
        self.ope["ff"] = {} #--this is the unpolarized FF OPE
        # TODO: set up the proper OPE for the Sivers function, Collins function, and transversity function
        self.ope["Sivers"] = {} #--this is the Sivers function OPE
        # self.ope["Collins"] = {} #--this is the Collins function OPE
        # self.ope["h1"] = {} #--this is the transversity function OPE
        for expt_setup in experimental_target_fragmented_hadron:
            #--set up the OPE for the initial hadron
            self.ope["pdf"][expt_setup[0]] = {}
            self.setup_ope(rootdir=rootdir, type="pdf", hadron=expt_setup[0])
            #TODO: set up the OPE for the Sivers function
            self.ope["Sivers"][expt_setup[0]] = {}
            print('Setting up the Sivers function OPE for the initial hadron. WARNING: THIS IS NOT SET UP PROPERLY YET. It is just a stopgap measure. Refer to the init for future development.')
            self.setup_ope(rootdir=rootdir, type="Sivers", hadron=expt_setup[0]) #WARNING: THIS IS NOT SET UP PROPERLY YET. It is just a stopgap measure.
            #TODO: set up the OPE for the transversity function
            #self.ope["h1"][expt_setup[0]] = {}
            #self.setup_ope(rootdir=rootdir, type="h1", hadron=expt_setup[0])

            #--set up the OPE for the fragmented hadron
            self.ope["ff"][expt_setup[1]] = {}
            self.setup_ope(rootdir=rootdir, type="ff", hadron=expt_setup[1])
            #TODO: set up the OPE for the Collins function
            #self.ope["Collins"][expt_setup[0]] = {}
            #self.setup_ope(rootdir=rootdir, type="Collins", hadron=expt_setup[1])

        self.expt_setup = experimental_target_fragmented_hadron

        self.Q20 = self.conf.Q20

        self.evo = PERTURBATIVE_EVOLUTION(order=self.conf.tmd_resummation_order)

        self.ogata_for_J0 = OGATA(nu=0) #-- J0 Hankel transform for FUUT
        self.ogata_for_J1 = OGATA(nu=1) #-- J1 Hankel transform for FUTS

        # Create fNP manager based on its config (includes Sivers function)
        self.nonperturbative = create_fnp_manager(config_dict=self.fnpconf)

        #--set up alpha_EM
        self.alpha_em = EWEAK()

    def setup_ope(self, rootdir: pathlib.Path, type: str = "pdf", hadron: str = "p"):
        """
        Setup the OPE for the given type and hadron.
        We set up the instance of the class for each flavor of the hadron.
        Here we also perform flavor mixing,
            - for neutron, we take the proton u -> d, d -> u, ub -> db, db -> ub
            - for pi-, we take the pi+ u -> d, d -> u, ub -> db, db -> ub
        Args:
            type: "pdf" or "ff" or "Sivers"
            hadron: "p" or "n" or "pi_plus" or "pi_minus"
        Returns:
            None
        """

        if type == "Sivers":
            for flav in self.flavs:
                self.ope[type][hadron][flav] = OPE(rootdir.joinpath(self.conf.ope.grid_files["pdf"][hadron][flav]))
        else:
            for flav in self.flavs:
                self.ope[type][hadron][flav] = OPE(rootdir.joinpath(self.conf.ope.grid_files[type][hadron][flav]))

        # for flav in self.flavs:
        #     self.ope[type][hadron][flav] = OPE(rootdir.joinpath(self.conf.ope.grid_files[type][hadron][flav]))

        if hadron == "n":
            ope_copy = {}
            ope_copy["u"] = self.ope[type][hadron]["d"]
            ope_copy["d"] = self.ope[type][hadron]["u"]
            ope_copy["ub"] = self.ope[type][hadron]["db"]
            ope_copy["db"] = self.ope[type][hadron]["ub"]
            self.ope[type][hadron] = ope_copy
        if hadron == "pi_minus":
            ope_copy = {}
            ope_copy["u"] = self.ope[type][hadron]["d"]
            ope_copy["d"] = self.ope[type][hadron]["u"]
            ope_copy["ub"] = self.ope[type][hadron]["db"]
            ope_copy["db"] = self.ope[type][hadron]["ub"]
            self.ope[type][hadron] = ope_copy

    def get_tmd_bT(self, xi: torch.Tensor, Q2: torch.Tensor, bT: torch.Tensor, type: str = "pdf", hadron: str = "p", flav: str = "u") -> torch.Tensor:
        """
        Compute the TMD PDF or FF in bT-space for the given type.
        Args:
            xi: x or z (depending on the type) (tensor of shape (n_events,))
            Q2: Hard scale squared (tensor of shape (n_events,))
            bT: Fourier inverse of qT (tensor of shape (n_events, n_bT))
            type: "pdf" or "Sivers" or "h1"
            hadron: "p" or "n" for the initial state hadron or "pi_plus" or "pi_minus" for the fragmented hadron
            flav: flavor of the TMD
        Returns:
            TMD in bT-space: tensor of shape (n_events, n_flavs)
        """
        evolution = self.evo(bT, self.Q20, Q2) #--this is the perturbative evolution evaluated from Q0**2 to Q**2

        ope_tmd = self.ope[type][hadron][flav](xi, bT) #--this is the OPE for the TMD evaluated at Q0 as interpolated over the grids

        """
        TODO: there needs to be a modification for the fNP calculations.
        We need the flavors to be able to be calculated separately.
        We need the NP CS kernel to be able to be calculated separately.
        We also need to unify the naming conventions for the fNP with the OPEs or whatever is in the config.

        Currently, this will calculate the entire gamet of flavors.
        
        As a side note as well. For some reason, the fNP creates negative values in the FUUT_integrand. We need to investigate this further.
        Checked by setting the fNP to 1, we are able to get positive FUUT_integrand values.
        """
        if 'b' in flav and len(flav) > 1:
            npflav = flav[0]+'bar'
        else:
            npflav = flav

        if type == "pdf":
            fNP = self.nonperturbative.forward_pdf(xi, bT, Q2**0.5)[npflav]
        elif type == "ff":
            fNP = self.nonperturbative.forward_ff(xi, bT, Q2**0.5)[npflav]
        elif type == "Sivers":
            fNP = self.nonperturbative.forward_sivers(xi, bT, Q2**0.5)#[npflav] #--does not have a flavor dependence yet.
        else:
            raise ValueError(f"Invalid type: {type}")

        return ope_tmd * fNP * evolution

    """
    The following functions are used to compute the structure functions in bT-space.
    """
    def get_FUUT_integrand(self, x: torch.Tensor, Q2: torch.Tensor, z: torch.Tensor, bT: torch.Tensor, initial_hadron: str = "p", fragmented_hadron: str = "pi_plus") -> torch.Tensor:
        """
        Compute the integrand for the FUUT structure function.
        Sum over all flavors and integrate over bT.

        Keep in mind that bT is multiplied here because the Ogata code is defined as the specific Ogata quadrature, not the general Hankel transform.

        """
        FUUT_integrand = torch.zeros_like(bT)
        for flav in self.flavs:
            FUUT_integrand += self.quark_charges_squared[flav] * bT * self.get_tmd_bT(x, Q2, bT, type="pdf", hadron=initial_hadron, flav=flav) * self.get_tmd_bT(z, Q2, bT, type="ff", hadron=fragmented_hadron, flav=flav)
        return FUUT_integrand

    def get_FUT_sin_phih_minus_phis_integrand(self, x: torch.Tensor, Q2: torch.Tensor, z: torch.Tensor, bT: torch.Tensor, initial_hadron: str = "p", fragmented_hadron: str = "pi_plus") -> torch.Tensor:
        """
        Compute the integrand for the FUT sin(phih - phis) structure function.
        Sum over all flavors and integrate over bT.

        Keep in mind that bT**2/2 is multiplied here because the Ogata code is defined as the specific Ogata quadrature, not the general Hankel transform.
        """
        FUTS_sin_phih_minus_phis_integrand = torch.zeros_like(bT)
        for flav in self.flavs:
            FUTS_sin_phih_minus_phis_integrand += self.quark_charges_squared[flav] * bT**2/2 * self.get_tmd_bT(x, Q2, bT, type="Sivers", hadron=initial_hadron, flav=flav) * self.get_tmd_bT(z, Q2, bT, type="ff", hadron=fragmented_hadron, flav=flav)
        return FUTS_sin_phih_minus_phis_integrand

    """
    The following functions are used to compute the structure functions in PhT-space.
    """
    def get_FUUT(self, x: torch.Tensor, Q2: torch.Tensor, z: torch.Tensor, qT: torch.Tensor, bT: torch.Tensor, initial_hadron: str = "p", fragmented_hadron: str = "pi_plus") -> torch.Tensor:
        """
        Compute the FUUT structure function.

        Args:
            x: Bjorken x (tensor of shape (n_events,))
            Q2: Hard scale squared (tensor of shape (n_events,))
            z: Energy fraction of hadron relative to struck quark (tensor of shape (n_events,))
            qT: Transverse momentum in the process (for SIDIS, qT = PhT/z) (tensor of shape (n_events,))
            bT: Fourier inverse of qT (tensor of shape (n_events, n_bT))
            initial_hadron: "p" or "n" for the initial state hadron
            fragmented_hadron: "pi_plus" or "pi_minus" for the fragmented hadron
        Returns:
            FUUT: FUUT structure function
        """
        FUUT_integrand = self.get_FUUT_integrand(x, Q2, z, bT, initial_hadron, fragmented_hadron)
        return self.ogata_for_J0.eval_ogata_func_var_h(FUUT_integrand, bT, qT)

    def get_FUT_sin_phih_minus_phis(self, x: torch.Tensor, Q2: torch.Tensor, z: torch.Tensor, qT: torch.Tensor, bT: torch.Tensor, initial_hadron: str = "p", fragmented_hadron: str = "pi_plus") -> torch.Tensor:
        """
        Compute the FUT sin(phih - phis) structure function.

        Args:
            x: Bjorken x (tensor of shape (n_events,))
            Q2: Hard scale squared (tensor of shape (n_events,))
            z: Energy fraction of hadron relative to struck quark (tensor of shape (n_events,))
            qT: Transverse momentum in the process (for SIDIS, qT = PhT/z) (tensor of shape (n_events,))
            bT: Fourier inverse of qT (tensor of shape (n_events, n_bT))
            initial_hadron: "p" or "n" for the initial state hadron
            fragmented_hadron: "pi_plus" or "pi_minus" for the fragmented hadron
        Returns:
            FUT_sin_phih_minus_phis: FUT sin(phih - phis) structure function
        """
        FUT_sin_phih_minus_phis_integrand = self.get_FUT_sin_phih_minus_phis_integrand(x, Q2, z, bT, initial_hadron, fragmented_hadron)
        return self.ogata_for_J1.eval_ogata_func_var_h(FUT_sin_phih_minus_phis_integrand, bT, qT)

    def forward(self, events_tensor: torch.Tensor, expt_setup: List[str] = ["p","pi_plus"], rs: float = 140.0) -> torch.Tensor:
        """
        Forward pass for batch of events.
        events_tensor: shape (n_events, 6) containing [x, PhT, Q, z, phih, phis] for each event
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

        # Compute qT. This is the qT for which the Fourier transform is defined.
        qT = PhT / z

        # Setting the bT values for ogata quadrature (different grids for J0 and J1)
        bT_for_J0 = self.ogata_for_J0.get_bTs(qT)  # bT grid for J0 (FUUT)
        bT_for_J1 = self.ogata_for_J1.get_bTs(qT)  # bT grid for J1 (FUTS)

        # Setting the final scale
        Q2 = Q**2

        # Setting the initial and fragmented hadrons
        initial_hadron = expt_setup[0]
        fragmented_hadron = expt_setup[1]

        # Hankel transform for FUUT structure function (J0)
        FUUT = self.get_FUUT(x, Q2, z, qT, bT_for_J0, initial_hadron, fragmented_hadron)

        # Hankel transform for FUTS structure function (J1)
        FUT_sin_phih_minus_phis = self.get_FUT_sin_phih_minus_phis(x, Q2, z, qT, bT_for_J1, initial_hadron, fragmented_hadron)

        # Compute the sigma0s for the SIDIS cross-section
        alpha_em = torch.tensor([self.alpha_em.get_alpha(_) for _ in Q2.numpy()])

        # Factors taken from Bacchetta, et al. JHEP 02 (2007) 093
        gamma = 2 * params.M2 * x / Q
        y = Q2 / x / (s - params.M2)
        epsilon = (1 - y - 1/4 * gamma**2 * y**2) / (1 - y + 1/2 * y**2 + 1/4 * gamma**2 * y**2)
        sigma0 = 8 * torch.pi**2 * alpha_em**2 * z**2 * qT / x / Q**3 * y**2 / 2 / (1 - epsilon) * (1 + gamma**2 / 2 / x)

        # Compute the SIDIS cross-section according to the sigma0s and the structure functions
        xsec = sigma0 * (FUUT + torch.sin(phih - phis) * FUT_sin_phih_minus_phis)

        """
        TODO
        Add the capability to compute the Hankel transforms for combined bT-space structure functions.
        For example, we want to compute the Hankel transform of order 1 for the F_{UT}^{sivers} and F_{UT}^{Collins}structure functions in one go.
        Currently, we are computing the PhT-space structure functions separately.

        Add in the FUT_sin_phih_plus_phis structure function.
        This is the structure function that is used to compute the SIDIS cross-section for the Collins function.
        It is given by:
        FUT_sin_phih_plus_phis = FUT_sin_phih_plus_phis * sin(phih + phis)
        """

        return xsec
