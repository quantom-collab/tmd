
from . import special
import torch
from . import config_loader as cfg

class MODEL_TORCH:
    
    def __init__(self):
        #--params
        euler=special.euler

        self.Q0 = torch.sqrt(torch.tensor(cfg.Q20))
        self.C1=2*torch.exp(torch.tensor(-euler))
        self.C2=1
        
        self.bmax = self.C1/self.Q0 #--so that mub > mu_0
    
    #============================================================
    #--Defining functions associated with bstar
    #============================================================
    #--typical logarithm in perturbative expansions
    def get_Log(self,bT,mu):  
        """
        Typical log in the expansion
        """
        return torch.log(mu*bT/self.C1)

    #--Setting up b_*
    def get_bstar(self,bT):
        """
        Returns b*
        """
        return bT/torch.sqrt(1+bT**2/self.bmax**2)

    def get_bmin(self,Q):
        """
        Returns bmin
        """
        return self.C1/Q

    #--Setting up b_* a la MAP
    def get_bstar_MAP(self,bT,Q):
        """
        Returns b*
        """
        bmax = self.bmax
        bmin = self.get_bmin(Q)
        return bmax * ((1 - torch.exp(-bT**4 / bmax**4)) / (1 - torch.exp(-bT**4 / bmin**4)))**(0.25)

    def get_mub(self,bT): 
        """
        Returns mub* 
        """
        return self.C1/self.get_bstar(bT)

    def get_mub_MAP(self,bT,Q): 
        """
        Returns mub* a la MAP
        """
        return self.C1/self.get_bstar_MAP(bT,Q)
