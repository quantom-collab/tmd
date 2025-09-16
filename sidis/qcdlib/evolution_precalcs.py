import mpmath
import torch

import qcdlib.params as params
import qcdlib.cfg as cfg
from qcdlib.alphaS  import ALPHAS

from scipy.special import zeta

alphaS = ALPHAS()



CF=params.CF
CA=params.CA
TR=params.TR
TF=params.TF
zeta2 = zeta(2)
zeta3 = zeta(3)
zeta4 = zeta(4)
zeta5 = zeta(5)

def get_betas():
    beta0 = lambda Nf: alphaS.beta[Nf,0]
    beta1 = lambda Nf: alphaS.beta[Nf,1]
    beta2 = lambda Nf: alphaS.beta[Nf,2]
    beta3 = lambda Nf: alphaS.beta[Nf,3]

    beta = torch.zeros((7,4))
    for Nf in range(3,7):
        beta[Nf] = torch.tensor([beta0(Nf),beta1(Nf),beta2(Nf),beta3(Nf)])
    return beta

betas=get_betas()

#=================================================================================
#--Setting up the roots of the beta function to evaluate the Sudakov analytically
#=================================================================================
"""
These are the setup of the deltas as in https://inspirehep.net/literature/1950250, Eqs. 3.11, 3.12, A.4, A.5, A.9, A.10.
This is how artemide performs the calculation.
Namely, we have the shape of the roots according to (Order of resummation (iorder), Order of summation trucation (n_sum), Nf)

\eta_\Gamma = -1 / 2 * \sum_{i=0}^{n_sum} r_{Gamma,i} \log((\alpha_S(mu) - 4 * np.pi * delta_{i,iorder}) / (\alpha_S(mu_0) - 4 * np.pi * delta_{i,iorder}))

K_\gamma = -1 / 2 * \sum_{i=0}^{n_sum} r_{gamma,i} \log((\alpha_S(mu) - 4 * np.pi * delta_{i,iorder}) / (\alpha_S(mu_0) - 4 * np.pi * delta_{i,iorder}))

K_\Gamma  is more complicated and is given by Eq. 3.17 of https://inspirehep.net/literature/1950250
"""

def setup_beta_roots():

    """
    These terms agree with artemide up to NNLL with the exception of Nf=6.
    The shape is (4,4,7). First index is the order of resummation, second is the order of truncation, third is the number of flavors.
    """
    delta=torch.zeros((4,4,7),dtype=torch.complex128)
    #--LL
    #--delta[0] = 0 for LL
    #--delta_0 = 0 for all orders

    beta = get_betas()

    #--NLL
    for Nf in range(3,7):
        delta[1][1][Nf] = -beta[Nf,0] / beta[Nf,1]
    
    #--N2LL
    for i in [1,2]:
        if i==1: sign = 1
        elif i==2: sign = -1
        for Nf in range(3,7):
            real = -beta[Nf,1] / 2 / beta[Nf,2]
        
            radicand = 4*beta[Nf,0] * beta[Nf,2] - beta[Nf,1]**2
            
            rad_sign = torch.sign(radicand)
            if rad_sign == -1:
                delta[2][i][Nf] = real - sign * torch.sqrt(torch.abs(radicand)) / 2 / beta[Nf,2]
            else:
                delta[2][i][Nf] = complex(real,sign * torch.sqrt(radicand) / 2 / beta[Nf,2])

    #--N3LL
    for Nf in range(3,7):
        #--In our case up to Nf=6, we do not have any negative values in square roots.
        Delta0 = beta[Nf,2]**2 - 3 * beta[Nf,1] * beta[Nf,3]
        Delta1 = beta[Nf,2]**3 - 9/2 * beta[Nf,3] * (beta[Nf,1] * beta[Nf,2] - 3 * beta[Nf,0] * beta[Nf,3])
        C = (Delta1 + torch.sqrt(Delta1**2 - Delta0**3))**(1/3)
        xi = complex(-1/2,torch.sqrt(torch.tensor(3))/2)

        for i in [1,2,3]:
            delta[3][i][Nf] = -1 / 3 / beta[Nf,3] * (beta[Nf,2] + xi**(i) * C + Delta0 / xi**(i) / C)
    
    return delta
delta = setup_beta_roots()

def setup_beta_prime():
    beta = get_betas()

    beta_prime=torch.zeros((4,4,7),dtype=torch.complex128)

    for i in range(4):
        #--looping over orders of resummation
        for j in range(4):
            #--looping over roots of the beta (from delta)
            for Nf in range(3,7):
                #--looping over number of flavors
                delta_ij = delta[i][j][Nf]
                for k in range(i+1): #--we go from 0 to i, but in torch we have range up to i+1
                    beta_prime[i][j][Nf] += -2 * (k+2) * beta[Nf,k] * delta_ij**(k+1)
    return beta_prime
beta_prime = setup_beta_prime()

#============================================================
#--Setting up gamma_K
#============================================================
"""
\$\Gamma\$-cusp

    notice the following correspondence from https://arxiv.org/pdf/1803.11089.pdf 

    Rogers17_PhysRevD.96.054011.pdf uses gamma_K = 2 gamma_Cusp
    - Cusp anomalous dimension expanded in alphaS, Eq. (59) of PhysRevD.96.054011
    https://inspirehep.net/literature/1452696 Appendix D notice that one should use \$\Gamma^{(n-1)}\$ in other parts 


the 4-loop expression is taken from [2001.11377] (appendix C)
    """

def get_Gamma0(Nf):
    """
    PCB checked consistency with artemide: 1/10/2025
    """
    #--aS**1
    #Nf=self.alphaS.get_Nf(mu**2)
    return 8. * CF

def get_Gamma1(Nf):
    """
    PCB checked consistency with artemide: 1/10/2025
    """
    #--aS**2
    #Nf=self.alphaS.get_Nf(mu**2)
    return 8 * CF * (CA * (67/9 - torch.pi**2/3)\
                - 20/9 * TR * Nf)

def get_Gamma2(Nf):
    """
    PCB checked consistency with artemide: 1/10/2025
    """
    #--aS**3
    #Nf=self.alphaS.get_Nf(mu**2)
    return 8 * CF *(CA**2 * (245/6 - 134/27*torch.pi**2 + 11/45*torch.pi**4 + 22/3*zeta3) \
                + CA * TR * Nf * (-418/27 + 40/27*torch.pi**2 - 56/3*zeta3)\
                + CF * TR * Nf * (-55/3 + 16 * zeta3) - 16/27 * TR**2 * Nf**2 )

def get_Gamma3(Nf):
    """
    PCB checked consistency with artemide: 1/10/2025
    """
    #--aS**4
    #Nf=self.alphaS.get_Nf(mu**2)
    dFA=15/16 #--d_F^{abcd}d_A^{abcd}/(N^2-1)
    dFF=5/96 #--d_F^{abcd}d_F^{abcd}/(N^2-1)

    return  2 * CF*( \
            dFA*(7040/3*zeta5 +256/3*zeta3-768*(zeta3**2) -256*zeta2 -15872/35*(zeta2**3))\
            +dFF*Nf*(-2560/3*zeta5 -512/3*zeta3 +512*zeta2)\
            +(Nf**3)*(-32/81 +64/27*zeta3)\
            +CF*(Nf**2)*(2392/81 -640/9*zeta3 +64/5*(zeta2**2))\
            +(CF**2)*Nf*(572/9 -320*zeta5 +592/3*zeta3)\
            +CA*(Nf**2)*(923/81 +2240/27*zeta3 -608/81*zeta2 -224/15*(zeta2**2))\
            +CF*CA*Nf*(-34066/81 +160*zeta5 +3712/9*zeta3 +440/3*zeta2 -128*zeta2*zeta3 -352/5*(zeta2**2))\
            +(CA**2)*Nf*(-24137/81 +2096/9*zeta5 -23104/27*zeta3 +20320/81*zeta2 \
            +448/3*zeta2*zeta3 -352/15*(zeta2**2))\
            +(CA**3)*(84278/81 - 3608/9*zeta5 +20944/27*zeta3 -16*(zeta3**2) \
            -88400/81*zeta2 -352/3*zeta2*zeta3 +3608/5*(zeta2**2)-20032/105*(zeta2**3)))       

Gamma = torch.zeros((7,4),dtype=torch.complex128)
for Nf in range(3,7):
    Gamma[Nf] = torch.tensor([get_Gamma0(Nf),get_Gamma1(Nf),get_Gamma2(Nf),get_Gamma3(Nf)])

def get_Gammas():
    Gammas = torch.zeros((4,4,7),dtype=torch.complex128)
    for i in range(4): #--this is the order of resummation
        for j in range(4): #--this is the order of the delta
            for Nf in range(3,7): #--this is the number of flavors
                delta_ij = delta[i][j][Nf]
                for k in range(i+1): #--we go from 0 to i, but in torch we have range up to i+1
                    Gammas[i][j][Nf] += Gamma[Nf][k] * delta_ij**(k+1)
    return Gammas

Gammas = get_Gammas()

#============================================================
#--Setting up gamma_F
#  https://inspirehep.net/literature/1452696 Appendix D
#============================================================
"""
Here we calculate the \$\gamma_{V}\$ https://inspirehep.net/literature/1452696 Appendix D
"""
def get_gammaV1(Nf):
    """
    PCB checked consistency with artemide: 1/10/2025
    """
    #--aS**1
    # Nf=self.alphaS.get_Nf(mu**2)
    return 6. * CF

def get_gammaV2(Nf):
    """
    PCB checked consistency with artemide: 1/10/2025
    """
    #--aS**2
    #Nf=self.alphaS.get_Nf(mu**2)
    return  CF**2 * (3 - 4*torch.pi**2 + 48*zeta3) + CF*CA*(961/27+11*torch.pi**2/3-52*zeta3) + \
            CF*TR*Nf*(-260/27-4*torch.pi**2/3.)

def get_gammaV3(Nf):
    """
    PCB checked consistency with artemide: 1/10/2025
    """
    #--aS**3
    #Nf=self.alphaS.get_Nf(mu**2)
    
    pi = torch.pi   
    return  CF**3 * (29 + 6*pi**2 + 16/5*pi**4 + 136*zeta3 - 32*pi**2/3 * zeta3 - 480*zeta5) + \
        CF**2*CA *(151/2 - 410/9*pi**2 - 494*pi**4/135 + 1688/3*zeta3 + 16/3*pi**2*zeta3 + 240*zeta5) + \
        CF*CA**2 *(139345/1458+7163/243*pi**2+83/45*pi**4-7052/9*zeta3+88/9*pi**2*zeta3+272*zeta5)+\
        CF**2*TR*Nf*(-5906/27+52/9*pi**2+56/27*pi**4-1024/9*zeta3) + \
        CF*CA*TR*Nf*(34636/729-5188/243*pi**2-44/45*pi**4+3856/27*zeta3) +  \
        CF*TR**2*Nf**2*(-19336/729+80/27*pi**2+64/27*zeta3)

gamma = torch.zeros((7,3),dtype=torch.complex128) #--note here that we have 3 because we do not count the gammaV0
for Nf in range(3,7):
    gamma[Nf] = torch.tensor([get_gammaV1(Nf),get_gammaV2(Nf),get_gammaV3(Nf)])

def get_gammas():
    gammas = torch.zeros((4,4,7),dtype=torch.complex128)
    for i in range(4): #--this is the order of resummation
        for j in range(4): #--this is the order of the delta
            for Nf in range(3,7): #--this is the number of flavors
                delta_ij = delta[i][j][Nf]
                for k in range(i): #--we go from 0 to i-1, so in torch we have range up to i
                    gammas[i][j][Nf] += gamma[Nf][k] * delta_ij**(k+1)
    return gammas

gammas = get_gammas()

# Replace division by zero with 0
r_Gamma = torch.where(beta_prime != 0, -2 * Gammas / beta_prime, 0)
r_gamma = torch.where(beta_prime != 0, -2 * gammas / beta_prime, 0)

rbar = torch.where(beta_prime != 0, -2 / beta_prime, 0)

for i in range(4):
    for j in range(4):
        if j!=0: continue
        for Nf in range(3,7):
            r_Gamma[i][j][Nf] = get_Gamma0(Nf) / betas[Nf][0]
            if i>0: r_gamma[i][j][Nf] = get_gammaV1(Nf) / betas[Nf][0]
            if i>0: rbar[i][j][Nf] = -betas[Nf][1] / betas[Nf][0]**2

rbar0prime = torch.zeros(7)
for Nf in range(3,7):
    rbar0prime[Nf] = 1 / betas[Nf][0]
