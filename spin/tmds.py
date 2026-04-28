import lhapdf
import numpy as np
from scipy.integrate import quad
from scipy.special import gamma, digamma, beta
import sys

try:
    from Collins_fit.TMDs.Evolve.anom import ANOM
    from Collins_fit.TMDs.Evolve.alpha import ALPHAS
    from Collins_fit.TMDs.Evolve.kernel_q import dig  as digamma
    from Collins_fit.TMDs.Evolve.kernel_q import trig as trigamma
    from Collins_fit.TMDs.default_params import defaparams
    from Collins_fit.kits.share import share
except:
    sys.path.append("../")
    sys.path.append("../..")
    from TMDs.Evolve.anom import ANOM
    from TMDs.Evolve.alpha import ALPHAS
    from kits.share import share
    #from TMDs.Evolve.kernel_q import dig  as digamma
    #from TMDs.Evolve.kernel_q import trig as trigamma
    from TMDs.default_params import defaparams

import warnings
warnings.filterwarnings("ignore")

class TMDs:

    def __init__(self, loops, logs, params = None):
        self.ffpip = share['LHAPDF']['FFs' ]
        self.pdfp  = share['LHAPDF']['PDFs']
        self.anom = ANOM(loops,logs)
        self.ALFS = ALPHAS(loops)
        self.scheme = share['scheme']

        #-- assign parameters

        if params != None:
            #- transversity
            self.NuuT = params[0]
            self.auuT = params[1]
            self.buuT = params[2]
            self.NddT = params[3]
            self.addT = params[4]
            self.bddT = params[5]
            self.NssT = params[6]
            self.assT = params[7]
            self.bssT = params[8]
            self.NsbT = params[9]
            self.asbT = params[10]
            self.bsbT = params[11]
            self.NdbT = params[12]
            self.adbT = params[13]
            self.bdbT = params[14]
            self.NubT = params[15]
            self.aubT = params[16]
            self.bubT = params[17]

            self.Nfav =     params[18]
            self.Nufv =     params[19]
            self.alphafav = params[20]
            self.alphaufv = params[21]
            self.betafav =  params[22]
            self.betaufv =  params[23]


            #- collinear PDF
            self.Au = defaparams['f1']['MSTW2008']['NLO']['Au']
            self.n1 = defaparams['f1']['MSTW2008']['NLO']['n1']
            self.n2 = defaparams['f1']['MSTW2008']['NLO']['n2']
            self.eu = defaparams['f1']['MSTW2008']['NLO']['eu']
            self.gu = defaparams['f1']['MSTW2008']['NLO']['gu']
            self.Ad = defaparams['f1']['MSTW2008']['NLO']['Ad']
            self.n3 = defaparams['f1']['MSTW2008']['NLO']['n3']
            self.n4 = defaparams['f1']['MSTW2008']['NLO']['n4']
            self.ed = defaparams['f1']['MSTW2008']['NLO']['ed']
            self.gd = defaparams['f1']['MSTW2008']['NLO']['gd']
            self.AS = defaparams['f1']['MSTW2008']['NLO']['AS']
            self.dS = defaparams['f1']['MSTW2008']['NLO']['dS']
            self.nS = defaparams['f1']['MSTW2008']['NLO']['nS']
            self.eS = defaparams['f1']['MSTW2008']['NLO']['eS']
            self.gS = defaparams['f1']['MSTW2008']['NLO']['gS']
            self.AD = defaparams['f1']['MSTW2008']['NLO']['AD']
            self.nD = defaparams['f1']['MSTW2008']['NLO']['nD']
            self.nS = defaparams['f1']['MSTW2008']['NLO']['nS']
            self.gD = defaparams['f1']['MSTW2008']['NLO']['gD']
            self.dD = defaparams['f1']['MSTW2008']['NLO']['dD']
            self.Ap = defaparams['f1']['MSTW2008']['NLO']['Ap']
            self.dS = defaparams['f1']['MSTW2008']['NLO']['dS']
            self.npp= defaparams['f1']['MSTW2008']['NLO']['npp']
            self.eS = defaparams['f1']['MSTW2008']['NLO']['eS']
            self.gS = defaparams['f1']['MSTW2008']['NLO']['gS']
            self.Am = defaparams['f1']['MSTW2008']['NLO']['Am']
            self.dm = defaparams['f1']['MSTW2008']['NLO']['dm']
            self.nm = defaparams['f1']['MSTW2008']['NLO']['nm']
            self.x0 = defaparams['f1']['MSTW2008']['NLO']['x0']

            #- collinear fragmentation
            self.Nup = defaparams['g1']['DSSV']['N']['u+']
            self.aup = defaparams['g1']['DSSV']['a']['u+']
            self.bup = defaparams['g1']['DSSV']['b']['u+']
            self.gup = defaparams['g1']['DSSV']['g']['u+']
            self.nup = defaparams['g1']['DSSV']['e']['u+']
            self.Ndp = defaparams['g1']['DSSV']['N']['d+']
            self.adp = defaparams['g1']['DSSV']['a']['d+']
            self.bdp = defaparams['g1']['DSSV']['b']['d+']
            self.gdp = defaparams['g1']['DSSV']['g']['d+']
            self.ndp = defaparams['g1']['DSSV']['e']['d+']
            self.Nub = defaparams['g1']['DSSV']['N']['ub']
            self.aub = defaparams['g1']['DSSV']['a']['ub']
            self.bub = defaparams['g1']['DSSV']['b']['ub']
            self.gub = defaparams['g1']['DSSV']['g']['ub']
            self.nub = defaparams['g1']['DSSV']['e']['ub']
            self.Ndb = defaparams['g1']['DSSV']['N']['db']
            self.adb = defaparams['g1']['DSSV']['a']['db']
            self.bdb = defaparams['g1']['DSSV']['b']['db']
            self.gdb = defaparams['g1']['DSSV']['g']['db']
            self.ndb = defaparams['g1']['DSSV']['e']['db']
            self.Nss = defaparams['g1']['DSSV']['N']['ss']
            self.ass = defaparams['g1']['DSSV']['a']['ss']
            self.bss = defaparams['g1']['DSSV']['b']['ss']
            self.gss = defaparams['g1']['DSSV']['g']['ss']
            self.nss = defaparams['g1']['DSSV']['e']['ss']


            self.NupDSS = defaparams['D1']['DSS']['N']['u+']
            self.NdpDSS = defaparams['D1']['DSS']['N']['d+']
            self.NubDSS = defaparams['D1']['DSS']['N']['ub']
            self.NddDSS = defaparams['D1']['DSS']['N']['dd']

            self.alphaupDSS = defaparams['D1']['DSS']['a']['u+']
            self.alphadpDSS = defaparams['D1']['DSS']['a']['d+']
            self.alphaubDSS = defaparams['D1']['DSS']['a']['ub']
            self.alphaddDSS = defaparams['D1']['DSS']['a']['dd']

            self.betaupDSS = defaparams['D1']['DSS']['b']['u+']
            self.betadpDSS = defaparams['D1']['DSS']['b']['d+']
            self.betaubDSS = defaparams['D1']['DSS']['b']['ub']
            self.betaddDSS = defaparams['D1']['DSS']['b']['dd']

            self.gammaupDSS = defaparams['D1']['DSS']['g']['u+']
            self.gammadpDSS = defaparams['D1']['DSS']['g']['d+']
            self.gammaubDSS = defaparams['D1']['DSS']['g']['ub']
            self.gammaddDSS = defaparams['D1']['DSS']['g']['dd']

            self.deltaupDSS = defaparams['D1']['DSS']['d']['u+']
            self.deltadpDSS = defaparams['D1']['DSS']['d']['d+']
            self.deltaubDSS = defaparams['D1']['DSS']['d']['ub']
            self.deltaddDSS = defaparams['D1']['DSS']['d']['dd']

    #-- define collinear fragmentation function (FF)
    def CxFF(self,z,Q):
        D = self.ffpip.xfxQ( 1,z,Q)/z
        U = self.ffpip.xfxQ( 2,z,Q)/z
        S = self.ffpip.xfxQ( 3,z,Q)/z
        SB= self.ffpip.xfxQ(-3,z,Q)/z
        UB= self.ffpip.xfxQ(-2,z,Q)/z
        DB= self.ffpip.xfxQ(-1,z,Q)/z
        return 0,D,U,S,SB,UB,DB

    #-- define collinear parton distribution function (PDF)
    def CxPDF(self,x,Q):
        D = self.pdfp.xfxQ( 1,x,Q)/x
        U = self.pdfp.xfxQ( 2,x,Q)/x
        S = self.pdfp.xfxQ( 3,x,Q)/x
        SB= self.pdfp.xfxQ(-3,x,Q)/x
        UB= self.pdfp.xfxQ(-2,x,Q)/x
        DB= self.pdfp.xfxQ(-1,x,Q)/x
        return 0,D,U,S,SB,UB,DB

    #-- define unpolarized TMD PDF
    def TMDPDF(self,x,Q,b):
        bmax = 1.5
        bstar = b/np.sqrt(1.+(b/bmax)**2.)
        c0 = 1.122919
        Q0 = c0/bstar
        Revo = self.anom.anomq(b,Q0,Q,Q0,Q)
        ggi,DDi,UUi,SSi,SBi,UBi,DBi = self.CxPDF(x,Q0)
        ktw = 0.424
        kappa2 = 0.84
        Qini = np.sqrt(2.4)
        kappa1 = ktw/4.
        FNP = np.exp( -(kappa1)*b*b-kappa2/2.*np.log(b/bstar)*np.log(Q/Qini) )
        DD= DDi*FNP*Revo
        UU= UUi*FNP*Revo
        SS= SSi*FNP*Revo
        SB= SBi*FNP*Revo
        UB= UBi*FNP*Revo
        DB= DBi*FNP*Revo
        return ggi,DD,UU,SS,SB,UB,DB

    #-- define unpolarized TMD FF
    def TMDFF(self,z,Q,b):
        bmax = 1.5
        bstar = b/np.sqrt(1.+(b/bmax)**2.)
        c0 = 1.122919
        Q0 = c0/bstar
        Revo = self.anom.anomq(b,Q0,Q,Q0,Q)
        ggi,DDi,UUi,SSi,SBi,UBi,DBi = self.CxFF(z,Q0)
        ktw = 0.168
        kappa2 = 0.84
        Qini = np.sqrt(2.4)
        kappa1 = ktw/4./z/z
        FNP = np.exp( -(kappa1)*b*b-kappa2/2.*np.log(b/bstar)*np.log(Q/Qini) )
        DD= DDi*FNP*Revo
        UU= UUi*FNP*Revo
        SS= SSi*FNP*Revo
        SB= SBi*FNP*Revo
        UB= UBi*FNP*Revo
        DB= DBi*FNP*Revo
        return ggi,DD,UU,SS,SB,UB,DB

    #-- Strong coupling constant
    def alphas(self,Q):
        return self.ALFS.alfs(Q)

    #-- EM coupling constant
    def alphaEM(self,Q):
        return self.ALFS.alfE(Q)

    def TransvuuQ0(self,n):
        pref = (self.auuT+self.buuT)**(self.auuT+self.buuT)/self.auuT**self.auuT/self.buuT**self.buuT
        val =  \
        -(self.Nub*self.NuuT*gamma(-1 + n + self.aub + self.auuT)*gamma(1 + self.bub + self.buuT))/(2.*gamma(n + self.aub + self.auuT + self.bub + self.buuT)) - \
        (self.Nub*self.NuuT*self.gub*gamma(-0.5 + n + self.aub + self.auuT)*gamma(1 + self.bub + self.buuT))/(2.*gamma(0.5 + n + self.aub + self.auuT + self.bub + self.buuT)) - \
        (self.Nub*self.NuuT*self.nub*gamma(n + self.aub + self.auuT)*gamma(1 + self.bub + self.buuT))/(2.*gamma(1 + n + self.aub + self.auuT + self.bub + self.buuT)) + \
        (self.Nup*self.NuuT*gamma(-1 + n + self.aup + self.auuT)*gamma(1 + self.bup + self.buuT))/(2.*gamma(n + self.aup + self.auuT + self.bup + self.buuT)) + \
        (self.Nup*self.NuuT*self.gup*gamma(-0.5 + n + self.aup + self.auuT)*gamma(1 + self.bup + self.buuT))/(2.*gamma(0.5 + n + self.aup + self.auuT + self.bup + self.buuT)) + \
        (self.Nup*self.NuuT*self.nup*gamma(n + self.aup + self.auuT)*gamma(1 + self.bup + self.buuT))/(2.*gamma(1 + n + self.aup + self.auuT + self.bup + self.buuT)) + \
        (self.Au*self.NuuT*gamma(-1 + n + self.auuT + self.n1)*gamma(1 + self.buuT + self.n2))/(2.*gamma(n + self.auuT + self.buuT + self.n1 + self.n2)) + \
        (self.Au*self.NuuT*self.eu*gamma(-0.5 + n + self.auuT + self.n1)*gamma(1 + self.buuT + self.n2))/(2.*gamma(0.5 + n + self.auuT + self.buuT + self.n1 + self.n2)) + \
        (self.Au*self.NuuT*self.gu*gamma(n + self.auuT + self.n1)*gamma(1 + self.buuT + self.n2))/(2.*gamma(1 + n + self.auuT + self.buuT + self.n1 + self.n2)) - \
        (self.Ap*self.NuuT*gamma(-1 + n + self.auuT + self.dS)*gamma(1 + self.buuT + self.npp))/(8.*gamma(n + self.auuT + self.buuT + self.dS + self.npp)) - \
        (self.Ap*self.NuuT*self.eS*gamma(-0.5 + n + self.auuT + self.dS)*gamma(1 + self.buuT + self.npp))/(8.*gamma(0.5 + n + self.auuT + self.buuT + self.dS + self.npp)) - \
        (self.Ap*self.NuuT*self.gS*gamma(n + self.auuT + self.dS)*gamma(1 + self.buuT + self.npp))/(8.*gamma(1 + n + self.auuT + self.buuT + self.dS + self.npp)) + \
        (self.AS*self.NuuT*gamma(-1 + n + self.auuT + self.dS)*gamma(1 + self.buuT + self.nS))/(8.*gamma(n + self.auuT + self.buuT + self.dS + self.nS)) + \
        (self.AS*self.NuuT*self.eS*gamma(-0.5 + n + self.auuT + self.dS)*gamma(1 + self.buuT + self.nS))/(8.*gamma(0.5 + n + self.auuT + self.buuT + self.dS + self.nS)) + \
        (self.AS*self.NuuT*self.gS*gamma(n + self.auuT + self.dS)*gamma(1 + self.buuT + self.nS))/(8.*gamma(1 + n + self.auuT + self.buuT + self.dS + self.nS)) - \
        (self.AD*self.NuuT*gamma(1 + self.buuT + self.nS)*gamma(-1 + n + self.auuT + self.nD))/(4.*gamma(n + self.auuT + self.buuT + self.nS + self.nD)) - \
        (self.AD*self.NuuT*self.gD*gamma(1 + self.buuT + self.nS)*gamma(-0.5 + n + self.auuT + self.nD))/(4.*gamma(0.5 + n + self.auuT + self.buuT + self.nS + self.nD)) + \
        (self.AD*self.NuuT*gamma(1 + self.buuT + self.nS)*gamma(n + self.auuT + self.nD))/(2.*gamma(1 + n + self.auuT + self.buuT + self.nS + self.nD)) - \
        (self.AD*self.NuuT*self.dD*gamma(1 + self.buuT + self.nS)*gamma(n + self.auuT + self.nD))/(4.*gamma(1 + n + self.auuT + self.buuT + self.nS + self.nD)) + \
        (self.AD*self.NuuT*self.gD*gamma(1 + self.buuT + self.nS)*gamma(0.5 + n + self.auuT + self.nD))/(2.*gamma(1.5 + n + self.auuT + self.buuT + self.nS + self.nD)) - \
        (self.AD*self.NuuT*gamma(1 + self.buuT + self.nS)*gamma(1 + n + self.auuT + self.nD))/(4.*gamma(2 + n + self.auuT + self.buuT + self.nS + self.nD)) + \
        (self.AD*self.NuuT*self.dD*gamma(1 + self.buuT + self.nS)*gamma(1 + n + self.auuT + self.nD))/(2.*gamma(2 + n + self.auuT + self.buuT + self.nS + self.nD)) - \
        (self.AD*self.NuuT*self.gD*gamma(1 + self.buuT + self.nS)*gamma(1.5 + n + self.auuT + self.nD))/(4.*gamma(2.5 + n + self.auuT + self.buuT + self.nS + self.nD)) - \
        (self.AD*self.NuuT*self.dD*gamma(1 + self.buuT + self.nS)*gamma(2 + n + self.auuT + self.nD))/(4.*gamma(3 + n + self.auuT + self.buuT + self.nS + self.nD))
        return pref*val

    def TransvddQ0(self,n):
        pref = (self.addT+self.bddT)**(self.addT+self.bddT)/self.addT**self.addT/self.bddT**self.bddT
        val = \
        -(self.Ndb*self.NddT*gamma(-1 + n + self.adb + self.addT)*gamma(1 + self.bdb + self.bddT))/(2.*gamma(n + self.adb + self.addT + self.bdb + self.bddT)) - \
        (self.Ndb*self.NddT*self.gdb*gamma(-0.5 + n + self.adb + self.addT)*gamma(1 + self.bdb + self.bddT))/(2.*gamma(0.5 + n + self.adb + self.addT + self.bdb + self.bddT)) - \
        (self.Ndb*self.NddT*self.Ndb*gamma(n + self.adb + self.addT)*gamma(1 + self.bdb + self.bddT))/(2.*gamma(1 + n + self.adb + self.addT + self.bdb + self.bddT)) + \
        (self.NddT*self.Ndp*gamma(-1 + n + self.addT + self.adp)*gamma(1 + self.bddT + self.bdp))/(2.*gamma(n + self.addT + self.adp + self.bddT + self.bdp)) + \
        (self.NddT*self.Ndp*self.gdp*gamma(-0.5 + n + self.addT + self.adp)*gamma(1 + self.bddT + self.bdp))/(2.*gamma(0.5 + n + self.addT + self.adp + self.bddT + self.bdp)) + \
        (self.NddT*self.Ndp*self.Ndp*gamma(n + self.addT + self.adp)*gamma(1 + self.bddT + self.bdp))/(2.*gamma(1 + n + self.addT + self.adp + self.bddT + self.bdp)) + \
        (self.Ad*self.NddT*gamma(-1 + n + self.addT + self.n3)*gamma(1 + self.bddT + self.n4))/(2.*gamma(n + self.addT + self.bddT + self.n3 + self.n4)) + \
        (self.Ad*self.NddT*self.ed*gamma(-0.5 + n + self.addT + self.n3)*gamma(1 + self.bddT + self.n4))/(2.*gamma(0.5 + n + self.addT + self.bddT + self.n3 + self.n4)) + \
        (self.Ad*self.NddT*self.gd*gamma(n + self.addT + self.n3)*gamma(1 + self.bddT + self.n4))/(2.*gamma(1 + n + self.addT + self.bddT + self.n3 + self.n4)) - \
        (self.Ap*self.NddT*gamma(-1 + n + self.addT + self.dS)*gamma(1 + self.bddT + self.npp))/(8.*gamma(n + self.addT + self.bddT + self.dS + self.npp)) - \
        (self.Ap*self.NddT*self.eS*gamma(-0.5 + n + self.addT + self.dS)*gamma(1 + self.bddT + self.npp))/(8.*gamma(0.5 + n + self.addT + self.bddT + self.dS + self.npp)) - \
        (self.Ap*self.NddT*self.gS*gamma(n + self.addT + self.dS)*gamma(1 + self.bddT + self.npp))/(8.*gamma(1 + n + self.addT + self.bddT + self.dS + self.npp)) + \
        (self.AS*self.NddT*gamma(-1 + n + self.addT + self.dS)*gamma(1 + self.bddT + self.nS))/(8.*gamma(n + self.addT + self.bddT + self.dS + self.nS)) + \
        (self.AS*self.NddT*self.eS*gamma(-0.5 + n + self.addT + self.dS)*gamma(1 + self.bddT + self.nS))/(8.*gamma(0.5 + n + self.addT + self.bddT + self.dS + self.nS)) + \
        (self.AS*self.NddT*self.gS*gamma(n + self.addT + self.dS)*gamma(1 + self.bddT + self.nS))/(8.*gamma(1 + n + self.addT + self.bddT + self.dS + self.nS)) + \
        (self.AD*self.NddT*gamma(1 + self.bddT + self.nS)*gamma(-1 + n + self.addT + self.nD))/(4.*gamma(n + self.addT + self.bddT + self.nS + self.nD)) + \
        (self.AD*self.NddT*self.gD*gamma(1 + self.bddT + self.nS)*gamma(-0.5 + n + self.addT + self.nD))/(4.*gamma(0.5 + n + self.addT + self.bddT + self.nS + self.nD)) - \
        (self.AD*self.NddT*gamma(1 + self.bddT + self.nS)*gamma(n + self.addT + self.nD))/(2.*gamma(1 + n + self.addT + self.bddT + self.nS + self.nD)) + \
        (self.AD*self.NddT*self.dD*gamma(1 + self.bddT + self.nS)*gamma(n + self.addT + self.nD))/(4.*gamma(1 + n + self.addT + self.bddT + self.nS + self.nD)) - \
        (self.AD*self.NddT*self.gD*gamma(1 + self.bddT + self.nS)*gamma(0.5 + n + self.addT + self.nD))/(2.*gamma(1.5 + n + self.addT + self.bddT + self.nS + self.nD)) + \
        (self.AD*self.NddT*gamma(1 + self.bddT + self.nS)*gamma(1 + n + self.addT + self.nD))/(4.*gamma(2 + n + self.addT + self.bddT + self.nS + self.nD)) - \
        (self.AD*self.NddT*self.dD*gamma(1 + self.bddT + self.nS)*gamma(1 + n + self.addT + self.nD))/(2.*gamma(2 + n + self.addT + self.bddT + self.nS + self.nD)) + \
        (self.AD*self.NddT*self.gD*gamma(1 + self.bddT + self.nS)*gamma(1.5 + n + self.addT + self.nD))/(4.*gamma(2.5 + n + self.addT + self.bddT + self.nS + self.nD)) + \
        (self.AD*self.NddT*self.dD*gamma(1 + self.bddT + self.nS)*gamma(2 + n + self.addT + self.nD))/(4.*gamma(3 + n + self.addT + self.bddT + self.nS + self.nD))
        return pref*val

    def Pqqh1(self,n):
        ## splitting kernel in Mellin space
        ## from equation 69 in 10.1103/PhysRevD.93.014009
        return 4./3.*(-2.*(digamma(n)+1./n+np.euler_gamma)+1.5)

    def DGLAP(self,n,Q):
        Q0 = 1.
        mc = 1.275
        mb = 4.5
        if Q<= mc:
            nf = 3.
            b0 = 11.-2./3.*nf
            evo  = (self.alphas(Q)/self.alphas(Q0))**(-self.Pqqh1(n)/b0)
        elif Q<= mb:
            nf = 3.
            b0 = 11.-2./3.*nf
            evoc = (self.alphas(mc)/self.alphas(Q0))**(-self.Pqqh1(n)/b0)
            nf = 4.
            b0 = 11.-2./3.*nf
            evob = (self.alphas(Q )/self.alphas(mc))**(-self.Pqqh1(n)/b0)
            evo  = evoc*evob
        elif Q > mb:
            nf = 3.
            b0 = 11.-2./3.*nf
            evoc = (self.alphas(mc)/self.alphas(Q0))**(-self.Pqqh1(n)/b0)
            nf = 4.
            b0 = 11.-2./3.*nf
            evob = (self.alphas(mb)/self.alphas(mc))**(-self.Pqqh1(n)/b0)
            nf = 5.
            b0 = 11.-2./3.*nf
            evoQ = (self.alphas(Q )/self.alphas(mb))**(-self.Pqqh1(n)/b0)
            evo  = evoc*evob*evoQ
        return evo

    #-- evolved Transversity
    def TransvuuQ(self,n,Q):
        ## h_1 from equation 66 in 10.1103/PhysRevD.93.014009
        h_mu_b = self.TransvuuQ0(n)

        ## splitting kernel from equation 69 in 10.1103/PhysRevD.93.014009
        evo = self.DGLAP(n,Q)

        ## matching coefficient from equation 67 in 10.1103/PhysRevD.93.014009
        if self.scheme == 'MSbar':
            coefficient = 1. - self.alphas(Q) * (4.0 / 3.0) * (np.pi * np.pi / 6.0) / (4.0 * np.pi)
        elif self.scheme == 'JCC':
            coefficient = 1.
        elif self.scheme == 'CSS':
            coefficient = 1. - self.alphas(Q) * (4.0 / 3.0) * 8.0 / (4.0 * np.pi)

        return h_mu_b * evo * coefficient

    def TransvddQ(self,n,Q):
        h_mu_b = self.TransvddQ0(n)
        evo = self.DGLAP(n,Q)

        ## matching coefficient from equation 67 in 10.1103/PhysRevD.93.014009
        if self.scheme == 'MSbar':
            coefficient = 1. - self.alphas(Q) * (4.0 / 3.0) * (np.pi * np.pi / 6.0) / (4.0 * np.pi)
        elif self.scheme == 'JCC':
            coefficient = 1.
        elif self.scheme == 'CSS':
            coefficient = 1. - self.alphas(Q) * (4.0 / 3.0) * 8.0 / (4.0 * np.pi)
        return h_mu_b * evo * coefficient

    def TransvssQ(self,n,Q):
        h_mu_b = self.TransvssQ0(n)
        evo = self.DGLAP(n,Q)

        ## matching coefficient from equation 67 in 10.1103/PhysRevD.93.014009
        if self.scheme == 'MSbar':
            coefficient = 1. - self.alphas(Q) * (4.0 / 3.0) * (np.pi * np.pi / 6.0) / (4.0 * np.pi)
        elif self.scheme == 'JCC':
            coefficient = 1.
        elif self.scheme == 'CSS':
            coefficient = 1. - self.alphas(Q) * (4.0 / 3.0) * 8.0 / (4.0 * np.pi)
        return h_mu_b * evo * coefficient

    def TransvubQ(self,n,Q):
        h_mu_b = self.TransvubQ0(n)
        evo = self.DGLAP(n,Q)

        ## matching coefficient from equation 67 in 10.1103/PhysRevD.93.014009
        if self.scheme == 'MSbar':
            coefficient = 1. - self.alphas(Q) * (4.0 / 3.0) * (np.pi * np.pi / 6.0) / (4.0 * np.pi)
        elif self.scheme == 'JCC':
            coefficient = 1.
        elif self.scheme == 'CSS':
            coefficient = 1. - self.alphas(Q) * (4.0 / 3.0) * 8.0 / (4.0 * np.pi)
        return h_mu_b * evo * coefficient

    def TransvdbQ(self,n,Q):
        h_mu_b = self.TransvdbQ0(n)
        evo = self.DGLAP(n,Q)

        ## matching coefficient from equation 67 in 10.1103/PhysRevD.93.014009
        if self.scheme == 'MSbar':
            coefficient = 1. - self.alphas(Q) * (4.0 / 3.0) * (np.pi * np.pi / 6.0) / (4.0 * np.pi)
        elif self.scheme == 'JCC':
            coefficient = 1.
        elif self.scheme == 'CSS':
            coefficient = 1. - self.alphas(Q) * (4.0 / 3.0) * 8.0 / (4.0 * np.pi)
        return h_mu_b * evo * coefficient

    def TransvsbQ(self,n,Q):
        h_mu_b = self.TransvsbQ0(n)
        evo = self.DGLAP(n,Q)

        ## matching coefficient from equation 67 in 10.1103/PhysRevD.93.014009
        if self.scheme == 'MSbar':
            coefficient = 1. - self.alphas(Q) * (4.0 / 3.0) * (np.pi * np.pi / 6.0) / (4.0 * np.pi)
        elif self.scheme == 'JCC':
            coefficient = 1.
        elif self.scheme == 'CSS':
            coefficient = 1. - self.alphas(Q) * (4.0 / 3.0) * 8.0 / (4.0 * np.pi)
        return h_mu_b * evo * coefficient

    #--- transversity in x and Q
    def TransvuuxQ(self,x,Q):
        phi = 3.*np.pi/4.
        c = 2.
        integrand = lambda z: 1./np.pi*np.imag(np.exp(1j*phi)*x**(-c-z*np.exp(1j*phi))*self.TransvuuQ(c+z*np.exp(1j*phi),Q))
        return quad(integrand,0.,20.,epsabs = 0.,epsrel = 0.05)[0]

    def TransvddxQ(self,x,Q):
        phi = 3.*np.pi/4.
        c = 2.
        integrand = lambda z: 1./np.pi*np.imag(np.exp(1j*phi)*x**(-c-z*np.exp(1j*phi))*self.TransvddQ(c+z*np.exp(1j*phi),Q))
        return quad(integrand,0.,20.,epsabs = 0.,epsrel = 0.05)[0]

    #--- transversity TMD in b-space
    def h1buu(self,x,Q,b):
        bmax = 1.5
        bstar = b/np.sqrt(1.+(b/bmax)**2.)
        c0 = 1.122919
        Q0 = c0/bstar
        Revo = self.anom.anomq(b,Q0,Q,Q0,Q)
        UUi = self.TransvuuxQ(x,Q0)
        ktw = 0.424
        kappa2 = 0.84
        Qini = np.sqrt(2.4)
        kappa1 = ktw/4.
        FNP = np.exp( -(kappa1)*b*b-kappa2/2.*np.log(b/bstar)*np.log(Q/Qini) )
        UU= UUi*FNP*Revo
        return UU

    def h1bdd(self,x,Q,b):
        bmax = 1.5
        bstar = b/np.sqrt(1.+(b/bmax)**2.)
        c0 = 1.122919
        Q0 = c0/bstar
        Revo = self.anom.anomq(b,Q0,Q,Q0,Q)
        DDi = self.TransvddxQ(x,Q0)
        ktw = 0.424
        kappa2 = 0.84
        Qini = np.sqrt(2.4)
        kappa1 = ktw/4.
        FNP = np.exp( -(kappa1)*b*b-kappa2/2.*np.log(b/bstar)*np.log(Q/Qini) )
        DD= DDi*FNP*Revo
        return DD

    #--- Mellin transform of our parameterization for Collins FF
    def ColnsCDKTddQ0(self,n):
        return \
        (self.NddDSS*self.Nufv * \
        gamma(self.alphaddDSS + n + self.alphaufv) * \
        (self.gammaddDSS * \
        gamma(1 + self.betaddDSS + self.betaufv + self.deltaddDSS) / \
        gamma(1 + self.alphaddDSS + n + self.alphaufv + self.betaddDSS + self.betaufv + self.deltaddDSS) + \
        gamma(1 + self.betaddDSS + self.betaufv) / \
        gamma(1 + self.alphaddDSS + n + self.alphaufv + self.betaddDSS + self.betaufv))) / \
        (beta(2 + self.alphaddDSS,1 + self.betaddDSS) + \
        self.gammaddDSS*beta(2 + self.alphaddDSS,1 + self.betaddDSS + self.deltaddDSS))

    def ColnsCDKTuuQ0(self,n):
        return \
        self.Nfav*(-((self.NubDSS*gamma(self.alphaubDSS + n + self.alphafav)* \
        (self.gammaubDSS * \
        gamma(1 + self.betafav + self.betaubDSS + self.deltaubDSS) / \
        gamma(1 + self.alphaubDSS + n + self.alphafav + self.betafav + self.betaubDSS + self.deltaubDSS) + \
        gamma(1 + self.betafav + self.betaubDSS) / \
        gamma(1 + self.alphaubDSS + n + self.alphafav + self.betafav + self.betaubDSS))) / \
        (beta(2 + self.alphaubDSS,1 + self.betaubDSS) +  \
        self.gammaubDSS*beta(2 + self.alphaubDSS,1 + self.betaubDSS + self.deltaubDSS))) \
        + \
        (self.NupDSS*gamma(self.alphaupDSS + n + self.alphafav)* \
        (self.gammaupDSS * \
        gamma(1 + self.betafav + self.betaupDSS + self.deltaupDSS) / \
        gamma(1 + self.alphaupDSS + n + self.alphafav + self.betafav + self.betaupDSS + self.deltaupDSS) + \
        gamma(1 + self.betafav + self.betaupDSS) / \
        gamma(1 + self.alphaupDSS + n + self.alphafav + self.betafav + self.betaupDSS))) / \
        (beta(2 + self.alphaupDSS,1 + self.betaupDSS) +  \
        self.gammaupDSS*beta(2 + self.alphaupDSS,1 + self.betaupDSS + self.deltaupDSS)))

    #--- Mellin transform of our parameterization for Collins FF with Q dependence
    def H1TuuQ(self,n,Q):
        ## H^(3) from equation 66 in 10.1103/PhysRevD.93.014009
        h_mu_b = self.ColnsCDKTuuQ0(n)
        ## splitting kernel from equation 71 in 10.1103/PhysRevD.93.014009
        evo = self.DGLAP(n,Q)

        CF =  4./3.
        As = self.alphas(Q)
        ## matching coefficient from equation 68 in 10.1103/PhysRevD.93.014009
        if self.scheme == 'MSbar':
            coefficient = 1.+As/np.pi*CF*(1/(n**2.0) + 1/((n+1.0)**2.0)-2*trigamma(n))-As*CF*(np.pi * np.pi / 6.0)/(4.0 * np.pi)
        elif self.scheme == 'JCC':
            coefficient = 1.+As/np.pi*CF*(1/(n**2.0) + 1/((n+1.0)**2.0)-2*trigamma(n))
        elif self.scheme == 'CSS':
            coefficient = 1.+As/np.pi*CF*(1/(n**2.0) + 1/((n+1.0)**2.0)-2*trigamma(n))-As*CF*8.0/(4.0 * np.pi)
        return h_mu_b * evo * coefficient

    def H1TddQ(self,n,Q):
        h_mu_b = self.ColnsCDKTddQ0(n)
        evo = self.DGLAP(n,Q)

        CF =  4./3.
        As = self.alphas(Q)
        ## matching coefficient from equation 68 in 10.1103/PhysRevD.93.014009
        if self.scheme == 'MSbar':
            coefficient = 1.+As/np.pi*CF*(1/(n**2.0) + 1/((n+1.0)**2.0)-2*trigamma(n))-As*CF*(np.pi * np.pi / 6.0)/(4.0 * np.pi)
        elif self.scheme == 'JCC':
            coefficient = 1.+As/np.pi*CF*(1/(n**2.0) + 1/((n+1.0)**2.0)-2*trigamma(n))
        elif self.scheme == 'CSS':
            coefficient = 1.+As/np.pi*CF*(1/(n**2.0) + 1/((n+1.0)**2.0)-2*trigamma(n))-As*CF*8.0/(4.0 * np.pi)
        return h_mu_b * evo * coefficient

    #--- Collins FF with x and Q dependence
    def H1TuuxQ(self,x,Q):
        phi = 3.*np.pi/4.
        c = 3.
        integrand = lambda z: 1./np.pi*np.imag(np.exp(1j*phi)*x**(-c-z*np.exp(1j*phi))*self.H1TuuQ(c+z*np.exp(1j*phi),Q))
        return quad(integrand,0.,20.,epsabs = 0.,epsrel = 0.05)[0]

    def H1TddxQ(self,x,Q):
        phi = 3.*np.pi/4.
        c = 3.
        integrand = lambda z: 1./np.pi*np.imag(np.exp(1j*phi)*x**(-c-z*np.exp(1j*phi))*self.H1TddQ(c+z*np.exp(1j*phi),Q))
        return quad(integrand,0.,20.,epsabs = 0.,epsrel = 0.05)[0]

    #Collins TMD FF
    def H1Tuub(self,z,Q,b):
        bmax = 1.5
        bstar = b/np.sqrt(1.+(b/bmax)**2.)
        c0 = 1.122919
        Q0 = c0/bstar
        Revo = self.anom.anomq(b,Q0,Q,Q0,Q)
        UUi = self.H1TuuxQ(z,Q0)
        ptw = (0.042-0.0236)*4
        kappa2 = 0.84
        Qini = np.sqrt(2.4)
        kappa1 = ptw/4.
        FNP = np.exp( -(kappa1)*b*b/z/z-kappa2/2.*np.log(b/bstar)*np.log(Q/Qini) )
        UU= UUi*FNP*Revo/z/z
        return UU

    def H1Tddb(self,z,Q,b):
        bmax = 1.5
        bstar = b/np.sqrt(1.+(b/bmax)**2.)
        c0 = 1.122919
        Q0 = c0/bstar
        Revo = self.anom.anomq(b,Q0,Q,Q0,Q)
        DDi = self.H1TddxQ(z,Q0)
        ptw = (0.042-0.0236)*4
        kappa2 = 0.84
        Qini = np.sqrt(2.4)
        kappa1 = ptw/4.
        FNP = np.exp( -(kappa1)*b*b/z/z-kappa2/2.*np.log(b/bstar)*np.log(Q/Qini) )
        DD= DDi*FNP*Revo/z/z
        return DD

