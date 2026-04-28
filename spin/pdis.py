import lhapdf
import numpy as np
import sys

try:
    from Collins_fit.TMDs.Numerical.FBT import FBT
    from Collins_fit.TMDs.tmds import TMDs
    from Collins_fit.kits.share import share
except:
    sys.path.append("../")
    sys.path.append("../..")
    from TMDs.Numerical.FBT import FBT
    from TMDs.tmds import TMDs
    from kits.share import share

import warnings
warnings.filterwarnings("ignore")

class PDIS:

    def __init__(self,params, loops = 1, logs = 2):
        self.scheme = share['scheme']
        self.tmds = TMDs(loops,logs,params)

        self.fbt1 = FBT(1)

    #-- Import u Collins function from tmds.py
    def H1Tuub(self,z,Q,b):
        return self.tmds.H1Tuub(z,Q,b)

    #-- Import d Collins function from tmds.py
    def H1Tddb(self,z,Q,b):
        return self.tmds.H1Tddb(z,Q,b)

    #-- Import u transversity function from tmds.py
    def h1buu(self,x,Q,b):
        return self.tmds.h1buu(x,Q,b)

    #-- Import d Collins function from tmds.py
    def h1bdd(self,x,Q,b):
        return self.tmds.h1bdd(x,Q,b)

    #-- The polarized cross section in b-space (ds/dx dy dz d2PhT)
    def dsigmadxdydzd2PhT_b_SIDIS(self,b,x,z,Q):
        FDD,FUU = self.h1bdd(x,Q,b) ,self.h1buu(x,Q,b)
        DDD,DUU = self.H1Tddb(z,Q,b),self.H1Tuub(z,Q,b)
        als = self.tmds.alphas(Q)
        CF = 4./3.
        if self.scheme == 'MSbar':
            HQ = 1+als*CF/2./np.pi*(-8.+np.pi**2./6.)
        elif self.scheme == 'JCC':
            HQ = 1+als*CF/2./np.pi*(-8.)
        elif self.scheme == 'CSS':
            HQ = 1
        eu2 = 4./9.
        ed2 = 1./9.
        return HQ*(eu2*(FUU*DUU)+ed2*(FDD*DDD))

    #-- The polarized cross section in momentum space (ds/dx dy dz d2PhT)
    def dsigmadxdydzd2PhT_p_SIDIS(self,PhT,x,y,z,Q):
        sigma0  = 2.*np.pi*self.tmds.alphaEM(Q)**2./Q**4.*(1.+(1.-y)**2.)/y
        sigma0 *= 2.0 * (1.0 - y) / (1.0 + ((1.0 - y) ** 2.0))
        FUTb = lambda b: b*b*self.dsigmadxdydzd2PhT_b_SIDIS(b,x,z,Q)
        return sigma0*self.fbt1.fbt(FUTb, PhT/z, 20,Q)

if __name__ == '__main__':
    from Collins_fit.input_params import inputparams

    PDF_name = 'CxFCT14nlo'
    FF_name  = 'CxFDSS14pinlo'
    share['LHAPDF'] = {}
    share['LHAPDF']['PDFs'] = lhapdf.mkPDFs(PDF_name)
    share['LHAPDF']['FFs' ] = lhapdf.mkPDFs(FF_name )

    parameters = []
    parameters.append(inputparams['h1']['KPSY']['N']['uu'])
    parameters.append(inputparams['h1']['KPSY']['a']['uu'])
    parameters.append(inputparams['h1']['KPSY']['b']['uu'])
    parameters.append(inputparams['h1']['KPSY']['N']['dd'])
    parameters.append(inputparams['h1']['KPSY']['a']['dd'])
    parameters.append(inputparams['h1']['KPSY']['b']['dd'])
    parameters.append(inputparams['h1']['KPSY']['N']['ss'])
    parameters.append(inputparams['h1']['KPSY']['a']['ss'])
    parameters.append(inputparams['h1']['KPSY']['b']['ss'])
    parameters.append(inputparams['h1']['KPSY']['N']['sb'])
    parameters.append(inputparams['h1']['KPSY']['a']['sb'])
    parameters.append(inputparams['h1']['KPSY']['b']['sb'])
    parameters.append(inputparams['h1']['KPSY']['N']['db'])
    parameters.append(inputparams['h1']['KPSY']['a']['db'])
    parameters.append(inputparams['h1']['KPSY']['b']['db'])
    parameters.append(inputparams['h1']['KPSY']['N']['ub'])
    parameters.append(inputparams['h1']['KPSY']['a']['ub'])
    parameters.append(inputparams['h1']['KPSY']['b']['ub'])

    parameters.append(inputparams['H3']['KPSY']['N']['fav'])
    parameters.append(inputparams['H3']['KPSY']['N']['unf'])
    parameters.append(inputparams['H3']['KPSY']['a']['fav'])
    parameters.append(inputparams['H3']['KPSY']['a']['unf'])
    parameters.append(inputparams['H3']['KPSY']['b']['fav'])
    parameters.append(inputparams['H3']['KPSY']['N']['unf'])

    distribution = PDIS(parameters)

    ## benchmark with results from Mathematica
    test = lambda b: np.exp(- b * b) * np.sin(b) / (b + 0.3)
    result_0 = distribution.fbt0.fbt(test, 2.0, 20, 0.0)
    result_1 = distribution.fbt1.fbt(test, 2.0, 20, 0.0)
    result_0 *= 2.0 * np.pi
    result_1 *= 2.0 * np.pi
    print(result_0, result_1)

