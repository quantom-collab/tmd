# map24fnp.py

import math


class MAP24_FD:
    """
    Python analogue of the C++ class NangaParbat::MAP24_FD.
    It provides:
      - A parameter list self._pars
      - The Evaluate(x, b, zeta, ifunc, fl) method
      - The LatexFormula(), GetParameterNames(), and GetDescription() methods
    """

    def __init__(self):
        """
        The constructor mirrors the C++ constructor:
          MAP24_FD(): Parameterisation{"MAP24_FD", 3, <long list of 0.1, etc.>}
        We store the parameters in a Python list self._pars.
        """
        self._name = "MAP24_FD"
        self._ifunc = 3  
        
        # Here is the long list of default parameters from your C++ code:
        self._pars = [
                    0.2,      0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,  # (indices 0..10)
                    0.1,      0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,       # (11..20)
                    0.1,      0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,       # (21..30)
                    0.1,      0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,       # (31..40)
                    0.1,      0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,       # (41..50)
                    0.1,      0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,       # (51..60)
                    0.1,      0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,       # (61..70)
                    0.1,      0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,       # (71..80)
                    0.1,      0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,       # (81..90)
                    0.1,      0.1, 0.1, 0.1, 0.1, 0.1,                           # (91..96 or so)
                ]

        # The "Q0^2 = 1" in the C++ code
        self._Q02 = 1.0

    def Evaluate(self, x, b, zeta, ifunc, fl):
        """
        Python translation of the entire 'Evaluate(...)' method in MAP24_FD.h.
        - x     in [0,1] (the partonic momentum fraction or hadronic fraction)
        - b     the transverse impact parameter
        - zeta  scale^2 (like Q^2), etc.
        - ifunc which functional piece (0 -> TMD PDFs, 2 -> TMD FF pion, etc.)
        - fl    flavor ID, e.g. 1 => d-quark, 2 => u-quark, etc.
        """
        if ifunc < 0 or ifunc > 3:
            raise RuntimeError("[MAP24_FD::Evaluate]: function index out of range.")

        # If x >= 1, the distribution is zero
        if x >= 1.0:
            return 0.0

        # Common definitions
        g2 = self._pars[0]
        b2 = b * b
        lnz = math.log(zeta / self._Q02)
        # NPevol = exp( - g2^2 * b^2 * lnz / 4 )
        NPevol = math.exp(-(g2**2) * b2 * lnz / 4.0)

        xhat = 0.1
        zhat = 0.5

        # ----------------------
        # TMD PDFs: ifunc == 0 or ifunc == 1
        # ----------------------
        if ifunc == 0 or ifunc == 1:
            
            # Example: fl == 1 => d-quark
            if fl == 1:
                # parse out parameters as in the c++ code
                N1d = self._pars[1]
                N2d = self._pars[2]
                N3d = self._pars[3]
                alpha1d = self._pars[4]
                alpha2d = self._pars[5]
                alpha3d = self._pars[6]
                sigma1d = self._pars[7]
                sigma2d = self._pars[8]
                sigma3d = self._pars[8]  # same as c++
                lambda1d = self._pars[9]
                lambda2d = self._pars[10]

                g1d = (
                    N1d * (x / xhat) ** sigma1d * ((1 - x) / (1 - xhat)) ** (alpha1d**2)
                )
                g2d = (
                    N2d * (x / xhat) ** sigma2d * ((1 - x) / (1 - xhat)) ** (alpha2d**2)
                )
                g3d = (
                    N3d * (x / xhat) ** sigma3d * ((1 - x) / (1 - xhat)) ** (alpha3d**2)
                )

                term = (
                    g1d * math.exp(-g1d * (b / 2.0) ** 2)
                    + (lambda1d**2)
                    * (g2d**2)
                    * (1 - g2d * (b / 2.0) ** 2)
                    * math.exp(-g2d * (b / 2.0) ** 2)
                    + g3d * (lambda2d**2) * math.exp(-g3d * (b / 2.0) ** 2)
                )
                denom = g1d + (lambda1d**2) * (g2d**2) + g3d * (lambda2d**2)

                return NPevol * (term / denom)

            elif fl == -1:
                # Anti-d
                N1db = self._pars[11]
                N2db = self._pars[12]
                N3db = self._pars[13]
                alpha1db = self._pars[14]
                alpha2db = self._pars[15]
                alpha3db = self._pars[16]
                sigma1db = self._pars[17]
                sigma2db = self._pars[18]
                sigma3db = self._pars[18]  # repeated
                lambda1db = self._pars[19]
                lambda2db = self._pars[20]

                g1db = (
                    N1db
                    * (x / xhat) ** sigma1db
                    * ((1 - x) / (1 - xhat)) ** (alpha1db**2)
                )
                g2db = (
                    N2db
                    * (x / xhat) ** sigma2db
                    * ((1 - x) / (1 - xhat)) ** (alpha2db**2)
                )
                g3db = (
                    N3db
                    * (x / xhat) ** sigma3db
                    * ((1 - x) / (1 - xhat)) ** (alpha3db**2)
                )

                term = (
                    g1db * math.exp(-g1db * (b / 2.0) ** 2)
                    + (lambda1db**2)
                    * (g2db**2)
                    * (1 - g2db * (b / 2.0) ** 2)
                    * math.exp(-g2db * (b / 2.0) ** 2)
                    + g3db * (lambda2db**2) * math.exp(-g3db * (b / 2.0) ** 2)
                )
                denom = g1db + (lambda1db**2) * (g2db**2) + g3db * (lambda2db**2)

                return NPevol * (term / denom)

            elif fl == 2:
                # u-quark
                N1u = self._pars[21]
                N2u = self._pars[22]
                N3u = self._pars[23]
                alpha1u = self._pars[24]
                alpha2u = self._pars[25]
                alpha3u = self._pars[26]
                sigma1u = self._pars[27]
                sigma2u = self._pars[28]
                sigma3u = self._pars[28]  # repeated
                lambda1u = self._pars[29]
                lambda2u = self._pars[30]

                g1u = (
                    N1u * (x / xhat) ** sigma1u * ((1 - x) / (1 - xhat)) ** (alpha1u**2)
                )
                g2u = (
                    N2u * (x / xhat) ** sigma2u * ((1 - x) / (1 - xhat)) ** (alpha2u**2)
                )
                g3u = (
                    N3u * (x / xhat) ** sigma3u * ((1 - x) / (1 - xhat)) ** (alpha3u**2)
                )

                term = (
                    g1u * math.exp(-g1u * (b / 2.0) ** 2)
                    + (lambda1u**2)
                    * (g2u**2)
                    * (1 - g2u * (b / 2.0) ** 2)
                    * math.exp(-g2u * (b / 2.0) ** 2)
                    + g3u * (lambda2u**2) * math.exp(-g3u * (b / 2.0) ** 2)
                )
                denom = g1u + (lambda1u**2) * (g2u**2) + g3u * (lambda2u**2)

                return NPevol * (term / denom)

            elif fl == -2:
                # anti-u
                N1ub = self._pars[31]
                N2ub = self._pars[32]
                N3ub = self._pars[33]
                alpha1ub = self._pars[34]
                alpha2ub = self._pars[35]
                alpha3ub = self._pars[36]
                sigma1ub = self._pars[37]
                sigma2ub = self._pars[38]
                sigma3ub = self._pars[38]
                lambda1ub = self._pars[39]
                lambda2ub = self._pars[40]

                g1ub = (
                    N1ub
                    * (x / xhat) ** sigma1ub
                    * ((1 - x) / (1 - xhat)) ** (alpha1ub**2)
                )
                g2ub = (
                    N2ub
                    * (x / xhat) ** sigma2ub
                    * ((1 - x) / (1 - xhat)) ** (alpha2ub**2)
                )
                g3ub = (
                    N3ub
                    * (x / xhat) ** sigma3ub
                    * ((1 - x) / (1 - xhat)) ** (alpha3ub**2)
                )

                term = (
                    g1ub * math.exp(-g1ub * (b / 2.0) ** 2)
                    + (lambda1ub**2)
                    * (g2ub**2)
                    * (1 - g2ub * (b / 2.0) ** 2)
                    * math.exp(-g2ub * (b / 2.0) ** 2)
                    + g3ub * (lambda2ub**2) * math.exp(-g3ub * (b / 2.0) ** 2)
                )
                denom = g1ub + (lambda1ub**2) * (g2ub**2) + g3ub * (lambda2ub**2)

                return NPevol * (term / denom)

            elif abs(fl) > 2:
                # sea
                N1sea = self._pars[41]
                N2sea = self._pars[42]
                N3sea = self._pars[43]
                alpha1sea = self._pars[44]
                alpha2sea = self._pars[45]
                alpha3sea = self._pars[46]
                sigma1sea = self._pars[47]
                sigma2sea = self._pars[48]
                sigma3sea = self._pars[48]  # repeated
                lambda1sea = self._pars[49]
                lambda2sea = self._pars[50]

                g1sea = (
                    N1sea
                    * (x / xhat) ** sigma1sea
                    * ((1 - x) / (1 - xhat)) ** (alpha1sea**2)
                )
                g2sea = (
                    N2sea
                    * (x / xhat) ** sigma2sea
                    * ((1 - x) / (1 - xhat)) ** (alpha2sea**2)
                )
                g3sea = (
                    N3sea
                    * (x / xhat) ** sigma3sea
                    * ((1 - x) / (1 - xhat)) ** (alpha3sea**2)
                )

                term = (
                    g1sea * math.exp(-g1sea * (b / 2.0) ** 2)
                    + (lambda1sea**2)
                    * (g2sea**2)
                    * (1 - g2sea * (b / 2.0) ** 2)
                    * math.exp(-g2sea * (b / 2.0) ** 2)
                    + g3sea * (lambda2sea**2) * math.exp(-g3sea * (b / 2.0) ** 2)
                )
                denom = g1sea + (lambda1sea**2) * (g2sea**2) + g3sea * (lambda2sea**2)

                return NPevol * (term / denom)
            else:
                return 0.0

        # ----------------------
        # TMD FF of the pion: ifunc == 2
        # ----------------------
        elif ifunc == 2:
            z2 = x * x
            if fl == 2 or fl == -1:
                # e.g. u -> pi or anti-d -> pi
                N4upi = self._pars[51]
                N5upi = self._pars[52]
                beta1upi = self._pars[53]
                beta2upi = self._pars[54]
                delta1upi = self._pars[55]
                delta2upi = self._pars[56]
                gamma1upi = self._pars[57]
                gamma2upi = self._pars[58]
                lambdaFupi = self._pars[59]

                g4upi = (
                    N4upi
                    * ((x**beta1upi) + (delta1upi**2))
                    / ((zhat**beta1upi) + (delta1upi**2))
                    * ((1 - x) / (1 - zhat)) ** (gamma1upi**2)
                )
                g5upi = (
                    N5upi
                    * ((x**beta2upi) + (delta2upi**2))
                    / ((zhat**beta2upi) + (delta2upi**2))
                    * ((1 - x) / (1 - zhat)) ** (gamma2upi**2)
                )

                term = g4upi * math.exp(-g4upi * (b / 2.0) ** 2 / z2) + (
                    lambdaFupi / z2
                ) * (g5upi**2) * (1 - g5upi * (b / 2.0) ** 2 / z2) * math.exp(
                    -g5upi * (b / 2.0) ** 2 / z2
                )
                denom = g4upi + (lambdaFupi / z2) * (g5upi**2)

                return NPevol * (term / denom)

            elif fl == 1 or fl == -2 or abs(fl) > 2:
                # sea -> pi
                N4seapi = self._pars[60]
                N5seapi = self._pars[61]
                beta1seapi = self._pars[62]
                beta2seapi = self._pars[63]
                delta1seapi = self._pars[64]
                delta2seapi = self._pars[65]
                gamma1seapi = self._pars[66]
                gamma2seapi = self._pars[67]
                lambdaFseapi = self._pars[68]

                g4seapi = (
                    N4seapi
                    * ((x**beta1seapi) + (delta1seapi**2))
                    / ((zhat**beta1seapi) + (delta1seapi**2))
                    * ((1 - x) / (1 - zhat)) ** (gamma1seapi**2)
                )
                g5seapi = (
                    N5seapi
                    * ((x**beta2seapi) + (delta2seapi**2))
                    / ((zhat**beta2seapi) + (delta2seapi**2))
                    * ((1 - x) / (1 - zhat)) ** (gamma2seapi**2)
                )

                term = g4seapi * math.exp(-g4seapi * (b / 2.0) ** 2 / z2) + (
                    lambdaFseapi / z2
                ) * (g5seapi**2) * (1 - g5seapi * (b / 2.0) ** 2 / z2) * math.exp(
                    -g5seapi * (b / 2.0) ** 2 / z2
                )
                denom = g4seapi + (lambdaFseapi / z2) * (g5seapi**2)

                return NPevol * (term / denom)

            else:
                return 0.0

        # ----------------------
        # TMD FF of the kaon: ifunc == 3
        # ----------------------
        elif ifunc == 3:
            z2 = x * x
            if fl == 2:
                # u -> K
                N4uka = self._pars[69]
                N5uka = self._pars[70]
                beta1uka = self._pars[71]
                beta2uka = self._pars[72]
                delta1uka = self._pars[73]
                delta2uka = self._pars[74]
                gamma1uka = self._pars[75]
                gamma2uka = self._pars[76]
                lambdaFuka = self._pars[77]

                g4uka = (
                    N4uka
                    * ((x**beta1uka) + (delta1uka**2))
                    / ((zhat**beta1uka) + (delta1uka**2))
                    * ((1 - x) / (1 - zhat)) ** (gamma1uka**2)
                )
                g5uka = (
                    N5uka
                    * ((x**beta2uka) + (delta2uka**2))
                    / ((zhat**beta2uka) + (delta2uka**2))
                    * ((1 - x) / (1 - zhat)) ** (gamma2uka**2)
                )

                term = g4uka * math.exp(-g4uka * (b / 2.0) ** 2 / z2) + (
                    lambdaFuka / z2
                ) * (g5uka**2) * (1 - g5uka * (b / 2.0) ** 2 / z2) * math.exp(
                    -g5uka * (b / 2.0) ** 2 / z2
                )
                denom = g4uka + (lambdaFuka / z2) * (g5uka**2)

                return NPevol * (term / denom)

            elif fl == -3:
                # s-bar -> K^-
                N4sbka = self._pars[78]
                N5sbka = self._pars[79]
                beta1sbka = self._pars[80]
                beta2sbka = self._pars[81]
                delta1sbka = self._pars[82]
                delta2sbka = self._pars[83]
                gamma1sbka = self._pars[84]
                gamma2sbka = self._pars[85]
                lambdaFsbka = self._pars[86]

                g4sbka = (
                    N4sbka
                    * ((x**beta1sbka) + (delta1sbka**2))
                    / ((zhat**beta1sbka) + (delta1sbka**2))
                    * ((1 - x) / (1 - zhat)) ** (gamma1sbka**2)
                )
                g5sbka = (
                    N5sbka
                    * ((x**beta2sbka) + (delta2sbka**2))
                    / ((zhat**beta2sbka) + (delta2sbka**2))
                    * ((1 - x) / (1 - zhat)) ** (gamma2sbka**2)
                )

                term = g4sbka * math.exp(-g4sbka * (b / 2.0) ** 2 / z2) + (
                    lambdaFsbka / z2
                ) * (g5sbka**2) * (1 - g5sbka * (b / 2.0) ** 2 / z2) * math.exp(
                    -g5sbka * (b / 2.0) ** 2 / z2
                )
                denom = g4sbka + (lambdaFsbka / z2) * (g5sbka**2)

                return NPevol * (term / denom)

            elif fl == 3 or fl == -2 or abs(fl) == 1 or abs(fl) > 3:
                # sea -> K
                N4seaka = self._pars[87]
                N5seaka = self._pars[88]
                beta1seaka = self._pars[89]
                beta2seaka = self._pars[90]
                delta1seaka = self._pars[91]
                delta2seaka = self._pars[92]
                gamma1seaka = self._pars[93]
                gamma2seaka = self._pars[94]
                lambdaFseaka = self._pars[95]

                g4seaka = (
                    N4seaka
                    * ((x**beta1seaka) + (delta1seaka**2))
                    / ((zhat**beta1seaka) + (delta1seaka**2))
                    * ((1 - x) / (1 - zhat)) ** (gamma1seaka**2)
                )
                g5seaka = (
                    N5seaka
                    * ((x**beta2seaka) + (delta2seaka**2))
                    / ((zhat**beta2seaka) + (delta2seaka**2))
                    * ((1 - x) / (1 - zhat)) ** (gamma2seaka**2)
                )

                term = g4seaka * math.exp(-g4seaka * (b / 2.0) ** 2 / z2) + (
                    lambdaFseaka / z2
                ) * (g5seaka**2) * (1 - g5seaka * (b / 2.0) ** 2 / z2) * math.exp(
                    -g5seaka * (b / 2.0) ** 2 / z2
                )
                denom = g4seaka + (lambdaFseaka / z2) * (g5seaka**2)

                return NPevol * (term / denom)

            else:
                return 0.0

        else:
            # ifunc out of [0..3] => we already checked above, so 0
            return 0.0

    def LatexFormula(self):
        """
        Direct translation of the 'LatexFormula()' method that returns
        a string containing the relevant LaTeX.
        """
        formula = (
            r"$$f^{f}_{\rm NP}(x,\zeta, b_T)= \exp(S_{\rm NP}(\zeta, b_T))"
            r"\frac{g^f_1(x) \exp( - g^f_1(x) \frac{b_T^2}{4}) + \lambda^{f}{}^2 g^f_{2} {}^2(x) ( 1 - g^f_{2}(x) \frac{b_T^2}{4}) \exp( - g^f_{2}(x) \frac{b_T^2}{4}) + \lambda_2^f{}^2 g^f_{3}(x) \exp( - g^f_{3}(x) \frac{b_T^2}{4}) }{  g^f_1(x) +  \lambda^f g^f_{2}{}^2(x) + \lambda^f_2{}^2 g^f_{3}(x) }$$\n"
        )
        formula += (
            r"$$D_{\rm NP}^{f \righarrow \pi,K}(z,\zeta, b_T)= \exp(S_{\rm NP}(\zeta, b_T))"
            r"\frac{g_4^{f \rightarrow \pi,K}(z) \exp( - g_4^{f \rightarrow \pi,K}(z) \frac{b_T^2}{4z^2}) + \frac{\lambda^{f \rightarrow \pi,K}_F}{z^2} g^{f \rightarrow \pi,K}_{5}{}^2(z) \big ( 1 - g^{f \rightarrow\pi,K}_{5}(z) \frac{b_T^2}{4z^2} \big ) \exp( - g^{f \rightarrow\pi,K}_{5}(z) \frac{b_T^2}{4z^2}) }{  g^{f \rightarrow\pi,K}_4(z) +  \frac{\lambda^{f \rightarrow\pi,K}_F}{z^2} g^{f \rightarrow\pi,K}_{5}{}^2(z) }$$\n"
        )
        formula += r"$$S_{\rm NP} = \exp\left[ - g_2^2 \frac{b_T^2}{4} \log \big (\frac{\zeta}{Q_0^2}\big ) \right]$$\n"
        formula += r"$$g^{f}_{1,2,3}(x) = N^{f}_{1,2,3} \frac{x^{\sigma^{f}_{1,2,3}}(1-x)^{\alpha^{f}_{1,2,3} {}^2}}{\hat{x}^{\sigma^{f}_{1,2,3}}(1-\hat{x})^{\alpha^{f}_{1,2,3}{}^2}}$$\n"
        formula += r"$$g^{f \rightarrow \pi,K}_{4,5}(z) = N^{f \rightarrow \pi,K}_{4,5} \frac{(z^{\beta^{f \rightarrow \pi,K}_{1,2}}+\delta^{f \rightarrow \pi,K}_{1,2}{}^2)(1-z)^{\gamma^{f \rightarrow\pi,K}_{1,2}{}^2}}{(\hat{z}^{f \rightarrow \pi,K}_{1,2}}+\delta^{f \rightarrow \pi,K}_{1,2}{}^2)(1-\hat{z})^{f \rightarrow\pi,K}_{1,2}{}^2}$$\n"
        formula += (
            r"$$Q_0^2 = 1\;{\rm GeV}^2,\quad \hat{x} = 0.1,\quad \hat{z} = 0.5$$\n"
        )
        return formula

    def GetParameterNames(self):
        """
        A direct translation of 'GetParameterNames()', returning a list
        of 96 strings that label each entry in self._pars.
        """
        return [
            r"$g_2$",
            r"$N_{1d}$",
            r"$N_{2d}$",
            r"$N_{3d}$",
            r"$\alpha_{1d}$",
            r"$\alpha_{2d}$",
            r"$\alpha_{3d}$",
            r"$\sigma_{1d}$",
            r"$\sigma_{3d}$",
            r"$\lambda_{1d}$",
            r"$\lambda_{2d}$",
            r"$N_{1db}$",
            r"$N_{2db}$",
            r"$N_{3db}$",
            r"$\alpha_{1db}$",
            r"$\alpha_{2db}$",
            r"$\alpha_{3db}$",
            r"$\sigma_{1db}$",
            r"$\sigma_{3db}$",
            r"$\lambda_{1db}$",
            r"$\lambda_{2db}$",
            r"$N_{1u}$",
            r"$N_{2u}$",
            r"$N_{3u}$",
            r"$\alpha_{1u}$",
            r"$\alpha_{2u}$",
            r"$\alpha_{3u}$",
            r"$\sigma_{1u}$",
            r"$\sigma_{3u}$",
            r"$\lambda_{1u}$",
            r"$\lambda_{2u}$",
            r"$N_{1ub}$",
            r"$N_{2ub}$",
            r"$N_{3ub}$",
            r"$\alpha_{1ub}$",
            r"$\alpha_{2ub}$",
            r"$\alpha_{3ub}$",
            r"$\sigma_{1ub}$",
            r"$\sigma_{3ub}$",
            r"$\lambda_{1ub}$",
            r"$\lambda_{2ub}$",
            r"$N_{1sea}$",
            r"$N_{2sea}$",
            r"$N_{3sea}$",
            r"$\alpha_{1sea}$",
            r"$\alpha_{2sea}$",
            r"$\alpha_{3sea}$",
            r"$\sigma_{1sea}$",
            r"$\sigma_{3sea}$",
            r"$\lambda_{1sea}$",
            r"$\lambda_{2sea}$",
            r"$N_{4upi}$",
            r"$N_{5upi}$",
            r"$\beta_{1upi}$",
            r"$\beta_{2upi}$",
            r"$\delta_{1upi}$",
            r"$\delta_{2upi}$",
            r"$\gamma_{1upi}$",
            r"$\gamma_{2upi}$",
            r"$\lambda_{Fupi}$",
            r"$N_{4seapi}$",
            r"$N_{5seapi}$",
            r"$\beta_{1seapi}$",
            r"$\beta_{2seapi}$",
            r"$\delta_{1seapi}$",
            r"$\delta_{2seapi}$",
            r"$\gamma_{1seapi}$",
            r"$\gamma_{2seapi}$",
            r"$\lambda_{Fseapi}$",
            r"$N_{4uka}$",
            r"$N_{5uka}$",
            r"$\beta_{1uka}$",
            r"$\beta_{2uka}$",
            r"$\delta_{1uka}$",
            r"$\delta_{2uka}$",
            r"$\gamma_{1uka}$",
            r"$\gamma_{2uka}$",
            r"$\lambda_{Fuka}$",
            r"$N_{4sbka}$",
            r"$N_{5sbka}$",
            r"$\beta_{1sbka}$",
            r"$\beta_{2sbka}$",
            r"$\delta_{1sbka}$",
            r"$\delta_{2sbka}$",
            r"$\gamma_{1sbka}$",
            r"$\gamma_{2sbka}$",
            r"$\lambda_{Fsbka}$",
            r"$N_{4seaka}$",
            r"$N_{5seaka}$",
            r"$\beta_{1seaka}$",
            r"$\beta_{2seaka}$",
            r"$\delta_{1seaka}$",
            r"$\delta_{2seaka}$",
            r"$\gamma_{1seaka}$",
            r"$\gamma_{2seaka}$",
            r"$\lambda_{Fseaka}$",
        ]

    def GetDescription(self):
        """
        Returns a string describing this parameterisation.
        """
        return "Parameterisation used for the MAP 2024 Flavour Dependent TMD analysis."
