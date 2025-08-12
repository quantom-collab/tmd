//
// Author: Valerio Bertone: valerio.bertone@cern.ch,
//         Alessandro Bacchetta: alessandro.bacchetta@unipv.it
//         Lorenzo Rossi : lorenzo.rossi04@universitadipavia.it

#pragma once

#include "NangaParbat/parameterisation.h"

#include <math.h>

namespace NangaParbat
{
  /**
   * @brief MAP 2024 Flavour Dependent parameterisation derived from the
   * "Parameterisation" mother class.
   */
  class MAP24_FD: public NangaParbat::Parameterisation
  {
  public:
    MAP24_FD(): Parameterisation{"MAP24_FD", 3, std::vector<double>{0.2E+00, 0.1E+00, 0.1E+00, 0.1E+00, 0.1E+00, 0.1E+00, 0.1E+00, 0.1E+00, 0.1E+00, 0.1E+00, 0.1E+00, 0.1E+00, 0.1E+00, 0.1E+00, 0.1E+00, 0.1E+00, 0.1E+00, 0.1E+00, 0.1E+00, 0.1E+00, 0.1E+00, 0.1E+00, 0.1E+00, 0.1E+00, 0.1E+00, 0.1E+00, 0.1E+00, 0.1E+00, 0.1E+00, 0.1E+00, 0.1E+00, 0.1E+00, 0.1E+00, 0.1E+00, 0.1E+00, 0.1E+00, 0.1E+00, 0.1E+00, 0.1E+00, 0.1E+00, 0.1E+00, 0.1E+00, 0.1E+00, 0.1E+00, 0.1E+00, 0.1E+00, 0.1E+00, 0.1E+00, 0.1E+00, 0.1E+00, 0.1E+00, 0.1E+00, 0.1E+00, 0.1E+00, 0.1E+00, 0.1E+00, 0.1E+00, 0.1E+00, 0.1E+00, 0.1E+00, 0.1E+00, 0.1E+00, 0.1E+00, 0.1E+00, 0.1E+00, 0.1E+00, 0.1E+00, 0.1E+00, 0.1E+00, 0.1E+00, 0.1E+00, 0.1E+00, 0.1E+00, 0.1E+00, 0.1E+00, 0.1E+00, 0.1E+00, 0.1E+00, 0.1E+00, 0.1E+00, 0.1E+00, 0.1E+00, 0.1E+00, 0.1E+00, 0.1E+00, 0.1E+00, 0.1E+00, 0.1E+00, 0.1E+00, 0.1E+00, 0.1E+00, 0.1E+00, 0.1E+00, 0.1E+00, 0.1E+00, 0.1E+00}} { };

    double Evaluate(double const& x, double const& b, double const& zeta, int const& ifunc, int const& fl) const
    {
      if (ifunc < 0 || ifunc > 3)
        throw std::runtime_error("[MAP24_FD::Evaluate]: function index out of range");

      // If the value of 'x' exceeds one returns zero
      if (x >= 1)
        return 0;

      // Evolution
      const double g2     = this->_pars[0];
      const double b2     = b * b;
      const double lnz    = log(zeta / _Q02);
      const double NPevol = exp( - pow(g2,2) * b2 * lnz / 4 );
      const double xhat   = 0.1;
      const double zhat   = 0.5;

      // TMD PDFs
      if (ifunc == 0 || ifunc == 1)
        {
          if (fl == 1)
            {
              const double N1d      = this->_pars[1];
              const double N2d      = this->_pars[2];
              const double N3d      = this->_pars[3];
              const double alpha1d  = this->_pars[4];
              const double alpha2d  = this->_pars[5];
              const double alpha3d  = this->_pars[6];
              const double sigma1d  = this->_pars[7];
              const double sigma2d  = this->_pars[8];
              const double sigma3d  = this->_pars[8];
              const double lambda1d = this->_pars[9];
              const double lambda2d = this->_pars[10];
              const double g1d     = N1d * pow(x / xhat, sigma1d) * pow((1 - x) / (1 - xhat), pow(alpha1d,2));
              const double g2d     = N2d * pow(x / xhat, sigma2d) * pow((1 - x) / (1 - xhat), pow(alpha2d,2));
              const double g3d     = N3d * pow(x / xhat, sigma3d) * pow((1 - x) / (1 - xhat), pow(alpha3d,2));
              return NPevol * ( g1d * exp( - g1d * pow(b / 2, 2))
                                +  pow(lambda1d, 2)  * pow(g2d, 2) * ( 1 - g2d * pow(b / 2, 2)) * exp( - g2d * pow(b / 2, 2)) + g3d * pow(lambda2d, 2) * exp( - g3d * pow(b / 2, 2)))
                     / ( g1d +  pow(lambda1d, 2)  * pow(g2d, 2) + g3d * pow(lambda2d, 2));
            }
          if (fl == -1)
            {
              const double N1db      = this->_pars[11];
              const double N2db      = this->_pars[12];
              const double N3db      = this->_pars[13];
              const double alpha1db  = this->_pars[14];
              const double alpha2db  = this->_pars[15];
              const double alpha3db  = this->_pars[16];
              const double sigma1db  = this->_pars[17];
              const double sigma2db  = this->_pars[18];
              const double sigma3db  = this->_pars[18];
              const double lambda1db = this->_pars[19];
              const double lambda2db = this->_pars[20];
              const double g1db     = N1db * pow(x / xhat, sigma1db) * pow((1 - x) / (1 - xhat), pow(alpha1db,2));
              const double g2db     = N2db * pow(x / xhat, sigma2db) * pow((1 - x) / (1 - xhat), pow(alpha2db,2));
              const double g3db     = N3db * pow(x / xhat, sigma3db) * pow((1 - x) / (1 - xhat), pow(alpha3db,2));
              return NPevol * ( g1db * exp( - g1db * pow(b / 2, 2))
                                +  pow(lambda1db, 2)  * pow(g2db, 2) * ( 1 - g2db * pow(b / 2, 2)) * exp( - g2db * pow(b / 2, 2)) + g3db * pow(lambda2db, 2) * exp( - g3db * pow(b / 2, 2)))
                     / ( g1db +  pow(lambda1db, 2)  * pow(g2db, 2) + g3db * pow(lambda2db, 2));
            }

          else if (fl == 2)
            {
              const double N1u      = this->_pars[21];
              const double N2u      = this->_pars[22];
              const double N3u      = this->_pars[23];
              const double alpha1u  = this->_pars[24];
              const double alpha2u  = this->_pars[25];
              const double alpha3u  = this->_pars[26];
              const double sigma1u  = this->_pars[27];
              const double sigma2u  = this->_pars[28];
              const double sigma3u  = this->_pars[28];
              const double lambda1u = this->_pars[29];
              const double lambda2u = this->_pars[30];
              const double g1u     = N1u * pow(x / xhat, sigma1u) * pow((1 - x) / (1 - xhat), pow(alpha1u,2));
              const double g2u     = N2u * pow(x / xhat, sigma2u) * pow((1 - x) / (1 - xhat), pow(alpha2u,2));
              const double g3u     = N3u * pow(x / xhat, sigma3u) * pow((1 - x) / (1 - xhat), pow(alpha3u,2));
              return NPevol * ( g1u * exp( - g1u * pow(b / 2, 2))
                                +  pow(lambda1u, 2)  * pow(g2u, 2) * ( 1 - g2u * pow(b / 2, 2)) * exp( - g2u * pow(b / 2, 2)) + g3u * pow(lambda2u, 2) * exp( - g3u * pow(b / 2, 2)))
                     / ( g1u +  pow(lambda1u, 2)  * pow(g2u, 2) + g3u * pow(lambda2u, 2));
            }
          else if (fl == -2)
            {
              const double N1ub      = this->_pars[31];
              const double N2ub      = this->_pars[32];
              const double N3ub      = this->_pars[33];
              const double alpha1ub  = this->_pars[34];
              const double alpha2ub  = this->_pars[35];
              const double alpha3ub  = this->_pars[36];
              const double sigma1ub  = this->_pars[37];
              const double sigma2ub  = this->_pars[38];
              const double sigma3ub  = this->_pars[38];
              const double lambda1ub = this->_pars[39];
              const double lambda2ub = this->_pars[40];
              const double g1ub     = N1ub * pow(x / xhat, sigma1ub) * pow((1 - x) / (1 - xhat), pow(alpha1ub,2));
              const double g2ub     = N2ub * pow(x / xhat, sigma2ub) * pow((1 - x) / (1 - xhat), pow(alpha2ub,2));
              const double g3ub     = N3ub * pow(x / xhat, sigma3ub) * pow((1 - x) / (1 - xhat), pow(alpha3ub,2));
              return NPevol * ( g1ub * exp( - g1ub * pow(b / 2, 2))
                                +  pow(lambda1ub, 2)  * pow(g2ub, 2) * ( 1 - g2ub * pow(b / 2, 2)) * exp( - g2ub * pow(b / 2, 2)) + g3ub * pow(lambda2ub, 2) * exp( - g3ub * pow(b / 2, 2)))
                     / ( g1ub +  pow(lambda1ub, 2)  * pow(g2ub, 2) + g3ub * pow(lambda2ub, 2));
            }

          else if (abs(fl) > 2)
            {
              const double N1sea      = this->_pars[41];
              const double N2sea      = this->_pars[42];
              const double N3sea      = this->_pars[43];
              const double alpha1sea  = this->_pars[44];
              const double alpha2sea  = this->_pars[45];
              const double alpha3sea  = this->_pars[46];
              const double sigma1sea  = this->_pars[47];
              const double sigma2sea  = this->_pars[48];
              const double sigma3sea  = this->_pars[48];
              const double lambda1sea = this->_pars[49];
              const double lambda2sea = this->_pars[50];
              const double g1sea     = N1sea * pow(x / xhat, sigma1sea) * pow((1 - x) / (1 - xhat), pow(alpha1sea,2));
              const double g2sea     = N2sea * pow(x / xhat, sigma2sea) * pow((1 - x) / (1 - xhat), pow(alpha2sea,2));
              const double g3sea     = N3sea * pow(x / xhat, sigma3sea) * pow((1 - x) / (1 - xhat), pow(alpha3sea,2));
              return NPevol * ( g1sea * exp( - g1sea * pow(b / 2, 2))
                                +  pow(lambda1sea, 2)  * pow(g2sea, 2) * ( 1 - g2sea * pow(b / 2, 2)) * exp( - g2sea * pow(b / 2, 2)) + g3sea * pow(lambda2sea, 2) * exp( - g3sea * pow(b / 2, 2)))
                     / ( g1sea +  pow(lambda1sea, 2)  * pow(g2sea, 2) + g3sea * pow(lambda2sea, 2));
            }

          else
            return 0;
        }
      // TMD FF of the pion
      else if (ifunc == 2)
        {
          const double z2         = x * x;

          if (fl == 2 || fl == -1)
            {
              const double N4upi      = this->_pars[51];
              const double N5upi      = this->_pars[52];
              const double beta1upi   = this->_pars[53];
              const double beta2upi   = this->_pars[54];
              const double delta1upi  = this->_pars[55];
              const double delta2upi  = this->_pars[56];
              const double gamma1upi  = this->_pars[57];
              const double gamma2upi  = this->_pars[58];
              const double lambdaFupi = this->_pars[59];
              const double g4upi      = N4upi * ( ( pow(x, beta1upi) + pow(delta1upi,2) ) / ( pow(zhat, beta1upi) + pow(delta1upi,2) ) ) * pow((1 - x) / (1 - zhat), pow(gamma1upi,2));
              const double g5upi      = N5upi * ( ( pow(x, beta2upi) + pow(delta2upi,2) ) / ( pow(zhat, beta2upi) + pow(delta2upi,2) ) ) * pow((1 - x) / (1 - zhat), pow(gamma2upi,2));
              return NPevol * ( g4upi * exp( - g4upi * pow(b / 2, 2) / z2 )
                                + ( lambdaFupi / z2 ) * pow(g5upi, 2) * ( 1 - g5upi * pow(b / 2, 2) / z2 ) * exp( - g5upi * pow(b / 2, 2) / z2 ) )
                     / ( g4upi + ( lambdaFupi / z2 ) * pow(g5upi, 2) );
            }

          else if (fl == 1 || fl == -2 || abs(fl) > 2)
            {
              const double N4seapi      = this->_pars[60];
              const double N5seapi      = this->_pars[61];
              const double beta1seapi   = this->_pars[62];
              const double beta2seapi   = this->_pars[63];
              const double delta1seapi  = this->_pars[64];
              const double delta2seapi  = this->_pars[65];
              const double gamma1seapi  = this->_pars[66];
              const double gamma2seapi  = this->_pars[67];
              const double lambdaFseapi = this->_pars[68];
              const double g4seapi      = N4seapi * ( ( pow(x, beta1seapi) + pow(delta1seapi,2) ) / ( pow(zhat, beta1seapi) + pow(delta1seapi,2) ) ) * pow((1 - x) / (1 - zhat), pow(gamma1seapi,2));
              const double g5seapi      = N5seapi * ( ( pow(x, beta2seapi) + pow(delta2seapi,2) ) / ( pow(zhat, beta2seapi) + pow(delta2seapi,2) ) ) * pow((1 - x) / (1 - zhat), pow(gamma2seapi,2));
              return NPevol * ( g4seapi * exp( - g4seapi * pow(b / 2, 2) / z2 )
                                + ( lambdaFseapi / z2 ) * pow(g5seapi, 2) * ( 1 - g5seapi * pow(b / 2, 2) / z2 ) * exp( - g5seapi * pow(b / 2, 2) / z2 ) )
                     / ( g4seapi + ( lambdaFseapi / z2 ) * pow(g5seapi, 2) );
            }

          else
            return 0;
        }
      // TMD FF of the kaon
      else if (ifunc == 3)
        {
          const double z2         = x * x;

          if (fl == 2)
            {
              const double N4uka      = this->_pars[69];
              const double N5uka      = this->_pars[70];
              const double beta1uka   = this->_pars[71];
              const double beta2uka   = this->_pars[72];
              const double delta1uka  = this->_pars[73];
              const double delta2uka  = this->_pars[74];
              const double gamma1uka  = this->_pars[75];
              const double gamma2uka  = this->_pars[76];
              const double lambdaFuka = this->_pars[77];
              const double g4uka      = N4uka * ( ( pow(x, beta1uka) + pow(delta1uka,2) ) / ( pow(zhat, beta1uka) + pow(delta1uka,2) ) ) * pow((1 - x) / (1 - zhat), pow(gamma1uka,2));
              const double g5uka      = N5uka * ( ( pow(x, beta2uka) + pow(delta2uka,2) ) / ( pow(zhat, beta2uka) + pow(delta2uka,2) ) ) * pow((1 - x) / (1 - zhat), pow(gamma2uka,2));
              return NPevol * ( g4uka * exp( - g4uka * pow(b / 2, 2) / z2 )
                                + ( lambdaFuka / z2 ) * pow(g5uka, 2) * ( 1 - g5uka * pow(b / 2, 2) / z2 ) * exp( - g5uka * pow(b / 2, 2) / z2 ) )
                     / ( g4uka + ( lambdaFuka / z2 ) * pow(g5uka, 2) );
            }

          if (fl == -3)
            {
              const double N4sbka      = this->_pars[78];
              const double N5sbka      = this->_pars[79];
              const double beta1sbka   = this->_pars[80];
              const double beta2sbka   = this->_pars[81];
              const double delta1sbka  = this->_pars[82];
              const double delta2sbka  = this->_pars[83];
              const double gamma1sbka  = this->_pars[84];
              const double gamma2sbka  = this->_pars[85];
              const double lambdaFsbka = this->_pars[86];
              const double g4sbka      = N4sbka * ( ( pow(x, beta1sbka) + pow(delta1sbka,2) ) / ( pow(zhat, beta1sbka) + pow(delta1sbka,2) ) ) * pow((1 - x) / (1 - zhat), pow(gamma1sbka,2));
              const double g5sbka      = N5sbka * ( ( pow(x, beta2sbka) + pow(delta2sbka,2) ) / ( pow(zhat, beta2sbka) + pow(delta2sbka,2) ) ) * pow((1 - x) / (1 - zhat), pow(gamma2sbka,2));
              return NPevol * ( g4sbka * exp( - g4sbka * pow(b / 2, 2) / z2 )
                                + ( lambdaFsbka / z2 ) * pow(g5sbka, 2) * ( 1 - g5sbka * pow(b / 2, 2) / z2 ) * exp( - g5sbka * pow(b / 2, 2) / z2 ) )
                     / ( g4sbka + ( lambdaFsbka / z2 ) * pow(g5sbka, 2) );
            }

          else if (fl == 3 || fl == -2 || abs(fl) == 1 || abs(fl) > 3)
            {
              const double N4seaka      = this->_pars[87];
              const double N5seaka      = this->_pars[88];
              const double beta1seaka   = this->_pars[89];
              const double beta2seaka   = this->_pars[90];
              const double delta1seaka  = this->_pars[91];
              const double delta2seaka  = this->_pars[92];
              const double gamma1seaka  = this->_pars[93];
              const double gamma2seaka  = this->_pars[94];
              const double lambdaFseaka = this->_pars[95];
              const double g4seaka      = N4seaka * ( ( pow(x, beta1seaka) + pow(delta1seaka,2) ) / ( pow(zhat, beta1seaka) + pow(delta1seaka,2) ) ) * pow((1 - x) / (1 - zhat), pow(gamma1seaka,2));
              const double g5seaka      = N5seaka * ( ( pow(x, beta2seaka) + pow(delta2seaka,2) ) / ( pow(zhat, beta2seaka) + pow(delta2seaka,2) ) ) * pow((1 - x) / (1 - zhat), pow(gamma2seaka,2));
              return NPevol * ( g4seaka * exp( - g4seaka * pow(b / 2, 2) / z2 )
                                + ( lambdaFseaka / z2 ) * pow(g5seaka, 2) * ( 1 - g5seaka * pow(b / 2, 2) / z2 ) * exp( - g5seaka * pow(b / 2, 2) / z2 ) )
                     / ( g4seaka + ( lambdaFseaka / z2 ) * pow(g5seaka, 2) );
            }

          else
            return 0;
        }
      else
        {
          return 0;
        }
    };

    std::string LatexFormula() const
    {
      std::string formula;
      formula  = R"delimiter($$f^{f}_{\rm NP}(x,\zeta, b_T)= \exp(S_{\rm NP}(\zeta, b_T)))delimiter";
      formula += R"delimiter(\frac{g^f_1(x) \exp( - g^f_1(x) \frac{b_T^2}{4}) + \lambda^{f}{}^2 g^f_{2} {}^2(x) ( 1 - g^f_{2}(x) \frac{b_T^2}{4}) \exp( - g^f_{2}(x) \frac{b_T^2}{4}) + \lambda_2^f{}^2 g^f_{3}(x) \exp( - g^f_{3}(x) \frac{b_T^2}{4}) }{  g^f_1(x) +  \lambda^f g^f_{2}{}^2(x) + \lambda^f_2{}^2 g^f_{3}(x) }$$)delimiter";
      formula += R"delimiter($$D_{\rm NP}^{f \righarrow \pi,K}(z,\zeta, b_T)= \exp(S_{\rm NP}(\zeta, b_T)))delimiter";
      formula += R"delimiter(\frac{g_4^{f \rightarrow \pi,K}(z) \exp( - g_4^{f \rightarrow \pi,K}(z) \frac{b_T^2}{4z^2}) + \frac{\lambda^{f \rightarrow \pi,K}_F}{z^2} g^{f \rightarrow \pi,K}_{5}{}^2(z) \big ( 1 - g^{f \rightarrow\pi,K}_{5}(z) \frac{b_T^2}{4z^2} \big ) \exp( - g^{f \rightarrow\pi,K}_{5}(z) \frac{b_T^2}{4z^2}) }{  g^{f \rightarrow\pi,K}_4(z) +  \frac{\lambda^{f \rightarrow\pi,K}_F}{z^2} g^{f \rightarrow\pi,K}_{5}{}^2(z) }$$)delimiter";
      formula += R"delimiter($$S_{\rm NP} = \exp\left[ - g_2^2 \frac{b_T^2}{4} \log \big (\frac{\zeta}{Q_0^2}\big ) \right]$$)delimiter";
      formula += R"delimiter($$    g^{f}_{1,2,3}(x) = N^{f}_{1,2,3} \frac{x^{\sigma^{f}_{1,3,3}}(1-x)^{\alpha^{f}_{1,2,3} {}^2}}{\hat{x}^{\sigma^{f}_{1,3,3}}(1-\hat{x})^{\alpha^{f}_{1,2,3}{}^2}}$$)delimiter";
      formula += R"delimiter($$g^{f \rightarrow \pi,K}_{4,5}(z) = N^{f \rightarrow \pi,K}_{4,5} \frac{(z^{\beta^{f \rightarrow \pi,K}_{1,2}}+\delta^{f \rightarrow \pi,K}_{1,2}{}^2)(1-z)^{\gamma^{f \rightarrow\pi,K}_{1,2}{}^2}}{(\hat{z}^{f \rightarrow\beta^{\pi,K}_{1,2}}+\delta^{f \rightarrow\pi,K}_{1,2}{}^2)(1-\hat{z})^{f \rightarrow\gamma^{\pi,K}_{1,2}{}^2}}$$)delimiter";
      formula += R"delimiter($$Q_0^2 = 1\;{\rm GeV}^2$$)delimiter";
      formula += R"delimiter($$\hat{x} = 0.1$$)delimiter";
      formula += R"delimiter($$\hat{z} = 0.5$$)delimiter";

      return formula;
    };
    std::vector<std::string> GetParameterNames() const
    {
      return {R"delimiter($g_2$)delimiter",
              R"delimiter($N_{1d}$)delimiter",
              R"delimiter($N_{2d}$)delimiter",
              R"delimiter($N_{3d}$)delimiter",
              R"delimiter($\alpha_{1d}$)delimiter",
              R"delimiter($\alpha_{2d}$)delimiter",
              R"delimiter($\alpha_{3d}$)delimiter",
              R"delimiter($\sigma_{1d}$)delimiter",
              R"delimiter($\sigma_{3d}$)delimiter",
              R"delimiter($\lambda_{1d}$)delimiter",
              R"delimiter($\lambda_{2d}$)delimiter",
              R"delimiter($N_{1db}$)delimiter",
              R"delimiter($N_{2db}$)delimiter",
              R"delimiter($N_{3db}$)delimiter",
              R"delimiter($\alpha_{1db}$)delimiter",
              R"delimiter($\alpha_{2db}$)delimiter",
              R"delimiter($\alpha_{3db}$)delimiter",
              R"delimiter($\sigma_{1db}$)delimiter",
              R"delimiter($\sigma_{3db}$)delimiter",
              R"delimiter($\lambda_{1db}$)delimiter",
              R"delimiter($\lambda_{2db}$)delimiter",
              R"delimiter($N_{1u}$)delimiter",
              R"delimiter($N_{2u}$)delimiter",
              R"delimiter($N_{3u}$)delimiter",
              R"delimiter($\alpha_{1u}$)delimiter",
              R"delimiter($\alpha_{2u}$)delimiter",
              R"delimiter($\alpha_{3u}$)delimiter",
              R"delimiter($\sigma_{1u}$)delimiter",
              R"delimiter($\sigma_{3u}$)delimiter",
              R"delimiter($\lambda_{1u}$)delimiter",
              R"delimiter($\lambda_{2u}$)delimiter",
              R"delimiter($N_{1ub}$)delimiter",
              R"delimiter($N_{2ub}$)delimiter",
              R"delimiter($N_{3ub}$)delimiter",
              R"delimiter($\alpha_{1ub}$)delimiter",
              R"delimiter($\alpha_{2ub}$)delimiter",
              R"delimiter($\alpha_{3ub}$)delimiter",
              R"delimiter($\sigma_{1ub}$)delimiter",
              R"delimiter($\sigma_{3ub}$)delimiter",
              R"delimiter($\lambda_{1ub}$)delimiter",
              R"delimiter($\lambda_{2ub}$)delimiter",
              R"delimiter($N_{1sea}$)delimiter",
              R"delimiter($N_{2sea}$)delimiter",
              R"delimiter($N_{3sea}$)delimiter",
              R"delimiter($\alpha_{1sea}$)delimiter",
              R"delimiter($\alpha_{2sea}$)delimiter",
              R"delimiter($\alpha_{3sea}$)delimiter",
              R"delimiter($\sigma_{1sea}$)delimiter",
              R"delimiter($\sigma_{3sea}$)delimiter",
              R"delimiter($\lambda_{1sea}$)delimiter",
              R"delimiter($\lambda_{2sea}$)delimiter",
              R"delimiter($N_{4upi}$)delimiter",
              R"delimiter($N_{5upi}$)delimiter",
              R"delimiter($\beta_{1upi}$)delimiter",
              R"delimiter($\beta_{2upi}$)delimiter",
              R"delimiter($\delta_{1upi}$)delimiter",
              R"delimiter($\delta_{2upi}$)delimiter",
              R"delimiter($\gamma_{1upi}$)delimiter",
              R"delimiter($\gamma_{2upi}$)delimiter",
              R"delimiter($\lambda_{Fupi}$)delimiter",
              R"delimiter($N_{4seapi}$)delimiter",
              R"delimiter($N_{5seapi}$)delimiter",
              R"delimiter($\beta_{1seapi}$)delimiter",
              R"delimiter($\beta_{2seapi}$)delimiter",
              R"delimiter($\delta_{1seapi}$)delimiter",
              R"delimiter($\delta_{2seapi}$)delimiter",
              R"delimiter($\gamma_{1seapi}$)delimiter",
              R"delimiter($\gamma_{2seapi}$)delimiter",
              R"delimiter($\lambda_{Fseapi}$)delimiter",
              R"delimiter($N_{4uka}$)delimiter",
              R"delimiter($N_{5uka}$)delimiter",
              R"delimiter($\beta_{1uka}$)delimiter",
              R"delimiter($\beta_{2uka}$)delimiter",
              R"delimiter($\delta_{1uka}$)delimiter",
              R"delimiter($\delta_{2uka}$)delimiter",
              R"delimiter($\gamma_{1uka}$)delimiter",
              R"delimiter($\gamma_{2uka}$)delimiter",
              R"delimiter($\lambda_{Fuka}$)delimiter",
              R"delimiter($N_{4sbka}$)delimiter",
              R"delimiter($N_{5sbka}$)delimiter",
              R"delimiter($\beta_{1sbka}$)delimiter",
              R"delimiter($\beta_{2sbka}$)delimiter",
              R"delimiter($\delta_{1sbka}$)delimiter",
              R"delimiter($\delta_{2sbka}$)delimiter",
              R"delimiter($\gamma_{1sbka}$)delimiter",
              R"delimiter($\gamma_{2sbka}$)delimiter",
              R"delimiter($\lambda_{Fsbka}$)delimiter",
              R"delimiter($N_{4seaka}$)delimiter",
              R"delimiter($N_{5seaka}$)delimiter",
              R"delimiter($\beta_{1seaka}$)delimiter",
              R"delimiter($\beta_{2seaka}$)delimiter",
              R"delimiter($\delta_{1seaka}$)delimiter",
              R"delimiter($\delta_{2seaka}$)delimiter",
              R"delimiter($\gamma_{1seaka}$)delimiter",
              R"delimiter($\gamma_{2seaka}$)delimiter",
              R"delimiter($\lambda_{Fseaka}$)delimiter"};

    };

    std::string GetDescription() const
    {
      return "Parameterisation used for the MAP 2024 Flavour Dependent TMD analysis.";
    };

  private:
    const double _Q02  = 1;
  };
}
