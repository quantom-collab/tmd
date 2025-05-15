//
// APFEL++ 2017
//
// Author: Valerio Bertone: valerio.bertone@cern.ch
//

#include <apfel/apfelxx.h>
#include <apfel/betaqcd.h>
#include <apfel/gammak.h>
#include <apfel/kcs.h>

#include <LHAPDF/LHAPDF.h>

#include <NangaParbat/bstar.h>

// Analitic g-functions
double gK0(int const &nf, double const &lambda)
{
   const double bt0 = -2 * apfel::beta0qcd(nf);
   return apfel::CF * apfel::gammaK0() / bt0 * log(1 - lambda);
}

double gK1(int const &nf, double const &lambda)
{
   const double bt0 = -2 * apfel::beta0qcd(nf);
   const double b1  = apfel::beta1qcd(nf) / apfel::beta0qcd(nf);
   return apfel::CF * apfel::KCS00() +
          b1 * apfel::CF * apfel::gammaK0() / bt0 * (lambda + log(1 - lambda)) / (1 - lambda) -
          apfel::CF * apfel::gammaK1(nf) / bt0 * lambda / (1 - lambda);
}

double gK2(int const &nf, double const &lambda)
{
   const double bt0 = -2 * apfel::beta0qcd(nf);
   const double b1  = apfel::beta1qcd(nf) / apfel::beta0qcd(nf);
   const double b2  = apfel::beta2qcd(nf) / apfel::beta0qcd(nf);
   return apfel::CF * apfel::KCS10(nf) -
          b2 * apfel::CF * apfel::gammaK0() / bt0 * pow(lambda, 2) / 2 / pow(1 - lambda, 2) +
          pow(b1, 2) * apfel::CF * apfel::gammaK0() / bt0 *
              (pow(lambda, 2) - pow(log(1 - lambda), 2)) / 2 / pow(1 - lambda, 2) +
          b1 * apfel::CF * apfel::gammaK1(nf) / bt0 *
              (2 * log(1 - lambda) - lambda * (lambda - 2)) / 2 / pow(1 - lambda, 2) +
          apfel::CF * apfel::gammaK2(nf) / bt0 * lambda * (lambda - 2) / 2 / pow(1 - lambda, 2);
}

// Main program
int main()
{
   // Open LHAPDF set
   LHAPDF::PDF *distpdf = LHAPDF::mkPDF("MMHT2014nnlo68cl", 0);

   // Heavy-quark thresholds
   std::vector<double> Thresholds;
   for (auto const &v : distpdf->flavors())
      if (v > 0 && v < 7) Thresholds.push_back(distpdf->quarkThreshold(v));

   // Alpha_s (from PDFs)
   const auto Alphas = [=](double const &mu) -> double {
      return distpdf->alphasQ(mu);
   };

   // x-space grid (to setup APFEL++ computation)
   const apfel::Grid g{{{100, 1e-6, 3}, {60, 1e-1, 3}, {50, 6e-1, 3}, {50, 8e-1, 3}}};

   // Initialize TMD objects
   const auto TmdObj = apfel::InitializeTmdObjects(g, Thresholds);

   // Perturbative order
   const int PerturbativeOrder = 0;

   // Get perturbative part of the Collins-Soper kernel
   const std::function<double(double const &, double const &)> CSKernel =
       apfel::CollinsSoperKernel(TmdObj, Alphas, PerturbativeOrder);

   // Get b* prescription
   // const std::function<double(double const&, double const&)> bs =
   // NangaParbat::bstarMap.at("bstarmin");
   // const std::function<double(double const&, double const&)> bs =
   // NangaParbat::bstarMap.at("bstarCSS");
   const std::function<double(double const &, double const &)> bs = [](double const &bT,
                                                                       double const &) -> double {
      return bT;
   };

   // Non-perturbative part of the evolution (modify at will)
   const double g2                                    = 0; // 0.2;
   const std::function<double(double const &)> NPEvol = [=](double const &b) -> double {
      return -g2 * pow(b, 2);
   };

   // Tabulate total Collins-Soper kernel
   const double Q    = 2;
   const int nb      = 100;
   const double bmin = 0.01;
   const double bmax = 1;
   const double bstp = (bmax - bmin) / (nb - 1);
   std::cout << std::scientific;
   for (double b = bmin; b <= bmax * 1.0000001; b += bstp) {
      // Build analytic solution
      const double mub    = 2 * exp(-apfel::emc) / bs(b, Q);
      const double as     = Alphas(mub) / apfel::FourPi;
      const int nf        = apfel::NF(mub, Thresholds);
      const double lambda = -2 * apfel::beta0qcd(nf) * as * log(Q / mub);
      double CSKernelAn   = gK0(nf, lambda);
      if (PerturbativeOrder > 0) CSKernelAn += as * gK1(nf, lambda);
      if (PerturbativeOrder > 1) CSKernelAn += pow(as, 2) * gK2(nf, lambda);

      std::cout << b << "\t" << -NPEvol(b) - CSKernel(bs(b, Q), Q) << "\t"
                << -NPEvol(b) - CSKernelAn << "\t" << std::endl;
   }

   return 0;
}
