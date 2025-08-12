/*
 * SIDIS Cross Section Computation
 *
 * Authors: Valerio Bertone: valerio.bertone@cern.ch
 *          Chiara Bissolotti: chiara.bissolotti01@gmail.com
 *
 * This program computes the SIDIS differential cross section (numerator only)
 * using TMD factorization framework with APFEL++, LHAPDF, and NangaParbat.
 */

#include <LHAPDF/LHAPDF.h>
#include <apfel/apfelxx.h>
#include <yaml-cpp/yaml.h>
#include <cstring>
#include <functional>
#include <algorithm>
#include <sys/stat.h>
#include <fstream>
#include <numeric>
#include <iostream>
#include <vector>
#include <map>
#include <cmath>

#include <NangaParbat/bstar.h>
#include <NangaParbat/nonpertfunctions.h>
#include <NangaParbat/createtmdgrid.h>
#include <NangaParbat/datahandler.h>

/**
 * @brief Compute the differential SIDIS cross section
 * @param config YAML configuration
 * @param x Bjorken x value
 * @param z Fragmentation variable z
 * @param Q Hard scale Q
 * @param qT Transverse momentum qT
 * @param target_iso Target isoscalarity factor
 * @param ff_set Fragmentation function set name
 * @return Differential cross section value
 */
double computeDifferentialCrossSection(const YAML::Node &config, double x, double z, double Q,
                                       double qT, double target_iso = 1.0,
                                       const std::string &ff_set = "DSS14_NLO_Pip")
{
   // Perturbative order
   const int pto = config["PerturbativeOrder"].as<int>();

   // Open LHAPDF PDF set
   LHAPDF::PDF *distpdf = LHAPDF::mkPDF(config["pdfset"]["name"].as<std::string>(),
                                        config["pdfset"]["member"].as<int>());

   // Rotate PDF set into the QCD evolution basis
   const auto RotPDFs = [=](double const &xval, double const &mu) -> std::map<int, double> {
      return apfel::PhysToQCDEv(distpdf->xfxQ(xval, mu));
   };

   // Get heavy-quark thresholds from the PDF LHAPDF set
   std::vector<double> Thresholds;
   for (auto const &v : distpdf->flavors())
      if (v > 0 && v < 7) Thresholds.push_back(distpdf->quarkThreshold(v));

   // Alpha_s tabulation
   const apfel::TabulateObject<double> TabAlphas{[&](double const &mu) -> double {
                                                    return distpdf->alphasQ(mu);
                                                 },
                                                 100,
                                                 distpdf->qMin() * 0.9,
                                                 distpdf->qMax(),
                                                 3,
                                                 Thresholds};

   // Setup APFEL++ x-space grid for PDFs
   std::vector<apfel::SubGrid> vsgp;
   for (auto const &sg : config["xgridpdf"])
      vsgp.push_back({sg[0].as<int>(), sg[1].as<double>(), sg[2].as<int>()});
   const apfel::Grid gpdf{vsgp};

   // Scale-variation factors
   const double Ci = config["TMDscales"]["Ci"].as<double>();
   const double Cf = config["TMDscales"]["Cf"].as<double>();

   // Get non-perturbative functions
   const NangaParbat::Parameterisation *fNP = NangaParbat::GetParametersation("PV17");

   // Electromagnetic coupling squared
   const double aref = config["alphaem"]["aref"].as<double>();
   apfel::AlphaQED alphaem{
       aref, config["alphaem"]["Qref"].as<double>(), Thresholds, {0, 0, 1.777}, 0};
   const apfel::TabulateObject<double> TabAlphaem{alphaem, 100, 0.9, 1001, 3};

   // Construct set of distributions as a function of the scale
   const auto EvolvedPDFs = [=, &gpdf](double const &mu) -> apfel::Set<apfel::Distribution> {
      return apfel::Set<apfel::Distribution>{apfel::EvolutionBasisQCD{apfel::NF(mu, Thresholds)},
                                             DistributionMap(gpdf, RotPDFs, mu)};
   };

   // Tabulate collinear PDFs
   const apfel::TabulateObject<apfel::Set<apfel::Distribution>> TabPDFs{
       EvolvedPDFs, 100, distpdf->qMin() * 0.9, distpdf->qMax(), 3, Thresholds};
   const auto CollPDFs = [&](double const &mu) -> apfel::Set<apfel::Distribution> {
      return TabPDFs.Evaluate(mu);
   };

   // Initialize TMD PDF objects
   const auto TmdObjPDF = apfel::InitializeTmdObjects(gpdf, Thresholds);

   // Build evolved TMD PDFs
   const auto EvTMDPDFs = BuildTmdPDFs(
       TmdObjPDF, CollPDFs,
       [&](double const &mu) -> double {
          return TabAlphas.Evaluate(mu);
       },
       pto, Ci);
   const auto MatchTMDPDFs = MatchTmdPDFs(
       TmdObjPDF, CollPDFs,
       [&](double const &mu) -> double {
          return TabAlphas.Evaluate(mu);
       },
       pto, Ci);

   // Open LHAPDF FF set
   LHAPDF::PDF *distff = LHAPDF::mkPDF(ff_set, config["ffset"]["member"].as<int>());

   // Rotate FF set into the QCD evolution basis
   const auto RotFFs = [=](double const &xval, double const &mu) -> std::map<int, double> {
      return apfel::PhysToQCDEv(distff->xfxQ(xval, mu));
   };

   // Setup APFEL++ x-space grid for FFs
   std::vector<apfel::SubGrid> vsgf;
   for (auto const &sg : config["xgridff"])
      vsgf.push_back({sg[0].as<int>(), sg[1].as<double>(), sg[2].as<int>()});
   const apfel::Grid gff{vsgf};

   // Construct FF distributions
   const auto EvolvedFFs = [=, &gff](double const &mu) -> apfel::Set<apfel::Distribution> {
      return apfel::Set<apfel::Distribution>{apfel::EvolutionBasisQCD{apfel::NF(mu, Thresholds)},
                                             DistributionMap(gff, RotFFs, mu)};
   };

   // Tabulate collinear FFs
   const apfel::TabulateObject<apfel::Set<apfel::Distribution>> TabFFs{
       EvolvedFFs, 200, distff->qMin() * 0.9, distff->qMax(), 3, Thresholds};
   const auto CollFFs = [&](double const &mu) -> apfel::Set<apfel::Distribution> {
      return TabFFs.Evaluate(mu);
   };

   // Initialize TMD FF objects
   const auto TmdObjFF = apfel::InitializeTmdObjects(gff, Thresholds);

   // Build evolved TMD FFs
   const auto EvTMDFFs = BuildTmdFFs(
       TmdObjFF, CollFFs,
       [&](double const &mu) -> double {
          return TabAlphas.Evaluate(mu);
       },
       pto, Ci);
   const auto MatchTMDFFs = MatchTmdFFs(
       TmdObjFF, CollFFs,
       [&](double const &mu) -> double {
          return TabAlphas.Evaluate(mu);
       },
       pto, Ci);

   // Evolution factor
   auto QuarkSudakov = QuarkEvolutionFactor(
       TmdObjPDF,
       [&](double const &mu) -> double {
          return TabAlphas.Evaluate(mu);
       },
       pto, Ci, 1e5);

   // Get hard-factor
   const std::function<double(double const &)> Hf = apfel::HardFactor(
       "SIDIS", TmdObjPDF,
       [&](double const &mu) -> double {
          return TabAlphas.Evaluate(mu);
       },
       pto, Cf);

   // Target isoscalarity handling
   const int sign   = (target_iso >= 0 ? 1 : -1);
   const double frp = std::abs(target_iso);
   const double frn = 1 - frp;

   // Tabulate initial scale TMD PDFs in b in the physical basis
   std::function<apfel::Set<apfel::Distribution>(double const &)> isTMDPDFs =
       [&](double const &b) -> apfel::Set<apfel::Distribution> {
      const apfel::Set<apfel::Distribution> xF = QCDEvToPhys(MatchTMDPDFs(b).GetObjects());
      std::map<int, apfel::Distribution> xFiso;

      // Apply isoscalarity correction
      xFiso.insert({1, frp * xF.at(sign) + frn * xF.at(sign * 2)});
      xFiso.insert({-1, frp * xF.at(-sign) + frn * xF.at(-sign * 2)});
      xFiso.insert({2, frp * xF.at(sign * 2) + frn * xF.at(sign)});
      xFiso.insert({-2, frp * xF.at(-sign * 2) + frn * xF.at(-sign)});

      // Other flavors
      for (int i = 3; i <= 6; i++) {
         const int ip = i * sign;
         xFiso.insert({i, xF.at(ip)});
         xFiso.insert({-i, xF.at(-ip)});
      }
      return apfel::Set<apfel::Distribution>{xFiso};
   };

   const apfel::TabulateObject<apfel::Set<apfel::Distribution>> TabMatchTMDPDFs{
       isTMDPDFs,
       200,
       1e-2,
       2,
       1,
       {},
       [](double const &xval) -> double {
          return log(xval);
       },
       [](double const &y) -> double {
          return exp(y);
       }};

   // Tabulate initial scale TMD FFs
   std::function<apfel::Set<apfel::Distribution>(double const &)> isTMDFFs =
       [&](double const &b) -> apfel::Set<apfel::Distribution> {
      return apfel::Set<apfel::Distribution>{QCDEvToPhys(MatchTMDFFs(b).GetObjects())};
   };

   const apfel::TabulateObject<apfel::Set<apfel::Distribution>> TabMatchTMDFFs{
       isTMDFFs,
       200,
       1e-2,
       2,
       1,
       {},
       [](double const &xval) -> double {
          return log(xval);
       },
       [](double const &y) -> double {
          return exp(y);
       }};

   // Computation of the cross section
   const double mu   = Cf * Q;
   const double zeta = Q * Q;

   // Non-perturbative functions
   const auto TabFunc = [](double const &b) -> double {
      return log(b);
   };
   const auto InvTabFunc = [](double const &fb) -> double {
      return exp(fb);
   };

   const apfel::TabulateObject<double> tf1NP{[=](double const &tb) -> double {
                                                return fNP->Evaluate(x, tb, zeta, 0);
                                             },
                                             100,
                                             5e-5,
                                             5,
                                             3,
                                             {},
                                             TabFunc,
                                             InvTabFunc};
   const apfel::TabulateObject<double> tf2NP{[=](double const &tb) -> double {
                                                return fNP->Evaluate(z, tb, zeta, 1);
                                             },
                                             100,
                                             5e-5,
                                             5,
                                             3,
                                             {},
                                             TabFunc,
                                             InvTabFunc};

   // Number of active flavors
   const int nf = apfel::NF(mu, Thresholds);

   // Ogata quadrature for b-integration
   apfel::OgataQuadrature DEObj{0, 1e-9, 0.00001};

   // b-integration kernel
   const std::function<double(double const &)> bIntegrand =
       [=, &TabMatchTMDPDFs, &TabMatchTMDFFs, &tf1NP, &tf2NP, &QuarkSudakov, &TabAlphaem,
        &Hf](double const &b) -> double {
      // bstar prescription
      auto bs = NangaParbat::bstarmin(b, Q);

      // Sum contributions from active flavors
      double Lumiq = 0;
      for (int q = -nf; q <= nf; q++) {
         if (q == 0) continue; // Skip gluon

         const double lumibsq = TabMatchTMDPDFs.EvaluatexQ(q, x, bs) / x *
                                apfel::QCh2[std::abs(q) - 1] * TabMatchTMDFFs.EvaluatexQ(q, z, bs);
         Lumiq += lumibsq;
      }

      return b * tf1NP.Evaluate(b) * tf2NP.Evaluate(b) * Lumiq *
             pow(QuarkSudakov(bs, mu, zeta), 2) / z * pow(TabAlphaem.Evaluate(Q), 2) * Hf(mu) /
             pow(Q, 3);
   };

   // Compute the differential cross section
   const double crossSection =
       apfel::ConvFact * apfel::FourPi * qT * DEObj.transform(bIntegrand, qT) / (2 * Q) / z;

   // Cleanup
   delete distpdf;
   delete distff;
   delete fNP;

   return crossSection;
}

int main(int argc, char *argv[])
{
   // Check command line arguments
   if (argc < 2 || strcmp(argv[1], "--help") == 0) {
      std::cout << "\nUsage:" << std::endl;
      std::cout << "Syntax: ./SIDISCrossSection <YAML configuration file> [options]\n" << std::endl;
      std::cout << "Options:" << std::endl;
      std::cout << "  --x <value>        Bjorken x value (default: 0.1)" << std::endl;
      std::cout << "  --z <value>        Fragmentation variable z (default: 0.5)" << std::endl;
      std::cout << "  --Q <value>        Hard scale Q in GeV (default: 2.0)" << std::endl;
      std::cout << "  --qT <value>       Transverse momentum qT in GeV (default: 1.0)" << std::endl;
      std::cout << "  --target-iso <val> Target isoscalarity factor (default: 1.0)" << std::endl;
      std::cout << "  --ff-set <name>    Fragmentation function set (default: DSS14_NLO_Pip)"
                << std::endl;
      std::cout << "  --output <file>    Output file (default: stdout)" << std::endl;
      return -1;
   }

   // Default values
   double x                = 0.1;
   double z                = 0.5;
   double Q                = 2.0;
   double qT               = 1.0;
   double target_iso       = 1.0;
   std::string ff_set      = "DSS14_NLO_Pip";
   std::string output_file = "";

   // Parse command line arguments
   for (int i = 2; i < argc; i += 2) {
      if (i + 1 >= argc) break;

      std::string arg = argv[i];
      if (arg == "--x") x = std::stod(argv[i + 1]);
      else if (arg == "--z") z = std::stod(argv[i + 1]);
      else if (arg == "--Q") Q = std::stod(argv[i + 1]);
      else if (arg == "--qT") qT = std::stod(argv[i + 1]);
      else if (arg == "--target-iso") target_iso = std::stod(argv[i + 1]);
      else if (arg == "--ff-set") ff_set = argv[i + 1];
      else if (arg == "--output") output_file = argv[i + 1];
   }

   try {
      // Load configuration
      YAML::Node config = YAML::LoadFile(argv[1]);

      std::cout << "\033[1;37m\nComputing SIDIS differential cross section...\033[0m" << std::endl;
      std::cout << "\033[1;33mParameters:\033[0m" << std::endl;
      std::cout << "  x = " << x << std::endl;
      std::cout << "  z = " << z << std::endl;
      std::cout << "  Q = " << Q << " GeV" << std::endl;
      std::cout << "  qT = " << qT << " GeV" << std::endl;
      std::cout << "  Target isoscalarity = " << target_iso << std::endl;
      std::cout << "  FF set = " << ff_set << std::endl;

      // Compute cross section
      double result = computeDifferentialCrossSection(config, x, z, Q, qT, target_iso, ff_set);

      // Output results
      if (output_file.empty()) {
         std::cout << "\033[1;32m\nResult:\033[0m" << std::endl;
         std::cout << "Differential cross section: " << std::scientific << result << " GeV^-2"
                   << std::endl;
      } else {
         std::ofstream outfile(output_file);
         outfile << "# SIDIS Differential Cross Section Computation" << std::endl;
         outfile << "# x = " << x << ", z = " << z << ", Q = " << Q << " GeV, qT = " << qT << " GeV"
                 << std::endl;
         outfile << "# Target isoscalarity = " << target_iso << ", FF set = " << ff_set
                 << std::endl;
         outfile << std::scientific << result << std::endl;
         outfile.close();
         std::cout << "\033[1;32mResult written to: " << output_file << "\033[0m" << std::endl;
      }

   } catch (const std::exception &e) {
      std::cerr << "\033[1;31mError: " << e.what() << "\033[0m" << std::endl;
      return -1;
   }

   return 0;
}
