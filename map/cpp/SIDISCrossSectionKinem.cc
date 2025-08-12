/*
 * SIDIS Cross Section Computation
 * Author: Chiara Bissolotti: chiara.bissolotti01@gmail.com
 *
 * This program computes SIDIS differential cross sections using custom kinematics files
 * in YAML format, without relying on NangaParbat DataHandler.
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
#include <iomanip>

#include <NangaParbat/bstar.h>
#include <NangaParbat/nonpertfunctions.h>
#include <NangaParbat/createtmdgrid.h>

/**
 * @brief Custom data structure to hold kinematic data
 */
struct KinematicsData {
      std::string process;
      std::string observable;
      double target_isoscalarity;
      std::string hadron;
      int charge;
      double Vs;

      struct {
            double W;
            double ymin;
            double ymax;
      } PS_reduction;

      std::vector<double> PhT;
      std::vector<double> x;
      std::vector<double> z;
      std::vector<double> Q2;
      std::vector<double> y;

      size_t npoints() const
      {
         return PhT.size();
      }
};

/**
 * @brief Load kinematics data from YAML file
 */
KinematicsData loadKinematicsFile(const std::string &filename)
{
   YAML::Node data = YAML::LoadFile(filename);

   KinematicsData kin;

   // Load header information
   const auto &header      = data["header"];
   kin.process             = header["process"].as<std::string>();
   kin.observable          = header["observable"].as<std::string>();
   kin.target_isoscalarity = header["target_isoscalarity"].as<double>();
   kin.hadron              = header["hadron"].as<std::string>();
   kin.charge              = header["charge"].as<int>();
   kin.Vs                  = header["Vs"].as<double>();

   // Load phase space cuts
   const auto &ps        = header["PS_reduction"];
   kin.PS_reduction.W    = ps["W"].as<double>();
   kin.PS_reduction.ymin = ps["ymin"].as<double>();
   kin.PS_reduction.ymax = ps["ymax"].as<double>();

   // Load kinematic arrays
   const auto &dataNode = data["data"];
   kin.PhT              = dataNode["PhT"].as<std::vector<double>>();
   kin.x                = dataNode["x"].as<std::vector<double>>();
   kin.z                = dataNode["z"].as<std::vector<double>>();
   kin.Q2               = dataNode["Q2"].as<std::vector<double>>();
   kin.y                = dataNode["y"].as<std::vector<double>>();

   // Validate that all arrays have the same size
   size_t n = kin.PhT.size();
   if (kin.x.size() != n || kin.z.size() != n || kin.Q2.size() != n || kin.y.size() != n) {
      throw std::runtime_error("Kinematic arrays must have the same size");
   }

   return kin;
}

/**
 * @brief Save results to YAML file in detailed format (original)
 */
void saveResultsYAML(const std::string &filename, const KinematicsData &kin,
                     const std::vector<double> &predictions)
{
   YAML::Emitter out;

   out << YAML::BeginMap;
   out << YAML::Key << "Process" << YAML::Value << kin.process;
   out << YAML::Key << "Observable" << YAML::Value << kin.observable;
   out << YAML::Key << "Hadron" << YAML::Value << kin.hadron;
   out << YAML::Key << "Charge" << YAML::Value << kin.charge;
   out << YAML::Key << "Target_isoscalarity" << YAML::Value << kin.target_isoscalarity;
   out << YAML::Key << "Vs" << YAML::Value << kin.Vs;

   out << YAML::Key << "Kinematics" << YAML::Value << YAML::BeginSeq;
   for (size_t i = 0; i < kin.npoints(); i++) {
      out << YAML::BeginMap;
      out << YAML::Key << "point" << YAML::Value << i + 1;
      out << YAML::Key << "PhT" << YAML::Value << kin.PhT[i];
      out << YAML::Key << "x" << YAML::Value << kin.x[i];
      out << YAML::Key << "z" << YAML::Value << kin.z[i];
      out << YAML::Key << "Q2" << YAML::Value << kin.Q2[i];
      out << YAML::Key << "Q" << YAML::Value << std::sqrt(kin.Q2[i]);
      out << YAML::Key << "y" << YAML::Value << kin.y[i];
      out << YAML::Key << "qT" << YAML::Value << kin.PhT[i] / kin.z[i];
      out << YAML::Key << "cross_section" << YAML::Value << predictions[i];
      out << YAML::EndMap;
   }
   out << YAML::EndSeq;
   out << YAML::EndMap;

   std::ofstream file(filename);
   file << out.c_str();
   file.close();
}

/**
 * @brief Save results to YAML file in array format for plotting
 */
void saveResultsYAMLArrays(const std::string &filename, const KinematicsData &kin,
                           const std::vector<double> &predictions)
{
   YAML::Emitter out;

   // Calculate derived quantities - all vectors same length as predictions
   std::vector<double> Q_values, qT_values;
   for (size_t i = 0; i < kin.npoints(); i++) {
      Q_values.push_back(std::sqrt(kin.Q2[i]));
      qT_values.push_back(kin.PhT[i] / kin.z[i]);
   }

   // Calculate average values for metadata
   double Q_avg = std::accumulate(Q_values.begin(), Q_values.end(), 0.0) / Q_values.size();
   double x_avg = std::accumulate(kin.x.begin(), kin.x.end(), 0.0) / kin.x.size();
   double z_avg = std::accumulate(kin.z.begin(), kin.z.end(), 0.0) / kin.z.size();

   out << YAML::BeginMap;
   out << YAML::Key << "Name" << YAML::Value << kin.process + "_" + kin.observable;
   out << YAML::Key << "Q" << YAML::Value << Q_avg;
   out << YAML::Key << "x" << YAML::Value << x_avg;
   out << YAML::Key << "z" << YAML::Value << z_avg;

   // All arrays have exactly the same length as predictions
   out << YAML::Key << "PhT" << YAML::Value << YAML::Flow << YAML::BeginSeq;
   for (size_t i = 0; i < kin.PhT.size(); i++)
      out << kin.PhT[i];
   out << YAML::EndSeq;

   out << YAML::Key << "x_values" << YAML::Value << YAML::Flow << YAML::BeginSeq;
   for (size_t i = 0; i < kin.x.size(); i++)
      out << kin.x[i];
   out << YAML::EndSeq;

   out << YAML::Key << "z_values" << YAML::Value << YAML::Flow << YAML::BeginSeq;
   for (size_t i = 0; i < kin.z.size(); i++)
      out << kin.z[i];
   out << YAML::EndSeq;

   out << YAML::Key << "Q2" << YAML::Value << YAML::Flow << YAML::BeginSeq;
   for (size_t i = 0; i < kin.Q2.size(); i++)
      out << kin.Q2[i];
   out << YAML::EndSeq;

   out << YAML::Key << "Q_values" << YAML::Value << YAML::Flow << YAML::BeginSeq;
   for (size_t i = 0; i < Q_values.size(); i++)
      out << Q_values[i];
   out << YAML::EndSeq;

   out << YAML::Key << "y" << YAML::Value << YAML::Flow << YAML::BeginSeq;
   for (size_t i = 0; i < kin.y.size(); i++)
      out << kin.y[i];
   out << YAML::EndSeq;

   out << YAML::Key << "qT" << YAML::Value << YAML::Flow << YAML::BeginSeq;
   for (size_t i = 0; i < qT_values.size(); i++)
      out << qT_values[i];
   out << YAML::EndSeq;

   out << YAML::Key << "Predictions" << YAML::Value << YAML::Flow << YAML::BeginSeq;
   for (size_t i = 0; i < predictions.size(); i++)
      out << predictions[i];
   out << YAML::EndSeq;

   out << YAML::EndMap;

   std::ofstream file(filename);
   file << out.c_str();
   file.close();
}

/**
 * @brief Save results to text file with columns
 */
void saveResultsTXT(const std::string &filename, const KinematicsData &kin,
                    const std::vector<double> &predictions)
{
   std::ofstream file(filename);

   // Header
   file << "# SIDIS Cross Section Results\n";
   file << "# Process: " << kin.process << ", Observable: " << kin.observable << "\n";
   file << "# Hadron: " << kin.hadron << ", Charge: " << kin.charge << "\n";
   file << "# Target isoscalarity: " << kin.target_isoscalarity << "\n";
   file << "# Vs: " << kin.Vs << " GeV^2\n";
   file << "#\n";
   file << std::setw(8) << "# point" << std::setw(15) << "PhT[GeV]" << std::setw(15) << "x"
        << std::setw(15) << "z" << std::setw(15) << "Q2[GeV^2]" << std::setw(15) << "Q[GeV]"
        << std::setw(15) << "y" << std::setw(15) << "qT[GeV]" << std::setw(20)
        << "cross_section[GeV^-2]"
        << "\n";

   // Data
   file << std::scientific << std::setprecision(6);
   for (size_t i = 0; i < kin.npoints(); i++) {
      file << std::setw(8) << i + 1 << std::setw(15) << kin.PhT[i] << std::setw(15) << kin.x[i]
           << std::setw(15) << kin.z[i] << std::setw(15) << kin.Q2[i] << std::setw(15)
           << std::sqrt(kin.Q2[i]) << std::setw(15) << kin.y[i] << std::setw(15)
           << kin.PhT[i] / kin.z[i] << std::setw(20) << predictions[i] << "\n";
   }

   file.close();
}

// Main program
int main(int argc, char *argv[])
{
   // Check command line arguments
   if (argc < 4 || strcmp(argv[1], "--help") == 0) {
      std::cout << "\nUsage:" << std::endl;
      std::cout << "Syntax: ./SIDISCrossSection <config.yaml> <kinematics.yaml> <output_folder>\n"
                << std::endl;
      std::cout << "Arguments:" << std::endl;
      std::cout << "  config.yaml     - SIDIS computation configuration file" << std::endl;
      std::cout << "  kinematics.yaml - Kinematic points file" << std::endl;
      std::cout << "  output_folder   - Output directory for results" << std::endl;
      exit(-10);
   }

   // Read configuration file
   YAML::Node config = YAML::LoadFile(argv[1]);

   // Load kinematics data
   std::string kinematicsFile = argv[2];
   KinematicsData kin;

   try {
      kin = loadKinematicsFile(kinematicsFile);
      std::cout << "\033[1;32mLoaded " << kin.npoints() << " kinematic points from "
                << kinematicsFile << "\033[0m" << std::endl;
   } catch (const std::exception &e) {
      std::cerr << "\033[1;31mError loading kinematics file: " << e.what() << "\033[0m"
                << std::endl;
      exit(-1);
   }

   // Output folder
   const std::string OutputFolder = std::string(argv[3]);
   mkdir(OutputFolder.c_str(), ACCESSPERMS);
   std::cout << "\033[1;32mCreating folder '" + OutputFolder + "' to store the output.\033[0m"
             << std::endl;

   std::cout << "\033[1;37m\nComputing SIDIS cross sections ...\033[0m" << std::endl;
   std::cout << "\033[1;33mProcess: " << kin.process << ", Observable: " << kin.observable
             << "\033[0m" << std::endl;
   std::cout << "\033[1;33mHadron: " << kin.hadron << ", Charge: " << kin.charge << "\033[0m"
             << std::endl;

   // Perturbative order
   const int pto = config["PerturbativeOrder"].as<int>();
   std::cout << "\033[1;32mPerturbative order: " << NangaParbat::PtOrderMap.at(pto) << "\033[0m"
             << std::endl;

   // Open LHAPDF sets
   LHAPDF::PDF *distpdf = LHAPDF::mkPDF(config["pdfset"]["name"].as<std::string>(),
                                        config["pdfset"]["member"].as<int>());

   // Rotate PDF set into the QCD evolution basis
   const auto RotPDFs = [=](double const &x, double const &mu) -> std::map<int, double> {
      return apfel::PhysToQCDEv(distpdf->xfxQ(x, mu));
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

   // Set cut
   const double qToQcut = 3;

   // Electromagnetic coupling squared
   const double aref = config["alphaem"]["aref"].as<double>();
   apfel::AlphaQED alphaem{
       aref, config["alphaem"]["Qref"].as<double>(), Thresholds, {0, 0, 1.777}, 0};
   const apfel::TabulateObject<double> TabAlphaem{alphaem, 100, 0.9, 1001, 3};

   // Construct set of distributions as a function of the scale to be tabulated
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
   const auto MatchTMDPDFs = MatchTmdPDFs(
       TmdObjPDF, CollPDFs,
       [&](double const &mu) -> double {
          return TabAlphas.Evaluate(mu);
       },
       pto, Ci);

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

   // Open LHAPDF FFs sets
   std::string FFset   = config["ffset"]["name"].as<std::string>();
   LHAPDF::PDF *distff = LHAPDF::mkPDF(FFset, config["ffset"]["member"].as<int>());

   // Rotate FF set into the QCD evolution basis
   const auto RotFFs = [=](double const &x, double const &mu) -> std::map<int, double> {
      return apfel::PhysToQCDEv(distff->xfxQ(x, mu));
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
   const auto MatchTMDFFs = MatchTmdFFs(
       TmdObjFF, CollFFs,
       [&](double const &mu) -> double {
          return TabAlphas.Evaluate(mu);
       },
       pto, Ci);

   // Target isoscalarity handling
   const double targetiso = kin.target_isoscalarity;
   const int sign         = (targetiso >= 0 ? 1 : -1);
   const double frp       = std::abs(targetiso);
   const double frn       = 1 - frp;

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
       [](double const &x) -> double {
          return log(x);
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
       [](double const &x) -> double {
          return log(x);
       },
       [](double const &y) -> double {
          return exp(y);
       }};

   // Functions used for the tabulation
   const auto TabFunc = [](double const &b) -> double {
      return log(b);
   };
   const auto InvTabFunc = [](double const &fb) -> double {
      return exp(fb);
   };

   // Vector to store predictions
   std::vector<double> predictions(kin.npoints(), 0.0);

   std::cout << "\033[1;33mComputing cross sections for " << kin.npoints()
             << " kinematic points...\033[0m" << std::endl;

   // Loop over all kinematic points
   for (size_t ipt = 0; ipt < kin.npoints(); ipt++) {
      std::cout << "\033[1;36mPoint " << (ipt + 1) << "/" << kin.npoints() << "\033[0m"
                << std::endl;

      // Extract kinematics for this point
      const double PhT = kin.PhT[ipt];
      const double x   = kin.x[ipt];
      const double z   = kin.z[ipt];
      const double Q   = std::sqrt(kin.Q2[ipt]);
      const double y   = kin.y[ipt];
      const double qT  = PhT / z;

      std::cout << "  x=" << x << ", z=" << z << ", Q=" << Q << ", PhT=" << PhT << ", qT=" << qT
                << std::endl;

      // Check if qT > cut
      if (qT > qToQcut * Q) {
         std::cout << "  \033[1;31mSkipping: qT/Q = " << qT / Q << " > " << qToQcut << "\033[0m"
                   << std::endl;
         predictions[ipt] = 0.0;
         continue;
      }

      // Scales
      const double mu   = Cf * Q;
      const double zeta = Q * Q;

      const double Yp = 1 + pow(1 - pow(Q / kin.Vs, 2) / x, 2);

      // Number of active flavors
      const int nf = apfel::NF(mu, Thresholds);

      // Non-perturbative functions
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

            const double lumibsq = Yp * TabMatchTMDPDFs.EvaluatexQ(q, x, bs) / x *
                                   apfel::QCh2[std::abs(q) - 1] *
                                   TabMatchTMDFFs.EvaluatexQ(q, z, bs);
            Lumiq += lumibsq;
         }

         return b * tf1NP.Evaluate(b) * tf2NP.Evaluate(b) * Lumiq *
                pow(QuarkSudakov(bs, mu, zeta), 2) / z * pow(TabAlphaem.Evaluate(Q), 2) * Hf(mu) /
                pow(Q, 3);
      };

      // Compute the differential cross section
      const double crossSection =
          apfel::ConvFact * apfel::FourPi * qT * DEObj.transform(bIntegrand, qT) / (2 * Q) / z;

      predictions[ipt] = crossSection;

      std::cout << "  \033[1;32mResult: " << std::scientific << crossSection << " GeV^-2\033[0m"
                << std::endl;
   }

   // Save results
   std::string yamlOutput       = OutputFolder + "/predictions.yaml";
   std::string yamlArraysOutput = OutputFolder + "/predictions_arrays.yaml";
   std::string txtOutput        = OutputFolder + "/predictions.txt";

   saveResultsYAML(yamlOutput, kin, predictions);
   saveResultsYAMLArrays(yamlArraysOutput, kin, predictions);
   saveResultsTXT(txtOutput, kin, predictions);

   std::cout << "\033[1;32m\nResults saved to:\033[0m" << std::endl;
   std::cout << "  YAML (detailed): " << yamlOutput << std::endl;
   std::cout << "  YAML (arrays):   " << yamlArraysOutput << std::endl;
   std::cout << "  TXT:             " << txtOutput << std::endl;

   // Cleanup
   delete distpdf;
   delete distff;
   delete fNP;

   return 0;
}
