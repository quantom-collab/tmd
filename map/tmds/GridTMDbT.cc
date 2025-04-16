//
// Authors: Matteo Cerutti: matteo.cerutti02@universitadipavia.it
//          Chiara Bissolotti: chiara.bissolotti01@gmail.com
//

#include "NangaParbat/fastinterface.h"
#include "NangaParbat/bstar.h"
#include "NangaParbat/nonpertfunctions.h"

#include <fstream>
#include <cstring>
#include <algorithm>
#include <iostream>
#include <cmath>
#include <vector>
#include <map>
#include <functional>

// Include APFEL++ and LHAPDF headers
#include <apfel/apfelxx.h>
#include <LHAPDF/LHAPDF.h>
#include <yaml-cpp/yaml.h>

//_________________________________________________________________________________
int main(int argc, char* argv[])
{
  // Check that the input is correct; 6 parameters are expected.
  if (argc < 7 || strcmp(argv[1], "--help") == 0)
    {
      std::cout << "\nInvalid Parameters:" << std::endl;
      std::cout << "Syntax: ./PlotTMDs_bT <configuration file> <output file> <pdf/ff> <flavour ID> <Scale in GeV> <parameters file>\n" << std::endl;
      exit(-10);
    }

  // Read the configuration file (config.yaml)
  const YAML::Node config = YAML::LoadFile(argv[1]);

  // Get the output file name
  const std::string output = std::string(argv[2]);

  // Distribution prefix (for example, "pdf" or "ff")
  const std::string pf = argv[3];

  // Open the LHAPDF set
  LHAPDF::PDF* dist = LHAPDF::mkPDF(config[pf + "set"]["name"].as<std::string>(), config[pf + "set"]["member"].as<int>());

  // Lambda to rotate sets into the QCD evolution basis
  const auto RotDists = [&] (double const& x, double const& mu) -> std::map<int,double>
  {
      return apfel::PhysToQCDEv(dist->xfxQ(x, mu));
  };

  // Heavy-quark thresholds
  std::vector<double> Thresholds;
  std::vector<double> bThresholds;
  const double Ci = config["TMDscales"]["Ci"].as<double>();
  for (auto const& v : dist->flavors())
    {
      if (v > 0 && v < 7)
        {
          Thresholds.push_back(dist->quarkThreshold(v));
          bThresholds.push_back(Ci * 2 * exp(- apfel::emc) / dist->quarkThreshold(v));
        }
    }
  std::sort(bThresholds.begin(), bThresholds.end());

  // Set the verbosity level of APFEL++ to the minimum
  apfel::SetVerbosityLevel(0);

  // Define the x-space grid: 100 logarithmically spaced x values between x_min and x_max.
  double x_min = 1e-5;
  double x_max = 0.99;
  int nx = 100;

  // Check if the configuration file contains a grid definition for the TMDs. 
  // If so, use it.
  if (config["tmdxgrid"].IsDefined())
    {
      nx = config["tmdxgrid"]["n"].as<int>();
      x_min = config["tmdxgrid"]["xmin"].as<double>();
      x_max = config["tmdxgrid"]["xmax"].as<double>();
    }

  std::vector<double> xvec(nx);
  for (int i = 0; i < nx; i++)
    {
      // Compute a normalized parameter 't' that varies linearly from 0 to 1.
      // Static_cast<double>(i) converts the integer i into a double, to ensure that 
      // the division is done in floating point, not integer arithmetic.
      // Without static_cast<double>(i), if both 'i' and 'nx - 1' are integers, 
      // the division would truncate to zero for most of the iterations, which is not desired.
      double t = static_cast<double>(i) / (nx - 1);

      // Compute xvec[i] using an exponential interpolation in log-space.
      // First, find the logarithms of x_min and x_max.
      // Then, interpolate linearly in the logarithmic domain: log(x[i]) = log(x_min) + t * (log(x_max) - log(x_min))
      // Finally, exponentiate to obtain the logarithmically spaced x value.
      xvec[i] = std::exp(std::log(x_min) + t * (std::log(x_max) - std::log(x_min)));
    }

  // Define the bT grid: 50 uniformly spaced points from bTmin to bTmax.
  double bTmin = 0.01;
  double bTmax = 4.0;
  int nbT = 50;

  // Check if the configuration file contains a grid definition for bT.
  // If so, use it.
  if (config["bTgrid"].IsDefined())
    {
      nbT = config["bTgrid"]["n"].as<int>();
      bTmin = config["bTgrid"]["bTmin"].as<double>();
      bTmax = config["bTgrid"]["bTmax"].as<double>();
    }

  // Create a vector to hold the bT values. 
  // The vector is filled with values from bTmin to bTmax, spaced evenly.
  std::vector<double> bTvec(nbT);
  double bTstep = (bTmax - bTmin) / (nbT - 1);
  for (int j = 0; j < nbT; j++)
    {
      bTvec[j] = bTmin + j * bTstep;
    }

  // Construct the x grid for APFEL++ from the configuration (this grid is used for the collinear distributions)
  std::vector<apfel::SubGrid> vsg;
  for (auto const& sg : config["xgrid" + pf])
      vsg.push_back({sg[0].as<int>(), sg[1].as<double>(), sg[2].as<int>()});
  const apfel::Grid g{vsg};

  // Lambda function to construct evolved distributions at scale mu
  const auto EvolvedDists = [=,&g] (double const& mu) -> apfel::Set<apfel::Distribution>
  {
      return apfel::Set<apfel::Distribution>{
          apfel::EvolutionBasisQCD{apfel::NF(mu, Thresholds)},
          DistributionMap(g, RotDists, mu)
      };
  };

  // Tabulate collinear distributions using APFEL++'s TabulateObject
  const apfel::TabulateObject<apfel::Set<apfel::Distribution>> TabDists{
      EvolvedDists, 100, dist->qMin() * 0.9, dist->qMax(), 3, Thresholds
  };

  // Build evolved TMD distributions
  const int pto = config["PerturbativeOrder"].as<int>();
  const auto Alphas = [&] (double const& mu) -> double { return dist->alphasQ(mu); };
  const auto CollDists = [&] (double const& mu) -> apfel::Set<apfel::Distribution>
  {
      return TabDists.Evaluate(mu);
  };

  std::function<apfel::Set<apfel::Distribution>(double const&, double const&, double const&)> EvTMDs;
  if (pf == "pdf")
    EvTMDs = BuildTmdPDFs(apfel::InitializeTmdObjects(g, Thresholds), CollDists, Alphas, pto, Ci);
  else if (pf == "ff" || pf == "ff2" || pf == "ff3" || pf == "ff4")
    EvTMDs = BuildTmdFFs(apfel::InitializeTmdObjects(g, Thresholds), CollDists, Alphas, pto, Ci);
  else
    throw std::runtime_error("[PlotTMDs]: Unknown distribution prefix");

  // b* prescription: lookup the b* function from NangaParbat
  const std::function<double(double const&, double const&)> bs = NangaParbat::bstarMap.at(config["bstar"].as<std::string>());

  // Flavour index from command line argument
  const int ifl = std::stoi(argv[4]);

  // Final scale Q and its square (input as a double from command line)
  const double Q  = std::stod(argv[5]);
  const double Q2 = Q * Q;

  // Read the parameters file for the non-perturbative parameterisation
  const YAML::Node parfile = YAML::LoadFile(argv[6]);
  NangaParbat::Parameterisation *NPFunc = NangaParbat::GetParametersation(parfile["Parameterisation"].as<std::string>());
  const std::vector<std::vector<double>> pars = parfile["Parameters"].as<std::vector<std::vector<double>>>();

  // For simplicity, use the first set of parameters. 
  // So this is ONLY replica 0.
  NPFunc->SetParameters(pars[0]);

  // Create a 2D table to store the TMD value for each (x, bT) pair.
  // tmdTable[i][j] will correspond to the value at xvec[i] and bTvec[j]
  // The table is initialized to zero.
  std::vector<std::vector<double>> tmdTable(nx, std::vector<double>(nbT, 0.0));

  // Define the function to compute the TMD at a given (x, bT).
  auto tmdFunc = [&](double x, double bT) -> double {
      return QCDEvToPhys(EvTMDs(bs(bT, Q), Q, Q2).GetObjects()).at(ifl).Evaluate(x)
             * NPFunc->Evaluate(x, bT, Q2, (pf == "pdf" ? 0 : 1));
  };

  // Loop over the x and bT grids, filling the TMD table.
  for (int i = 0; i < nx; i++)
      for (int j = 0; j < nbT; j++)
          tmdTable[i][j] = tmdFunc(xvec[i], bTvec[j]);
        

  // Write the 2D table to a plain text file, tab-delimited.
  // The first row is a header: first column is "x" and then the bT values.
  std::ofstream fout(output);
  if (!fout)
    {
      std::cerr << "Error: Unable to open output file: " << output << std::endl;
      return 1;
    }

  // Write header
  fout << "x" << "       "   << "bT" << "      "<< "TMD" << "\n";
  
  // Output each (x, bT, TMD) combination on its own line.
  for (int i = 0; i < nx; i++)
      for (int j = 0; j < nbT; j++)
          fout << xvec[i] << "\t" << bTvec[j] << "\t" << tmdTable[i][j] << "\n";
  
          fout.close();

  // Clean up: Delete the LHAPDF set
  delete dist;

  return 0;
}
