/*
 * Authors: Valerio Bertone: valerio.bertone@cern.ch
 *          Chiara Bissolotti: chiara.bissolotti01@gmail.com
 */

#if 0 // Guard to avoid vscode intellisense to show errors

#include <LHAPDF/LHAPDF.h>
#include <apfel/apfelxx.h>
#include <yaml-cpp/yaml.h>
#include <cstring>
#include <functional> // std::divides
#include <algorithm>  // std::transform
#include <sys/stat.h>
#include <fstream>
#include <numeric> // std::accumulate

#include <NangaParbat/bstar.h>
#include <NangaParbat/nonpertfunctions.h>
#include <NangaParbat/createtmdgrid.h>
#include <NangaParbat/datahandler.h>

// Main program
int main(int argc, char* argv[])
{
  // Check that the input is correct otherwise stop the code
  if (argc < 4 || strcmp(argv[1], "--help") == 0)
    {
      std::cout << "\nUsage:" << std::endl;
      std::cout << "Syntax: ./SIDISMultiplicities <YAML configuration file [config.yaml]> <path to data folder> <output folder>\n" << std::endl;
      exit(-10);
    }

  // Read configuration file
  YAML::Node config = YAML::LoadFile(argv[1]);

  // Output folder
  const std::string OutputFolder = std::string(argv[3]);
  mkdir(OutputFolder.c_str(), ACCESSPERMS);
  std::cout << "Creating folder \'" + OutputFolder + "\' to store the output.\n" << std::endl;

  std::cout << "\033[1;37m\nComputing SIDIS multiplicities ...\033[0m" << std::endl;
  std::cout << "\033[1;33mComputation done at the average values, NOT INTEGRATED.\033[0m" << std::endl;

  // Perturbative order
  const int pto = config["PerturbativeOrder"].as<int>();
  std::cout << "\033[1;32mPerturbative order: " << NangaParbat::PtOrderMap.at(pto) << "\n\033[0m" << std::endl;

  // Open LHAPDF sets
  LHAPDF::PDF* distpdf = LHAPDF::mkPDF(config["pdfset"]["name"].as<std::string>(), config["pdfset"]["member"].as<int>());

  // Rotate PDF set into the QCD evolution basis
  const auto RotPDFs = [=] (double const& x, double const& mu) -> std::map<int,double> { return apfel::PhysToQCDEv(distpdf->xfxQ(x, mu)); };

  // Get heavy-quark thresholds from the PDF LHAPDF set
  std::vector<double> Thresholds;
  for (auto const& v : distpdf->flavors())
    if (v > 0 && v < 7)
      Thresholds.push_back(distpdf->quarkThreshold(v));

  // Alpha_s (from PDFs). Get it from the LHAPDF set and tabulate it.
  const auto Alphas = [&] (double const& mu) -> double{ return distpdf->alphasQ(mu); };
  const apfel::TabulateObject<double> TabAlphas {[&] (double const& mu) -> double{return distpdf->alphasQ(mu); },
                                                 100, distpdf->qMin() * 0.9, distpdf->qMax(), 3, Thresholds};

  // Setup APFEL++ x-space grid for PDFs
  std::vector<apfel::SubGrid> vsgp;
  for (auto const& sg : config["xgridpdf"])
    vsgp.push_back({sg[0].as<int>(), sg[1].as<double>(), sg[2].as<int>()});
  const apfel::Grid gpdf{vsgp};

  // Scale-variation factors
  const double Ci = config["TMDscales"]["Ci"].as<double>();
  const double Cf = config["TMDscales"]["Cf"].as<double>();

  // Get non-perturbative functions
  const NangaParbat::Parameterisation* fNP = NangaParbat::GetParametersation("PV17");

  // Set cut
  const double qToQcut = 3;

  // Electromagnetic coupling squared (provided by APFEL++)
  const double aref = config["alphaem"]["aref"].as<double>();
  apfel::AlphaQED alphaem{aref, config["alphaem"]["Qref"].as<double>(), Thresholds, {0, 0, 1.777}, 0};
  const apfel::TabulateObject<double> TabAlphaem{alphaem, 100, 0.9, 1001, 3};

  // Construct set of distributions as a function of the scale to be tabulated
  const auto EvolvedPDFs = [=,&gpdf] (double const& mu) -> apfel::Set<apfel::Distribution>
  {
    return apfel::Set<apfel::Distribution>{apfel::EvolutionBasisQCD{apfel::NF(mu, Thresholds)}, DistributionMap(gpdf, RotPDFs, mu)};
  };

  // Tabulate collinear PDFs
  const apfel::TabulateObject<apfel::Set<apfel::Distribution>> TabPDFs{EvolvedPDFs, 100, distpdf->qMin() * 0.9, distpdf->qMax(), 3, Thresholds};
  const auto CollPDFs = [&] (double const& mu) -> apfel::Set<apfel::Distribution> { return TabPDFs.Evaluate(mu); };

  // Initialize TMD PDF objects
  const auto TmdObjPDF = apfel::InitializeTmdObjects(gpdf, Thresholds);

  // Build evolved TMD PDFs
  const auto EvTMDPDFs    = BuildTmdPDFs(TmdObjPDF, CollPDFs, Alphas, pto, Ci);
  const auto MatchTMDPDFs = MatchTmdPDFs(TmdObjPDF, CollPDFs, Alphas, pto, Ci);

  auto QuarkSudakov = QuarkEvolutionFactor(TmdObjPDF, Alphas, pto, Ci, 1e5);

  // Get hard-factor
  const std::function<double(double const&)> Hf = apfel::HardFactor("SIDIS", TmdObjPDF, Alphas, pto, Cf);

  // Functions used for the tabulation
  const auto TabFunc    = [] (double const& b) -> double{ return log(b); };
  const auto InvTabFunc = [] (double const& fb) -> double{ return exp(fb); };

  // Initialise GSL random-number generator
  // (essential to initialize DataHandler object)
  gsl_rng *rng = gsl_rng_alloc(gsl_rng_ranlxs2);
  gsl_rng_set(rng, 1234);

  // Fluctuations (0 = central replica, no fluctuations)
  const int ReplicaID = 0;

  // YAML Emitter
  YAML::Emitter em;

  // Start YAML output file
  em.SetFloatPrecision(8);
  em.SetDoublePrecision(8);
  em << YAML::BeginMap;
  em << YAML::Key << "Parameterisation" << YAML::Value << "PV17";
  em << YAML::Key << "Non-perturbative function" << YAML::Value << fNP ->LatexFormula();

  em << YAML::Key << "Parameters" << YAML::Value << YAML::Flow;
  em << YAML::BeginMap;
  for (int i = 0; i < (int) fNP ->GetParameters().size(); i++)
    em << YAML::Key << fNP ->GetParameterNames()[i] << YAML::Value << fNP ->GetParameters()[i];
  em << YAML::EndMap;

  // Loop over the datasets
  em << YAML::Key << "Experiments" << YAML::Value << YAML::BeginSeq;

  // Start reading datasets
  const YAML::Node datasets = YAML::LoadFile(std::string(argv[2]) + "/datasets.yaml");
  for (auto const& exper : datasets)
    for (auto const& ds : exper.second)
      {
        std::cout << "\033[1;37m\nComputing SIDIS multiplicities for: " << ds["name"].as<std::string>() << "...\033[0m" << std::endl;

        std::string FFset;
        // Choose the correct FFs set depending on the experiment.
        if (ds["name"].as<std::string>().substr(0, 18) == "COMPASS_Deu_Pp_Kp_")
          FFset = "DSS07_NLO_HadronPlus";
        else if (ds["name"].as<std::string>().substr(0, 18) == "COMPASS_Deu_Pm_Km_")
          FFset = "DSS07_NLO_HadronMinus";
        else if (ds["name"].as<std::string>().substr(0, 14) == "HERMES_Pro_Pip" || ds["name"].as<std::string>().substr(0, 14) == "HERMES_Deu_Pip")
          FFset = "DSS14_NLO_Pip";
        else if (ds["name"].as<std::string>().substr(0, 14) == "HERMES_Pro_Pim" || ds["name"].as<std::string>().substr(0, 14) == "HERMES_Deu_Pim")
          FFset = "DSS14_NLO_Pim";
        else if (ds["name"].as<std::string>().substr(0, 13) == "HERMES_Pro_Kp"  || ds["name"].as<std::string>().substr(0, 13) == "HERMES_Deu_Kp")
          FFset = "DSS17_NLO_KaonPlus";
        else if (ds["name"].as<std::string>().substr(0, 13) == "HERMES_Pro_Km"  || ds["name"].as<std::string>().substr(0, 13) == "HERMES_Deu_Km")
          FFset = "DSS17_NLO_KaonMinus";
        else
          std::cout << "\033[1;31m [SIDISMultiplicities] Unknown SIDIS Experiment - FFs not initialized\033[0m" << std::endl;

        // Open LHAPDF FFs sets
        // FFset = config["ffset"]["name"].as<std::string>();
        LHAPDF::PDF* distff  = LHAPDF::mkPDF(FFset, config["ffset"]["member"].as<int>());

        // Rotate FF set into the QCD evolution basis
        const auto RotFFs  = [=] (double const& x, double const& mu) -> std::map<int,double> { return apfel::PhysToQCDEv(distff->xfxQ(x, mu)); };

        // Setup APFEL++ x-space grid for FFs
        std::vector<apfel::SubGrid> vsgf;
        for (auto const& sg : config["xgridff"])
          vsgf.push_back({sg[0].as<int>(), sg[1].as<double>(), sg[2].as<int>()});
        const apfel::Grid gff{vsgf};

        // Construct set of distributions as a function of the scale to be
        // tabulated
        const auto EvolvedFFs = [=,&gff] (double const& mu) -> apfel::Set<apfel::Distribution>
        {
          return apfel::Set<apfel::Distribution>{apfel::EvolutionBasisQCD{apfel::NF(mu, Thresholds)}, DistributionMap(gff, RotFFs, mu)};
        };

        // Tabulate collinear FFs
        const apfel::TabulateObject<apfel::Set<apfel::Distribution>> TabFFs{EvolvedFFs, 200, distff->qMin() * 0.9, distff->qMax(), 3, Thresholds};
        const auto CollFFs = [&] (double const& mu) -> apfel::Set<apfel::Distribution> { return TabFFs.Evaluate(mu); };

        // Initialize TMD FF objects
        const auto TmdObjFF  = apfel::InitializeTmdObjects(gff,  Thresholds);

        // Build evolved TMD FFs
        const auto EvTMDFFs    = BuildTmdFFs(TmdObjFF, CollFFs, Alphas, pto, Ci);
        const auto MatchTMDFFs = MatchTmdFFs(TmdObjFF, CollFFs, Alphas, pto, Ci);

        // Start reading datafiles
        const YAML::Node datafile = YAML::LoadFile(std::string(argv[2]) + "/" + exper.first.as<std::string>() + "/" + ds["file"].as<std::string>());
        NangaParbat::DataHandler* dh = new NangaParbat::DataHandler{ds["name"].as<std::string>(), datafile,
                                                                    rng, ReplicaID,
                                                                    std::vector<double>{}};

        // Keep track if the multiplicities are from HERMES or COMPASS
        bool COMPASS = (exper.first.as<std::string>() == "COMPASS" ? true : false);

        // Kinematics for the fully differential computation
        const double Vs = dh->GetKinematics().Vs;

        // Target isoscalarity
        const double targetiso = dh->GetTargetIsoscalarity();

        // Get x from the fist point, (x is constant in each bin)
        const double xav  = dh->GetBinning()[0].xav;

        // Get Q from the fist point, (Q is constant in each bin)
        const double Qav  = dh->GetBinning()[0].Qav;

        // Get z (which for HERMES changes slightly in a bin)
        std::vector<double> zv(dh->GetKinematics().ndata);
        for (int j = 0; j < dh->GetKinematics().ndata; j++)
          zv[j] = dh->GetBinning()[j].zav;
        // Get average z as average value in the bin
        const double zav = std::accumulate(zv.begin(), zv.end(), 0.) / zv.size();

        // Get PhT values
        std::vector<double> PhTv;
        for (auto const& vl : datafile["independent_variables"][0]["values"])
          PhTv.push_back(vl["value"].as<double>());

        // Compute qT vector, qT = PhT/z
        std::vector<double> qTv(PhTv.size());
        std::transform(PhTv.begin(), PhTv.end(), zv.begin(), qTv.begin(), std::divides<double>());

        // Experimental multiplicities
        const std::vector<double> multv = dh->GetMeanValues();

        // Get uncorrelated uncertainties
        const std::vector<double> uncv = dh->GetUncorrelatedUnc();

        // Get plotting labels
        const std::map<std::string, std::string> labels = dh->GetLabels();

        // Taking into account the isoscalarity of the target
        const int sign = (targetiso >= 0 ? 1 : -1);
        const double frp = std::abs(targetiso);
        const double frn = 1 - frp;

        // Tabulate initial scale TMD PDFs in b in the physical basis
        // taking into account the isoscalarity of the target.
        std::function<apfel::Set<apfel::Distribution>(double const&)> isTMDPDFs =
          [&] (double const& b) -> apfel::Set<apfel::Distribution>
        {
          const apfel::Set<apfel::Distribution> xF = QCDEvToPhys(MatchTMDPDFs(b).GetObjects());
          std::map<int, apfel::Distribution> xFiso;

          // Treat down and up separately to take isoscalarity of
          // the target into account.
          xFiso.insert({1,  frp * xF.at(sign) + frn * xF.at(sign*2)});
          xFiso.insert({-1, frp * xF.at(-sign) + frn * xF.at(-sign*2)});
          xFiso.insert({2,  frp * xF.at(sign*2) + frn * xF.at(sign)});
          xFiso.insert({-2, frp * xF.at(-sign*2) + frn * xF.at(-sign)});
          // Now run over the remaining flavours
          for (int i = 3; i <= 6; i++)
            {
              const int ip = i * sign;
              xFiso.insert({i, xF.at(ip)});
              xFiso.insert({-i, xF.at(-ip)});
            }
          return apfel::Set<apfel::Distribution>{xFiso};
        };
        const apfel::TabulateObject<apfel::Set<apfel::Distribution>> TabMatchTMDPDFs{isTMDPDFs, 200, 1e-2, 2, 1, {},
                                                                                     [] (double const& x) -> double{ return log(x); },
                                                                                     [] (double const& y) -> double{ return exp(y); }};

        // Tabulate initial scale TMD FFs in b in the physical basis
        std::function<apfel::Set<apfel::Distribution>(double const&)> isTMDFFs =
          [&] (double const& b) -> apfel::Set<apfel::Distribution>
        {
          return apfel::Set<apfel::Distribution>{QCDEvToPhys(MatchTMDFFs(b).GetObjects())};
        };
        const apfel::TabulateObject<apfel::Set<apfel::Distribution>> TabMatchTMDFFs{isTMDFFs, 200, 1e-2, 2, 1, {},
                                                                                    [] (double const& x) -> double{ return log(x); },
                                                                                    [] (double const& y) -> double{ return exp(y); }};

        apfel::Timer t;
        apfel::SetVerbosityLevel(0);
        /*
        std::cout << std::scientific;
        std::cout << "      Q       "
            << "        x        "
            << "        z       "
            << "       qT    "
            << "     sigma diff. "
            << "     incl. DIS "
            << "     theo. mult. "
            << "     expe. mult. "
            << std::endl;
        */

        // Denominator of multiplicities: compute inclusive cross section.
        // First adjust PDFs to account for the isoscalarity.
        const std::function<std::map<int, double>(double const&, double const&)> tPDFs = [&] (double const& x, double const& Q) -> std::map<int, double>
        {
          // Get PDFs in the physical basis
          const std::map<int, double> pr = apfel::QCDEvToPhys(TabPDFs.EvaluateMapxQ(x, Q));
          std::map<int, double> tg = pr;
          // Apply isoscalarity
          tg.at(1)  = frp * pr.at(1)  + frn * pr.at(2);
          tg.at(2)  = frp * pr.at(2)  + frn * pr.at(1);
          tg.at(-1) = frp * pr.at(-1) + frn * pr.at(-2);
          tg.at(-2) = frp * pr.at(-2) + frn * pr.at(-1);
          return tg;
        };

        // Rotate input PDF set back into the QCD evolution basis
        const auto tRotPDFs = [=] (double const& x, double const& mu) -> std::map<int, double> { return apfel::PhysToQCDEv(tPDFs(x, mu)); };

        // EW charges (only photon contribution, i.e. only electric
        // charges squared).
        std::function<std::vector<double>(double const&)> fBq = [] (double const&) -> std::vector<double> { return apfel::QCh2; };

        // Determine perturbative order according to the logarithmic
        // accuracy
        int PerturbativeOrder = 0;
        if (pto > 1 || pto < 0)
          PerturbativeOrder++;
        if (pto > 2 || pto < -1)
          PerturbativeOrder++;

        // Initialize inclusive structure functions
        const auto IF2 = BuildStructureFunctions(InitializeF2NCObjectsZM(gpdf, Thresholds), tRotPDFs, PerturbativeOrder,
                                                 [=] (double const& Q) -> double{ return TabAlphas.Evaluate(Q); }, fBq);
        const auto IFL = BuildStructureFunctions(InitializeFLNCObjectsZM(gpdf, Thresholds), tRotPDFs, PerturbativeOrder,
                                                 [=] (double const& Q) -> double{ return TabAlphas.Evaluate(Q); }, fBq);

        // Denominator for differential multiplicities
        const auto DiffInclusiveCS = [=](const double Q, const double x) -> double
        {
          const apfel::Distribution f2 = IF2.at(0).Evaluate(Q);
          const apfel::Distribution fl = IFL.at(0).Evaluate(Q);

          const double Yp = 1 + pow(1 - pow(Q / Vs, 2) / x, 2);

          /*
          // Check
          std::cout << "Q: " << Q << "   x: " << x << "Yp: " << Yp << std::endl;
          std::cout << "F2: " << f2.Evaluate(x) << "   FL: " << fl.Evaluate(x) << std::endl;
          std::cout << "xIntegrand: " << Yp * f2.Evaluate(x) / x - pow(Q / Vs, 4) * fl.Evaluate(x) / pow(x, 3) << std::endl;
          */

          return pow(TabAlphaem.Evaluate(Q), 2) / pow(Q, 3) * (Yp * f2.Evaluate(x) / x - pow(Q / Vs, 4) * fl.Evaluate(x) / pow(x, 3));
        };

        // Vector to store theoretical multiplicities
        std::vector<double> theomult(PhTv.size(), 0);

        // Compute differential multiplicities
        for (int iqT = 0; iqT < (int) qTv.size(); iqT++)
          {
            // Do not compute if qT > cut
            if (qTv[iqT] > qToQcut * Qav)
              continue;

            apfel::OgataQuadrature DEObj{0, 1e-9, 0.00001};

            // Bin central values
            const double qTm = qTv[iqT];
            const double Qm  = Qav;
            const double xm  = xav;
            // const double zm  = zv[iqT];
            const double zm  = zav;

            // Scales
            const double mu  = Cf * Qm;
            const double zeta = Qm * Qm;

            const double Yp   = 1 + pow(1 - pow(Qm / Vs, 2) / xm, 2);

            // Electroweak charges
            const std::vector<double> Bq = apfel::ElectroWeakCharges(Qm, false);

            // Number of active flavors
            const int nf = apfel::NF(mu, Thresholds);

            const apfel::TabulateObject<double> tf1NP{[=] (double const& tb) -> double { return fNP->Evaluate(xm, tb, zeta, 0); },
                                                      100, 5e-5, 5, 3, {}, TabFunc, InvTabFunc};
            const apfel::TabulateObject<double> tf2NP{[=] (double const& tb) -> double { return fNP->Evaluate(zm, tb, zeta, 1); },
                                                      100, 5e-5, 5, 3, {}, TabFunc, InvTabFunc};

            // Implement ll.741 of fastinterface.cc for the differential case
            const std::function<double(double const&)> bIntProva = [=] (double const& b) -> double
            {
              // bstar min prescription
              auto bs = NangaParbat::bstarmin(b, Qm);

              // Sum contributions from the active flavours
              double Lumiq = 0;
              for (int q = -nf; q <= nf; q++)
                {
                  // Skip the gluon
                  if (q == 0)
                    continue;

                  const double lumibsq = Yp * TabMatchTMDPDFs.EvaluatexQ(q, xm, bs) / xm * apfel::QCh2[std::abs(q)-1] * TabMatchTMDFFs.EvaluatexQ(q, zm, bs);
                  Lumiq += lumibsq;
                }

              return b * tf1NP.Evaluate(b) * tf2NP.Evaluate(b) * Lumiq * pow(QuarkSudakov(bs, mu, zeta), 2) / zm * pow(TabAlphaem.Evaluate(Qm), 2) * Hf(mu) / pow(Qm, 3);

            };

            // Differential cross section (numerator)
            const double differHERMES = apfel::ConvFact * apfel::FourPi * qTm * DEObj.transform(bIntProva, qTm) / (2 * Qm) / zm;
            const double differCOMPASS = apfel::ConvFact * apfel::FourPi * qTm * DEObj.transform(bIntProva, qTm) / (2 * Qm) / (2 * pow(zm, 2) * qTm);

            // Denominator
            const double denom = apfel::ConvFact * apfel::FourPi * DiffInclusiveCS(Qm, xm) / (2 * Qm);

            /*
            // Check denominator
            std::cout << DiffInclusiveCS(Qm, xm) / (2 * Qm) << std::endl;
            */

            // Multiplicity
            theomult[iqT] = (COMPASS ? differCOMPASS : differHERMES) / denom;

            /*
            // Write results on terminal
            std::cout << Qav << "\t" << xav << "\t" << zv[iqT] << "\t" << qTv[iqT] << "\t"
                << differ << "\t"
                << denom << "\t"
                << differ / denom << "\t"
                << multv[iqT]
                << std::endl;
            */
          }

        em << YAML::BeginMap;
        em << YAML::Key << "Name" << YAML::Value << dh->GetName();
        em << YAML::Key << "Plot title python" << YAML::Value << labels.at("titlepy");
        em << YAML::Key << "xlabelpy" << YAML::Value << labels.at("xlabelpy");
        em << YAML::Key << "ylabelpy" << YAML::Value << labels.at("ylabelpy");
        em << YAML::Key << "Q"   << YAML::Value << Qav;
        em << YAML::Key << "x"   << YAML::Value << xav;
        em << YAML::Key << "z"   << YAML::Value << zav;
        em << YAML::Key << "PhT"  << YAML::Value << YAML::Flow << YAML::BeginSeq;
        for (int i = 0; i < (int) PhTv.size(); i++)
          em << PhTv[i];
        em << YAML::EndSeq;
        em << YAML::Key << "qT"  << YAML::Value << YAML::Flow << YAML::BeginSeq;
        for (int iqT = 0; iqT < (int) qTv.size(); iqT++)
          em << qTv[iqT];
        em << YAML::EndSeq;
        em << YAML::Key << "Predictions" << YAML::Value << YAML::Flow << YAML::BeginSeq;
        for (int i = 0; i < (int) theomult.size(); i++)
          em << theomult[i];
        em << YAML::EndSeq;
        em << YAML::Key << "Central values" << YAML::Value << YAML::Flow << YAML::BeginSeq;
        for (int i = 0; i < (int) multv.size(); i++)
          em << multv[i];
        em << YAML::EndSeq;
        em << YAML::Key << "Uncorrelated uncertainties" << YAML::Value << YAML::Flow << YAML::BeginSeq;
        for (int i = 0; i < (int) uncv.size(); i++)
          em << uncv[i];
        em << YAML::EndSeq;
        em << YAML::EndMap;

        std::cout << "Total computation time, ";
        t.stop(true);

        delete distff;
      }

  // Output file
  std::ofstream fout(OutputFolder + "/Report.yaml");
  fout << em.c_str() << std::endl;
  fout.close();

  delete distpdf;
  delete fNP;

  // Delete random-number generator
  gsl_rng_free(rng);

  return 0;
}

#endif #if 0