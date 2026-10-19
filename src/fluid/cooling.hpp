// ***********************************************************************************
// Idefix MHD astrophysical code
//
// Source file src/fluid/cooling.hpp
//
// Last modified : 10/2026
//
// Copyright(C) by :
// - Geoffroy Lesur <geoffroy.lesur@univ-grenoble-alpes.fr> (IPAG/UGA/CNRS - 2023 - 2025)
// and other code contributors
//
// Licensed under CeCILL 2.1 License, see COPYING for more information
// ***********************************************************************************
#ifndef FLUID_COOLING_HPP_
#define FLUID_COOLING_HPP_

#include <limits>
#include <sstream>
#include <string>
#include "idefix.hpp"
#include "input.hpp"
#include "fluid_defs.hpp"
#include "eos.hpp"
#include "units.hpp"

// Forward class hydro declaration
template <typename Phys> class Fluid;
class DataBlock;

//----------------------------------------------------------------------------------------
// ISM cooling function Lambda(T) in erg cm^3 s^-1 (T in K), identical to AthenaK's ISMCoolFn
//  - log T <= 4.2       : Koyama & Inutsuka (2002)
//  - 4.2 < log T < 8.15 : SPEX table, Table 2 of Schure et al, A&A 508, 751 (2009)
//  - log T > 8.15       : CGOLS fit
KOKKOS_INLINE_FUNCTION
real ISMCoolFn(real temp) {
  // original data from Schure et al. paper, covers 4.12 < logt < 8.16
  const float lhd[102] = {
      -22.5977, -21.9689, -21.5972, -21.4615, -21.4789, -21.5497, -21.6211, -21.6595,
      -21.6426, -21.5688, -21.4771, -21.3755, -21.2693, -21.1644, -21.0658, -20.9778,
      -20.8986, -20.8281, -20.7700, -20.7223, -20.6888, -20.6739, -20.6815, -20.7051,
      -20.7229, -20.7208, -20.7058, -20.6896, -20.6797, -20.6749, -20.6709, -20.6748,
      -20.7089, -20.8031, -20.9647, -21.1482, -21.2932, -21.3767, -21.4129, -21.4291,
      -21.4538, -21.5055, -21.5740, -21.6300, -21.6615, -21.6766, -21.6886, -21.7073,
      -21.7304, -21.7491, -21.7607, -21.7701, -21.7877, -21.8243, -21.8875, -21.9738,
      -22.0671, -22.1537, -22.2265, -22.2821, -22.3213, -22.3462, -22.3587, -22.3622,
      -22.3590, -22.3512, -22.3420, -22.3342, -22.3312, -22.3346, -22.3445, -22.3595,
      -22.3780, -22.4007, -22.4289, -22.4625, -22.4995, -22.5353, -22.5659, -22.5895,
      -22.6059, -22.6161, -22.6208, -22.6213, -22.6184, -22.6126, -22.6045, -22.5945,
      -22.5831, -22.5707, -22.5573, -22.5434, -22.5287, -22.5140, -22.4992, -22.4844,
      -22.4695, -22.4543, -22.4392, -22.4237, -22.4087, -22.3928};

  real logt = std::log10(temp);

  // for temperatures less than 10^4 K, use Koyama & Inutsuka (2002)
  if (logt <= 4.2) {
    return (2.0e-19*std::exp(-1.184e5/(temp + 1.0e3))
            + 2.8e-28*std::sqrt(temp)*std::exp(-92.0/temp));
  }

  // for temperatures above 10^8.15 use CGOLS fit
  if (logt > 8.15) {
    return std::pow(10.0, (0.45*logt - 26.065));
  }

  // in between values of 4.2 < log(T) < 8.15
  // linear interpolation of tabulated SPEX cooling rate
  int ipps  = static_cast<int>(25.0*logt) - 103;
  ipps = (ipps < 100)? ipps : 100;
  ipps = (ipps > 0 )? ipps : 0;
  real x0 = 4.12 + 0.04*static_cast<real>(ipps);
  real dx = logt - x0;
  real logcool = (lhd[ipps+1]*dx - lhd[ipps]*(dx - 0.04))*25.0;
  return std::pow(10.0, logcool);
}

//----------------------------------------------------------------------------------------
// Lambda(T) in erg cm^3 s^-1, either by linear interpolation of a float table uniform in
// log10 T (one float log10 per call, cheap on GPUs with slow FP64) or by ISMCoolFn()
KOKKOS_INLINE_FUNCTION
real CoolLambda(const real temp, const bool useTab, const IdefixArray1D<float> &tab,
                const float lt0, const float dinv, const int ntab) {
  if(!useTab) return ISMCoolFn(temp);
  float x = (std::log10(static_cast<float>(temp)) - lt0)*dinv;
  x = std::fmin(std::fmax(x, 0.0f), static_cast<float>(ntab - 1) - 1.0e-3f);
  const int i0 = static_cast<int>(x);
  const float f = x - static_cast<float>(i0);
  return static_cast<real>(tab(i0) + f*(tab(i0+1) - tab(i0)));
}

// Optically thin radiative cooling (and optional constant heating)
//    dE/dt = -n_e n_H Lambda(T) + n Gamma,   with n = rho/(mu*amu)
// Lambda is normalised to n_e n_H (Schure+09). For a fully ionised H+He plasma of
// hydrogen mass fraction X: n_H = X rho/m_p and n_e = (1+X)/2 rho/m_p
//
// Two modes, selected by [Cooling] subcycle:
//  no (default): explicit update inside every RK stage (AddCooling). The timestep is
//                limited to CFL * t_cool,min.
//  yes:          cooling is removed from the RK stages. After the full hydro step, the
//                TimeIntegrator calls SubcycleCooling(), which integrates cooling over dt
//                with nsub uniform explicit sub-steps, nsub being the SAME for every cell
//                (no GPU warp divergence). The timestep is then only capped by
//                dt <= nsub_max * cool_cfl * t_cool,min.
class Cooling {
 public:
  enum class Type {ISM};

  template <typename Phys>
  Cooling(Input &, Fluid<Phys> *);
  void ShowConfig();                    // print configuration
  void AddCooling(const real);          // add cooling to Uc and its constraint to InvDt

  // Operator-split cooling over the full step dt, to be called after the RK stages
  void SubcycleCooling(const real dt);
  // Upper bound on the next dt set by the sub-cycled cooling (global over MPI ranks)
  real GetDtCap() const;
  // MPI-global minimum cooling time over active cells
  // (public: CUDA forbids extended device lambdas in private member functions)
  real MinCoolingTime();

  IdefixArray4D<real> Uc;
  IdefixArray4D<real> Vc;
  IdefixArray3D<real> InvDt;

  bool subcycle{false};   // operator-split, sub-cycled cooling
  int lastNsub{0};        // number of sub-steps used in the most recent cycle

 private:
  // Unit conversions and rates, captured by value in the kernels
  struct Coefs {
    real tempUnit;        // T(K) = tempUnit * P/rho
    real lambdaUnit;      // n_e n_H/n^2 / coolingUnit: Lambda(erg cm^3/s) -> code units
    real gheat;           // heating rate per particle in code units
    real Tfloor;          // temperature floor (K)
    bool haveFloor;
  };
  Coefs GetCoefs() const;

  DataBlock* data;
  EquationOfState eos;

  Type type;
  real mu;              // mean molecular weight
  real XH;              // hydrogen mass fraction (sets n_e and n_H)
  real Tfloor;          // temperature floor (K), disabled when <= 0
  real heatingRate;     // heating rate per particle (erg/s)

  // optional cooling mask on a passive tracer: cells whose tracer is below maskThreshold
  // are not cooled and do not constrain the timestep
  bool haveMask{false};
  int maskTracer{-1};   // tracer number given in the input file
  int maskIndex{0};     // index of the masking tracer in Vc
  real maskThreshold{0};

  // sub-cycling parameters
  int nsubMax{1};       // maximum number of cooling sub-steps per hydro step
  real coolCfl{1};      // each sub-step satisfies dt_sub <= coolCfl * t_cool,min
  real tcoolMinPost;    // global post-cooling t_cool,min, sets the next dt cap

  // tabulated cooling function (float, uniform in log10 T), used by the sub-cycled solver
  bool useTable{false};
  IdefixArray1D<float> coolTab;
  float tabLogT0{0};    // log10(T) of the first entry
  float tabDinv{1};     // 1/dlog10(T)
  int tabN{0};          // number of entries
};

#include "fluid.hpp"

template<typename Phys>
Cooling::Cooling(Input &input, Fluid<Phys> *hydroin):
                      Uc{hydroin->Uc},
                      Vc{hydroin->Vc},
                      InvDt{hydroin->InvDt} {
  idfx::pushRegion("Cooling::Cooling");
  this->data = hydroin->data;

  if constexpr(!Phys::pressure) {
    IDEFIX_ERROR("Cooling requires an energy equation (HAVE_ENERGY)");
  } else {
    this->eos = *(hydroin->eos.get());
  }

  if(!idfx::units.GetIsInitialized()) {
    IDEFIX_ERROR("Cooling requires physical units: add a [Units] block to your input file");
  }

  std::string blockName = std::string(Phys::prefix);
  std::string coolType = input.Get<std::string>(blockName,"cooling",0);
  if(coolType.compare("ism") == 0) {
    this->type = Type::ISM;
  } else {
    std::stringstream msg;
    msg << "Unknown cooling type: " << coolType << ". Can only be ism.";
    IDEFIX_ERROR(msg);
  }

  this->mu = input.Get<real>("Cooling","mu",0);
  this->XH = input.GetOrSet<real>("Cooling","X",0,0.7);
  if(XH <= 0 || XH > 1) {
    IDEFIX_ERROR("Cooling: hydrogen mass fraction X must be in (0,1]");
  }
  this->Tfloor = input.GetOrSet<real>("Cooling","Tfloor",0,0.0);
  this->heatingRate = input.GetOrSet<real>("Cooling","heating",0,0.0);

  // Optional mask: no cooling where tracer mask_tracer < mask_threshold
  this->maskTracer = input.GetOrSet<int>("Cooling","mask_tracer",0,-1);
  if(maskTracer >= 0) {
    if(maskTracer >= hydroin->nTracer) {
      std::stringstream msg;
      msg << "Cooling: mask_tracer=" << maskTracer << " but " << blockName
          << " has only " << hydroin->nTracer << " tracer(s)";
      IDEFIX_ERROR(msg);
    }
    this->haveMask = true;
    this->maskIndex = Phys::nvar + maskTracer;
    this->maskThreshold = input.GetOrSet<real>("Cooling","mask_threshold",0,1.0e-3);
  }

  this->subcycle = input.GetOrSet<bool>("Cooling","subcycle",0,false);
  this->tcoolMinPost = std::numeric_limits<real>::max();
  if(subcycle) {
    this->nsubMax = input.GetOrSet<int>("Cooling","nsub_max",0,100);
    this->coolCfl = input.GetOrSet<real>("Cooling","cool_cfl",0,0.5);
    if(nsubMax < 1 || coolCfl <= 0) {
      IDEFIX_ERROR("Cooling: nsub_max must be >= 1 and cool_cfl > 0");
    }
    this->useTable = input.GetOrSet<bool>("Cooling","table",0,true);
    if(useTable) {
      // Lambda(T) on a uniform grid in log10 T from 10 K to 10^9 K, stored as float.
      // Below/above that range the end values are used.
      const double lt0 = 1.0, lt1 = 9.0, dlt = 0.002;
      this->tabN = static_cast<int>(std::lround((lt1 - lt0)/dlt)) + 1;
      this->tabLogT0 = static_cast<float>(lt0);
      this->tabDinv = static_cast<float>(1.0/dlt);
      this->coolTab = IdefixArray1D<float>("Cooling_table", tabN);
      auto htab = Kokkos::create_mirror_view(coolTab);
      for(int n = 0 ; n < tabN ; n++) {
        htab(n) = static_cast<float>(ISMCoolFn(std::pow(10.0, lt0 + n*dlt)));
      }
      Kokkos::deep_copy(coolTab, htab);
    }
  }

  idfx::popRegion();
}

#endif // FLUID_COOLING_HPP_
