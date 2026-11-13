// ***********************************************************************************
// Idefix MHD astrophysical code
//
// Source file problems/single_double_scaling/setup.cpp
//
// Single vs double precision benchmark: turb_tcgm model without turbulence, uniform grid.
// Runaway BH (frame of the BH) moving through a uniform CGM, with a wind injected inside
// r < rinj, BH gravity and ISM cooling (sub-cycled). The same source is built twice,
// with -DIdefix_PRECISION=Double and -DIdefix_PRECISION=Single.
// All cgs arithmetic is done in double, so the setup is safe when real = float.
// ***********************************************************************************

#include "idefix.hpp"
#include "setup.hpp"
#include "units.hpp"
#include <cmath>

namespace{
  real GMbh, vbh, eps,
  ncgm0, dt_ncgm, Tcgm,
  Mdot0, dt_mdot,
  rinj, vw0, dt_vw, Volinj,
  gamma_gas, mu_gas, XH;

  void AddSourceTerms(Hydro *hydro, const real t, const real dtin);
  void AddBHGrav(Hydro *hydro, const real t, const real dtin);
  void AddOutflow(Hydro *hydro, const real t, const real dtin);
  void WindZBC(Hydro *hydro, int dir, BoundarySide side, const real t);
  real InjectionVolume(DataBlock &data, const real rinj);
  // turb_tcgm model (t in code time):
  //   n_cgm(t) = n_cgm0 exp(-t/dt_ncgm)
  //   Mdot(t)  = Mdot0  exp(-t/dt_mdot)
  //   v_w(t)   = v_w0   exp(-t/dt_vw)
  real N_CGM(const real t) {
      return ncgm0*std::exp(-t/dt_ncgm);
  }
  real Mdot_t(const real t) {
      return Mdot0*std::exp(-t/dt_mdot);
  }
  real Vw_t(const real t) {
      return vw0*std::exp(-t/dt_vw);
  }
  // CGM density and pressure (code units) for a number density n (cm^-3) at T_cgm.
  // cgs arithmetic in double (safe when real = float); only the code-unit result is real.
  real CGMDensity(const real n) {
      double rho0 = idfx::units.GetDensity();
      return static_cast<real>(static_cast<double>(n)*mu_gas*idfx::units.m_p/rho0);
  }
  real CGMPressure(const real n) {
      double rho0 = idfx::units.GetDensity();
      double v0   = idfx::units.GetVelocity();
      double kB   = idfx::units.k_B;
      return static_cast<real>(static_cast<double>(n)*kB*Tcgm/(rho0*v0*v0));
  }
}

// ---------------------------------------------------------------------------
// Constructor: read parameters, convert to code units, enroll functions
// ---------------------------------------------------------------------------
Setup::Setup(Input &input, Grid &grid, DataBlock &data, Output &output) {
  // Physical inputs
  double Mbh_       = input.Get<double>("Setup","Mbh",0);    // M_sun (double: Mbh*M_sun ~ 4e40 g > FLT_MAX)
  real vbh_         = input.Get<real>("Setup","vbh",0);      // code units
  real eps_         = input.Get<real>("Setup","epsilon",0);  // code_units
  real ncgm0_       = input.Get<real>("Setup","n_cgm0",0);   // cm^-3
  real dt_ncgm_     = input.Get<real>("Setup","dt_ncgm",0);
  real Tcgm_        = input.Get<real>("Setup","T_cgm",0);    // K
  real mu_          = input.Get<real>("Cooling","mu",0);
  real gamma_       = input.Get<real>("Hydro","gamma",0);
  double Mdot0_     = input.Get<double>("Setup","Mdot0",0);  // Msun/yr, at t=0
  real dt_mdot_     = input.Get<real>("Setup","dt_mdot",0);
  real vw0_         = input.Get<real>("Setup","vw0",0);
  real dt_vw_       = input.Get<real>("Setup","dt_vw",0);
  real rinj_        = input.Get<real>("Setup","rinj",0);
  real XH_          = input.GetOrSet<real>("Cooling","X",0,0.7);

  // Conversions (idfx::units is initialised from [Units] before Setup is built).
  // All cgs arithmetic is done in double: M0 = rho0*L0^3 ~ 5e40 g and Mbh*M_sun ~ 4e40 g
  // overflow float (FLT_MAX ~ 3.4e38). Only the final code-unit values are cast to real.
  double L0         = idfx::units.GetLength();
  double v0         = idfx::units.GetVelocity();
  double rho0       = idfx::units.GetDensity();
  double M0         = rho0*L0*L0*L0;
  double t0         = L0/v0;
  double Msun       = idfx::units.M_sun;
  double Gcgs       = idfx::units.G;
  double syr        = idfx::units.s_yr;

  double Mbh_code   = (Mbh_*Msun)/M0;
  double G_code     = Gcgs*rho0*t0*t0;
  double Mdot0_cgs  = (Mdot0_*Msun)/syr;
  double Mdot0_code = Mdot0_cgs*t0/M0;

  GMbh              = static_cast<real>(G_code*Mbh_code);
  vbh               = vbh_;
  eps               = eps_;
  ncgm0             = ncgm0_;
  dt_ncgm           = dt_ncgm_;
  Tcgm              = Tcgm_;
  Mdot0             = static_cast<real>(Mdot0_code);
  dt_mdot           = dt_mdot_;
  rinj              = rinj_;
  vw0               = vw0_;
  dt_vw             = dt_vw_;
  gamma_gas         = gamma_;
  mu_gas            = mu_;
  XH                = XH_;
  // Actual volume of the cells with r < rinj (design notes §2), so the
  // injected Mdot is exact on the stair-stepped sphere
  Volinj            = InjectionVolume(data, rinj_);
  if(Volinj <= ZERO_F) IDEFIX_ERROR("No cell centre inside rinj: increase rinj or refine the grid");
  idfx::cout << "Injection volume: " << Volinj << " (analytic sphere: "
             << 4*M_PI*rinj_*rinj_*rinj_/3 << ")" << std::endl;
  #ifdef SINGLE_PRECISION
    idfx::cout << "Precision: SINGLE (real = float)" << std::endl;
  #else
    idfx::cout << "Precision: DOUBLE (real = double)" << std::endl;
  #endif
  data.hydro->EnrollUserDefBoundary(WindZBC);
  data.hydro->EnrollUserSourceTerm(AddSourceTerms);
}

// ---------------------------------------------------------------------------
// Initial condition: uniform CGM moving at -vbh z_hat, plus the wind inside rinj
// ---------------------------------------------------------------------------
void Setup::InitFlow(DataBlock &data) {
  DataBlockHost d(data);
  real rho_cgm0         = CGMDensity(N_CGM(ZERO_F));
  real pres_cgm0        = CGMPressure(N_CGM(ZERO_F));
  real vbh_             = vbh;
  for(int k = 0; k < d.np_tot[KDIR] ; k++) {
    for(int j = 0; j < d.np_tot[JDIR] ; j++) {
      for(int i = 0; i < d.np_tot[IDIR] ; i++) {
        d.Vc(RHO,k,j,i) = rho_cgm0;
        d.Vc(VX1,k,j,i) = ZERO_F;
        d.Vc(VX2,k,j,i) = -vbh_;
        d.Vc(VX3,k,j,i) = ZERO_F;
        d.Vc(PRS,k,j,i) = pres_cgm0;
        d.Vc(TRG,k,j,i) = 0.0;
      }
    }
  }
  d.SyncToDevice();

  // Wind inside r < rinj
  IdefixArray4D<real> Vc = data.hydro->Vc;
  IdefixArray1D<real> x1 = data.x[IDIR];  // R
  IdefixArray1D<real> x2 = data.x[JDIR];  // z
  real rinj_    = rinj;
  real vinj     = Vw_t(ZERO_F)/sqrt(2);
  real rho_w    = Mdot_t(ZERO_F)/(4*M_PI*rinj*rinj*vinj);   // free wind at r = rinj
  real pres_w   = rho_w*vinj*vinj/gamma_gas;        // c_s = vw, consistent with Se

  idefix_for("InitWind",
    0, data.np_tot[KDIR],
    0, data.np_tot[JDIR],
    0, data.np_tot[IDIR],
    KOKKOS_LAMBDA (int k, int j, int i) {
        real R    = x1(i);
        real z    = x2(j);
        real rad  = sqrt(R*R + z*z);
        if(rad < rinj_) {
          Vc(RHO,k,j,i) = rho_w;
          Vc(VX1,k,j,i) = vinj*(R/rad);
          Vc(VX2,k,j,i) = vinj*(z/rad);
          Vc(PRS,k,j,i) = pres_w;
          Vc(TRG,k,j,i) = 1.0;
        }
  });
}

namespace{
void AddSourceTerms(Hydro *hydro, const real t, const real dtin){
  AddBHGrav(hydro, t, dtin);
  AddOutflow(hydro, t, dtin);
}
void AddBHGrav(Hydro *hydro, const real t, const real dtin){
  auto *data = hydro->data;
  IdefixArray4D<real> Vc = hydro->Vc;     // read primitives
  IdefixArray4D<real> Uc = hydro->Uc;     // update conserved
  IdefixArray1D<real> x1 = data->x[IDIR]; // R
  IdefixArray1D<real> x2 = data->x[JDIR]; // z

  // Copy globals into locals: the lambda below runs on the GPU
  real dt     = dtin;
  real GMbh_  = GMbh;
  real eps2   = eps*eps;

  idefix_for("Add_bh_grav",
    data->beg[KDIR], data->end[KDIR],
    data->beg[JDIR], data->end[JDIR],
    data->beg[IDIR], data->end[IDIR],
    KOKKOS_LAMBDA (int k, int j, int i) {
      real R          = x1(i);
      real z          = x2(j);
      real r2         = R*R + z*z + eps2;   //r^2

      real vr         = Vc(VX1,k,j,i);
      real vz         = Vc(VX2,k,j,i);
      real rho        = Vc(RHO,k,j,i);
      real gradphi_r  = GMbh_/(r2*sqrt(r2));   // GM / (r^2+eps^2)^(3/2)
      real vdot_r     = vr*R+vz*z;

      Uc(MX1,k,j,i) -= rho*dt*gradphi_r*R;
      Uc(MX2,k,j,i) -= rho*dt*gradphi_r*z;
      #if HAVE_ENERGY
        Uc(ENG,k,j,i) -= rho*dt*gradphi_r*vdot_r;
      #endif
  });
}
void AddOutflow(Hydro *hydro, const real t, const real dtin){
  auto *data = hydro->data;
  IdefixArray4D<real> Uc = hydro->Uc;     // update conserved
  IdefixArray1D<real> x1 = data->x[IDIR]; // R
  IdefixArray1D<real> x2 = data->x[JDIR]; // z

  // Copy globals into locals: the lambda below runs on the GPU
  real dt       = dtin;
  real rinj_    = rinj;
  real gamma_   = gamma_gas;
  real gm1      = gamma_-1;

  real vinj_    = Vw_t(t)/sqrt(2);
  real Srho     = Mdot_t(t)/Volinj;
  real Se       = (Srho*vinj_*vinj_)*(0.5+(1/(gamma_*gm1)));
  idefix_for("Add_outflow",
    data->beg[KDIR], data->end[KDIR],
    data->beg[JDIR], data->end[JDIR],
    data->beg[IDIR], data->end[IDIR],
    KOKKOS_LAMBDA (int k, int j, int i) {
      real R          = x1(i);
      real z          = x2(j);
      real rad        = sqrt(R*R + z*z);   //r
      if(rad<rinj_){
        Uc(RHO,k,j,i) += Srho*dt;
        Uc(MX1,k,j,i) += Srho*dt*vinj_*(R/rad);
        Uc(MX2,k,j,i) += Srho*dt*vinj_*(z/rad);
        #if HAVE_ENERGY
        Uc(ENG,k,j,i) += Se*dt;
        #endif
        Uc(TRG,k,j,i) = Uc(RHO,k,j,i)*1.0;
      }
  });
}
// Total volume of the active cells with r < rinj, summed over all ranks.
// In 2D cylindrical dV = R dR dz (per radian), hence the factor 2 pi.
real InjectionVolume(DataBlock &data, const real rinj) {
  IdefixArray1D<real> x1 = data.x[IDIR];
  IdefixArray1D<real> x2 = data.x[JDIR];
  IdefixArray3D<real> dV = data.dV;
  real vol = ZERO_F;
  idefix_reduce("Inj_vol",
    data.beg[KDIR], data.end[KDIR],
    data.beg[JDIR], data.end[JDIR],
    data.beg[IDIR], data.end[IDIR],
    KOKKOS_LAMBDA (int k, int j, int i, real &acc) {
      real R = x1(i), z = x2(j);
      if(R*R + z*z < rinj*rinj) acc += dV(k,j,i);
    },
    Kokkos::Sum<real>(vol));
  #ifdef WITH_MPI
    MPI_Allreduce(MPI_IN_PLACE, &vol, 1, realMPI, MPI_SUM, MPI_COMM_WORLD);
  #endif
  return 2*M_PI*vol;
}

// Time-dependent CGM flowing in at z_max with uniform velocity -vbh z_hat
void WindZBC(Hydro *hydro, int dir, BoundarySide side, const real t){
  auto *data = hydro->data;
  if(dir==JDIR && side==right) {
    IdefixArray4D<real> Vc = hydro->Vc;
    real ncgm_t     = N_CGM(t);
    real rho_cgm_   = CGMDensity(ncgm_t);
    real pres_cgm_  = CGMPressure(ncgm_t);
    real vbh_       = vbh;
    idefix_for("InflowZmax",
      0, data->np_tot[KDIR],
      data->end[JDIR], data->np_tot[JDIR],   // ghost cells beyond z_max
      0, data->np_tot[IDIR],
      KOKKOS_LAMBDA (int k, int j, int i) {
        Vc(RHO,k,j,i) = rho_cgm_;
        Vc(VX1,k,j,i) = ZERO_F;
        Vc(VX2,k,j,i) = -vbh_;
        Vc(VX3,k,j,i) = ZERO_F;
        Vc(PRS,k,j,i) = pres_cgm_;
        Vc(TRG,k,j,i) = 0.0;
      });
  }
}
}//namespace
