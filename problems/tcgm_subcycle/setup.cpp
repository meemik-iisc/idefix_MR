// ***********************************************************************************
// Idefix MHD astrophysical code
//
// Source file problems/tcgm/setup.cpp
// ***********************************************************************************

#include "idefix.hpp"
#include "setup.hpp"
#include "units.hpp"


namespace{
  real GMbh, vbh, eps, 
  ncgm0, dt_ncgm, Tcgm,
  Mdot0, dt_mdot, 
  rinj, vinj, Volinj,
  gamma_gas, mu_gas, XH;
  // real rho_sink, Macc{0}, Mdot{0};         // accreted mass and rate (code units)
  void AddSourceTerms(Hydro *hydro, const real t, const real dtin);
  void AddBHGrav(Hydro *hydro, const real t, const real dtin);
  void AddOutflow(Hydro *hydro, const real t, const real dtin);
  void AddCGMHeating(Hydro *hydro, const real t, const real dtin);
  void WindZBC(Hydro *hydro, int dir, BoundarySide side, const real t);
  // void ApplySink(DataBlock &data, const real t, const real dt);
  // void SinkAnalysis(DataBlock &data);
  // Functions Time dependent CGM
  KOKKOS_INLINE_FUNCTION
  static real N_CGM(const real t, const real ncgm0,  const real dt_ncgm) {
      real ncgm = ncgm0*exp(-t/dt_ncgm);
      return ncgm;
  }
  // Functions Time dependent Mdot
  KOKKOS_INLINE_FUNCTION
  static real Mdot_t(const real t, const real Mdot_0, const real dt_mdot) {
      return Mdot_0*exp(t/dt_mdot);
      // return Mdot_0;
  }
  // CGM density and pressure (code units) for a number density n (cm^-3) at T_cgm
  real CGMDensity(const real n) {
      return n*mu_gas*idfx::units.m_p/idfx::units.GetDensity();
  }
  real CGMPressure(const real n) {
      real v0 = idfx::units.GetVelocity();
      return n*idfx::units.k_B*Tcgm/(idfx::units.GetDensity()*v0*v0);
  }
} //namespace

// ---------------------------------------------------------------------------
// Constructor: read parameters, convert to code units, enroll functions
// ---------------------------------------------------------------------------
Setup::Setup(Input &input, Grid &grid, DataBlock &data, Output &output) {
  // Physical inputs
  real Mbh_       = input.Get<real>("Setup","Mbh",0);      // M_sun
  real vbh_       = input.Get<real>("Setup","vbh",0);      // code units
  real eps_       = input.Get<real>("Setup","epsilon",0);  // code_units
  real ncgm0_     = input.Get<real>("Setup","n_cgm0",0);   // m_p/cc = code density
  real dt_ncgm_   = input.Get<real>("Setup","dt_ncgm",0);
  // real sink_rat_  = input.Get<real>("Setup","sink_rat",0); // rho_sink / rho_cgm
  real Tcgm_      = input.Get<real>("Setup","T_cgm",0);    // K
  real mu_        = input.Get<real>("Cooling","mu",0);
  real gamma_     = input.Get<real>("Hydro","gamma",0);
  real Mdot0_     = input.Get<real>("Setup","Mdot0",0);
  real dt_mdot_   = input.Get<real>("Setup","dt_mdot",0);
  real vw_        = input.Get<real>("Setup","vw",0);
  real rinj_      = input.Get<real>("Setup","rinj",0);

  // Conversions (idfx::units is initialised from [Units] before Setup is built)
  real L0         = idfx::units.GetLength();
  real v0         = idfx::units.GetVelocity();
  real rho0       = idfx::units.GetDensity();
  real M0         = rho0*L0*L0*L0;
  real t0         = L0/v0;  

  real Mbh_code   = (Mbh_*idfx::units.M_sun)/M0;
  real G_code     = idfx::units.G*rho0*t0*t0;
  real Mdot0_cgs  = (Mdot0_*idfx::units.M_sun)/idfx::units.s_yr;
  real Mdot0_code = Mdot0_cgs*t0/M0;
  real vinj_      = vw_/sqrt(2);
  real Volinj_    = (4*M_PI*rinj_*rinj_*rinj_)/3;


  GMbh            = G_code*Mbh_code;
  vbh             = vbh_;
  eps             = eps_;
  ncgm0           = ncgm0_;
  dt_ncgm         = dt_ncgm_;
  Tcgm            = Tcgm_;
  Mdot0           = Mdot0_code;
  dt_mdot         = dt_mdot_;
  rinj            = rinj_;
  vinj            = vinj_;
  Volinj          = Volinj_;
  gamma_gas       = gamma_;
  mu_gas          = mu_;
  XH              = input.GetOrSet<real>("Cooling","X",0,0.7);  // same default as Cooling

  data.hydro->EnrollUserDefBoundary(WindZBC);
  data.hydro->EnrollUserSourceTerm(AddSourceTerms);

}

// ---------------------------------------------------------------------------
// Initial condition: uniform wind everywhere
// ---------------------------------------------------------------------------
void Setup::InitFlow(DataBlock &data) {
  DataBlockHost d(data);
  real rho_cgm0   = CGMDensity(ncgm0);
  real pres_cgm0  = CGMPressure(ncgm0);
  real vbh_       = vbh;
  for(int k = 0; k < d.np_tot[KDIR] ; k++) {
    for(int j = 0; j < d.np_tot[JDIR] ; j++) {
      for(int i = 0; i < d.np_tot[IDIR] ; i++) {
        d.Vc(RHO,k,j,i) = rho_cgm0;
        d.Vc(VX1,k,j,i) = ZERO_F;
        d.Vc(VX2,k,j,i) = -1*vbh_;
        d.Vc(VX3,k,j,i) = ZERO_F;
        d.Vc(PRS,k,j,i) = pres_cgm0;
        d.Vc(TRG,k,j,i) = 0.0;
      }
    }
  }

  d.SyncToDevice();
}

void MakeAnalysis(DataBlock & data) {
}

namespace{
void AddSourceTerms(Hydro *hydro, const real t, const real dtin){
  AddBHGrav(hydro, t, dtin);
  AddOutflow(hydro, t, dtin);
  // No AddCGMHeating with sub-cycled cooling: the CGM is kept in thermal balance by
  // [Cooling] mask_tracer instead (cells with wind fraction < mask_threshold do not cool)
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
        // Later: cooling goes here, e.g. Uc(ENG,k,j,i) -= dt*n*n*Lambda(T);
      #endif
  });
}
void AddOutflow(Hydro *hydro, const real t, const real dtin){
  auto *data = hydro->data;
  IdefixArray4D<real> Vc = hydro->Vc;     // read primitives
  IdefixArray4D<real> Uc = hydro->Uc;     // update conserved
  IdefixArray1D<real> x1 = data->x[IDIR]; // R
  IdefixArray1D<real> x2 = data->x[JDIR]; // z

  // Copy globals into locals: the lambda below runs on the GPU
  real dt       = dtin;
  real time     = t;
  real Mdot0_   = Mdot0;
  real dt_mdot_ = dt_mdot;
  real vinj_    = vinj;
  real rinj_    = rinj;
  real Volinj_  = Volinj;
  real gamma_   = gamma_gas;
  real gm1      = gamma_-1;

  real mdot_t   = Mdot_t(time, Mdot0_, dt_mdot_);
  real Srho     = mdot_t/Volinj_;
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
        // Later: cooling goes here, e.g. Uc(ENG,k,j,i) -= dt*n*n*Lambda(T);
        #endif
        Uc(TRG,k,j,i) = Uc(RHO,k,j,i)*1.0;
      }
      
  });
}
// Port of AthenaK's AddCGMHeating: in pure CGM cells (wind fraction < 1e-3) add back
// exactly the radiative losses of Cooling::AddCooling, so the unmixed CGM stays in
// thermal balance and only wind-polluted gas cools. Same units and n_e n_H factor as
// src/fluid/cooling.cpp; both use the stage-start primitives Vc, so they cancel exactly.
void AddCGMHeating(Hydro *hydro, const real t, const real dtin){
  auto *data = hydro->data;
  IdefixArray4D<real> Vc = hydro->Vc;     // read primitives
  IdefixArray4D<real> Uc = hydro->Uc;     // update conserved

  // Copy globals into locals: the lambda below runs on the GPU
  real dt       = dtin;
  real mu_      = mu_gas;
  real XH_      = XH;

  const real tempUnit     = mu_*idfx::units.GetKelvin();    // T(K) = tempUnit*P/rho
  const real nUnit        = idfx::units.GetDensity()/(mu_*idfx::units.u);
  const real powerUnit    = idfx::units.GetEnergy()/idfx::units.GetTime();
  const real coolingUnit  = powerUnit/(nUnit*nUnit);
  const real nRatio       = mu_*idfx::units.u/idfx::units.m_p;
  const real neHfac       = HALF_F*XH_*(ONE_F+XH_)*nRatio*nRatio;   // n_e n_H / n^2

  // f_wind < 1e-3 means essentially pure CGM
  const real wind_threshold = 1e-3;

  idefix_for("Add_cgm_heating",
    data->beg[KDIR], data->end[KDIR],
    data->beg[JDIR], data->end[JDIR],
    data->beg[IDIR], data->end[IDIR],
    KOKKOS_LAMBDA (int k, int j, int i) {
      real rho        = Vc(RHO,k,j,i);
      if(rho <= ZERO_F) return;
      real temp       = tempUnit*Vc(PRS,k,j,i)/rho;
      real lambda     = neHfac*ISMCoolFn(temp)/coolingUnit;
      // Primitive tracer = wind mass fraction, clipped against numerical overshoots
      real f_wind     = FMAX(ZERO_F, FMIN(ONE_F, Vc(TRG,k,j,i)));
      if(f_wind < wind_threshold) {
        Uc(ENG,k,j,i) += dt*rho*rho*lambda;
      }
  });
}
void WindZBC(Hydro *hydro, int dir, BoundarySide side, const real t){
  auto *data = hydro->data;
  if(dir==JDIR && side==right) {
    IdefixArray4D<real> Vc = hydro->Vc;
    // Time-dependent CGM flowing in at z_max
    real ncgm_t     = N_CGM(t, ncgm0, dt_ncgm);
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
        Vc(VX2,k,j,i) = -1*vbh_;
        Vc(VX3,k,j,i) = ZERO_F;
        Vc(PRS,k,j,i) = pres_cgm_;
        Vc(TRG,k,j,i) = 0.0;
      });
  }
}
// ---------------------------------------------------------------------------
// Sink: after each full step, drain gas inside r < eps down to rho_sink.
// rho and P are scaled together, so v and T are kept and mass, momentum
// and energy are removed consistently. Vc is enough: every stage starts
// with PrimToCons.
// ---------------------------------------------------------------------------
// void ApplySink(DataBlock &data, const real t, const real dt) {
//   IdefixArray4D<real> Vc = data.hydro->Vc;
//   IdefixArray3D<real> dV = data.dV;
//   IdefixArray1D<real> x1 = data.x[IDIR];  // R
//   IdefixArray1D<real> x2 = data.x[JDIR];  // z
//   real rs2       = eps*eps;               // sink radius = softening length
//   real rho_sink_ = rho_sink;

//   // Mass removed this step
//   real dM = 0;
//   idefix_reduce("SinkMass",
//     data.beg[KDIR], data.end[KDIR],
//     data.beg[JDIR], data.end[JDIR],
//     data.beg[IDIR], data.end[IDIR],
//     KOKKOS_LAMBDA (int k, int j, int i, real &acc) {
//       if(x1(i)*x1(i) + x2(j)*x2(j) < rs2 && Vc(RHO,k,j,i) > rho_sink_)
//         acc += (Vc(RHO,k,j,i) - rho_sink_)*dV(k,j,i);
//     },
//     Kokkos::Sum<real>(dM));
//   #ifdef WITH_MPI
//     MPI_Allreduce(MPI_IN_PLACE, &dM, 1, realMPI, MPI_SUM, MPI_COMM_WORLD);
//   #endif
//   dM  *= 2.0*M_PI;    // cylindrical dV = R dR dz (no 2pi)
//   Macc += dM;
//   Mdot  = dM/dt;

//   // Drain the sink
//   idefix_for("Sink",
//     data.beg[KDIR], data.end[KDIR],
//     data.beg[JDIR], data.end[JDIR],
//     data.beg[IDIR], data.end[IDIR],
//     KOKKOS_LAMBDA (int k, int j, int i) {
//       if(x1(i)*x1(i) + x2(j)*x2(j) < rs2 && Vc(RHO,k,j,i) > rho_sink_) {
//         #if HAVE_ENERGY
//           Vc(PRS,k,j,i) *= rho_sink_/Vc(RHO,k,j,i);
//         #endif
//         Vc(RHO,k,j,i) = rho_sink_;
//       }
//     });
// }

// // Append t, Mdot, Macc (code units) to sink.dat at every analysis output
// void SinkAnalysis(DataBlock &data) {
//   if(idfx::prank == 0) {
//     std::ofstream f("sink.dat", std::ios::app);
//     f << std::scientific << data.t << "\t" << Mdot << "\t" << Macc << "\n";
//   }
// }
}//namespace