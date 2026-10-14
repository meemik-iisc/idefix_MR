// ***********************************************************************************
// Idefix MHD astrophysical code
//
// Source file problems/RunawayBH/setup.cpp
// ***********************************************************************************

#include "idefix.hpp"
#include "setup.hpp"
#include "units.hpp"


namespace{
  real GMbh, vbh, eps, rho_cgm, rho_sink, pres_cgm, gm1;
  real Macc{0}, Mdot{0};         // accreted mass and rate (code units)
  void AddBHGrav(Hydro *hydro, const real t, const real dtin);
  void WindZBC(Hydro *hydro, int dir, BoundarySide side, const real t);
  void ApplySink(DataBlock &data, const real t, const real dt);
  void SinkAnalysis(DataBlock &data);
} //namespace

// ---------------------------------------------------------------------------
// Constructor: read parameters, convert to code units, enroll functions
// ---------------------------------------------------------------------------
Setup::Setup(Input &input, Grid &grid, DataBlock &data, Output &output) {
  // Physical inputs
  real Mbh_       = input.Get<real>("Setup","Mbh",0);      // M_sun
  real vbh_       = input.Get<real>("Setup","vbh",0);      // code units
  real eps_       = input.Get<real>("Setup","epsilon",0);  // pc
  real ncgm_      = input.Get<real>("Setup","n_cgm",0);    // m_p/cc = code density
  real sink_rat_  = input.Get<real>("Setup","sink_rat",0); // rho_sink / rho_cgm
  real Tcgm_      = input.Get<real>("Setup","T_cgm",0);    // K
  real mu_        = input.Get<real>("Setup","mu",0);
  real gamma_     = input.Get<real>("Hydro","gamma",0);

  // Conversions (idfx::units is initialised from [Units] before Setup is built)
  real L0         = idfx::units.GetLength();
  real v0         = idfx::units.GetVelocity();
  real rho0       = idfx::units.GetDensity();
  real M0         = rho0*L0*L0*L0;
  real t0         = L0/v0;  

  real Mbh_code   = (Mbh_*idfx::units.M_sun)/M0;
  real G_code     = idfx::units.G*rho0*t0*t0;
  GMbh            = G_code*Mbh_code;
  vbh             = vbh_;
  eps             = eps_;
  rho_cgm         = ncgm_*mu_*idfx::units.m_p/rho0;
  pres_cgm        = (ncgm_*idfx::units.k_B*Tcgm_)/(rho0*v0*v0);
  gm1             = gamma_-1;
  rho_sink        = sink_rat_*rho_cgm;

  // idfx::cout << "RunawayBH: GM   = " << GMGlob  << " (code)" << std::endl;
  // idfx::cout << "RunawayBH: eps  = " << epsGlob << " (code)" << std::endl;
  // idfx::cout << "RunawayBH: rho  = " << rhoGlob << "  P = " << prsGlob << std::endl;
  // idfx::cout << "RunawayBH: vz   = " << vzGlob  << std::endl;

  data.hydro->EnrollUserDefBoundary(WindZBC);
  data.hydro->EnrollUserSourceTerm(AddBHGrav);
  data.EnrollUserStepLast(ApplySink);
  output.EnrollAnalysis(SinkAnalysis);
}

// ---------------------------------------------------------------------------
// Initial condition: uniform wind everywhere
// ---------------------------------------------------------------------------
void Setup::InitFlow(DataBlock &data) {
  DataBlockHost d(data);
  real rho_cgm_   = rho_cgm;
  real pres_cgm_  = pres_cgm;
  real vbh_       = vbh;
  for(int k = 0; k < d.np_tot[KDIR] ; k++) {
    for(int j = 0; j < d.np_tot[JDIR] ; j++) {
      for(int i = 0; i < d.np_tot[IDIR] ; i++) {
        d.Vc(RHO,k,j,i) = rho_cgm_;
        d.Vc(VX1,k,j,i) = ZERO_F;
        d.Vc(VX2,k,j,i) = -1*vbh_;
        d.Vc(VX3,k,j,i) = ZERO_F;
        d.Vc(PRS,k,j,i) = pres_cgm_;
      }
    }
  }

  d.SyncToDevice();
}

void MakeAnalysis(DataBlock & data) {
}

namespace{
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

  idefix_for("SourceTerms",
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

void WindZBC(Hydro *hydro, int dir, BoundarySide side, const real t){
  auto *data = hydro->data;
  if(dir==JDIR && side==right) {
    IdefixArray4D<real> Vc = hydro->Vc;
    real rho_cgm_   = rho_cgm;
    real pres_cgm_  = pres_cgm;
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
      });
  }
}
// ---------------------------------------------------------------------------
// Sink: after each full step, drain gas inside r < eps down to rho_sink.
// rho and P are scaled together, so v and T are kept and mass, momentum
// and energy are removed consistently. Vc is enough: every stage starts
// with PrimToCons.
// ---------------------------------------------------------------------------
void ApplySink(DataBlock &data, const real t, const real dt) {
  IdefixArray4D<real> Vc = data.hydro->Vc;
  IdefixArray3D<real> dV = data.dV;
  IdefixArray1D<real> x1 = data.x[IDIR];  // R
  IdefixArray1D<real> x2 = data.x[JDIR];  // z
  real rs2       = eps*eps;               // sink radius = softening length
  real rho_sink_ = rho_sink;

  // Mass removed this step
  real dM = 0;
  idefix_reduce("SinkMass",
    data.beg[KDIR], data.end[KDIR],
    data.beg[JDIR], data.end[JDIR],
    data.beg[IDIR], data.end[IDIR],
    KOKKOS_LAMBDA (int k, int j, int i, real &acc) {
      if(x1(i)*x1(i) + x2(j)*x2(j) < rs2 && Vc(RHO,k,j,i) > rho_sink_)
        acc += (Vc(RHO,k,j,i) - rho_sink_)*dV(k,j,i);
    },
    Kokkos::Sum<real>(dM));
  #ifdef WITH_MPI
    MPI_Allreduce(MPI_IN_PLACE, &dM, 1, realMPI, MPI_SUM, MPI_COMM_WORLD);
  #endif
  dM  *= 2.0*M_PI;    // cylindrical dV = R dR dz (no 2pi)
  Macc += dM;
  Mdot  = dM/dt;

  // Drain the sink
  idefix_for("Sink",
    data.beg[KDIR], data.end[KDIR],
    data.beg[JDIR], data.end[JDIR],
    data.beg[IDIR], data.end[IDIR],
    KOKKOS_LAMBDA (int k, int j, int i) {
      if(x1(i)*x1(i) + x2(j)*x2(j) < rs2 && Vc(RHO,k,j,i) > rho_sink_) {
        #if HAVE_ENERGY
          Vc(PRS,k,j,i) *= rho_sink_/Vc(RHO,k,j,i);
        #endif
        Vc(RHO,k,j,i) = rho_sink_;
      }
    });
}

// Append t, Mdot, Macc (code units) to sink.dat at every analysis output
void SinkAnalysis(DataBlock &data) {
  if(idfx::prank == 0) {
    std::ofstream f("sink.dat", std::ios::app);
    f << std::scientific << data.t << "\t" << Mdot << "\t" << Macc << "\n";
  }
}
}//namespace