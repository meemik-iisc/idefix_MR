// ***********************************************************************************
// Idefix MHD astrophysical code
//
// Source file problems/RunawayBH/setup.cpp
// ***********************************************************************************

#include "idefix.hpp"
#include "setup.hpp"
#include "units.hpp"


namespace{
  real GMbh, vbh, eps, rho_cgm, pres_cgm, gm1;
  void AddBHGrav(Hydro *hydro, const real t, const real dtin);
  void WindZBC(Hydro *hydro, int dir, BoundarySide side, const real t);
} //namespace

// ---------------------------------------------------------------------------
// Constructor: read parameters, convert to code units, enroll functions
// ---------------------------------------------------------------------------
Setup::Setup(Input &input, Grid &grid, DataBlock &data, Output &output) {
  // Physical inputs
  real Mbh_   = input.Get<real>("Setup","Mbh",0);      // M_sun
  real vbh_   = input.Get<real>("Setup","vbh",0);      // code units
  real eps_   = input.Get<real>("Setup","epsilon",0);  // pc
  real ncgm_  = input.Get<real>("Setup","n_cgm",0);    // m_p/cc = code density
  real Tcgm_  = input.Get<real>("Setup","T_cgm",0);    // K
  real mu_    = input.Get<real>("Setup","mu",0);
  real gamma_ = input.Get<real>("Hydro","gamma",0);

  // Conversions (idfx::units is initialised from [Units] before Setup is built)
  real L0   = idfx::units.GetLength();
  real v0   = idfx::units.GetVelocity();
  real rho0 = idfx::units.GetDensity();
  real M0   = rho0*L0*L0*L0;
  real t0   = L0/v0;  

  real Mbh_code = (Mbh_*idfx::units.M_sun)/M0;
  real G_code   = idfx::units.G*rho0*t0*t0;
  GMbh          = G_code*Mbh_code;
  vbh           = vbh_;
  eps           = eps_;
  rho_cgm       = ncgm_*mu_*idfx::units.m_p/rho0;
  pres_cgm      = (ncgm_*idfx::units.k_B*Tcgm_)/(rho0*v0*v0);
  gm1           = gamma_-1;

  // idfx::cout << "RunawayBH: GM   = " << GMGlob  << " (code)" << std::endl;
  // idfx::cout << "RunawayBH: eps  = " << epsGlob << " (code)" << std::endl;
  // idfx::cout << "RunawayBH: rho  = " << rhoGlob << "  P = " << prsGlob << std::endl;
  // idfx::cout << "RunawayBH: vz   = " << vzGlob  << std::endl;

  data.hydro->EnrollUserDefBoundary(WindZBC);
  data.hydro->EnrollUserSourceTerm(AddBHGrav);
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
}//namespace