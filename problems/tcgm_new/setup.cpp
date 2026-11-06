// ***********************************************************************************
// Idefix MHD astrophysical code
//
// Source file problems/tcgm_new/setup.cpp
// ***********************************************************************************

#include "idefix.hpp"
#include "setup.hpp"
#include "units.hpp"
#include <cmath>
#include <random>
#include <vector>

namespace{
  real GMbh, vbh, eps,
  ncgm0, dt_ncgm, nfloor, Tcgm,
  Mdot,
  rinj, vw0, vwf, dt_vw, Volinj,
  gamma_gas, mu_gas, XH;

  // Bessel-mode turbulence: A_phi = sum_mn a_mn J1(kR_m R) cos(kz_n z' + phi_mn)
  struct TurbField {
    int M{0}, N{0};
    IdefixArray1D<real> kR, kz;      // kR_m = j1m/Rmax, kz_n = 2pi n/Lz
    IdefixArray2D<real> amp, phi;    // a_mn (0 outside [kmin,kmax]) and phases, M x N
    IdefixArray2D<real> J0, J1;      // J0(kR_m R_i), J1(kR_m R_i) on the local R grid, M x nx1
    IdefixArray2D<real> C, S;        // work arrays: z-sums per (m, j), M x nx2
  };
  TurbField *turb = nullptr;         // pointer, so the views are not destroyed after Kokkos::finalize

  void AddSourceTerms(Hydro *hydro, const real t, const real dtin);
  void AddBHGrav(Hydro *hydro, const real t, const real dtin);
  void AddOutflow(Hydro *hydro, const real t, const real dtin);
  void WindZBC(Hydro *hydro, int dir, BoundarySide side, const real t);
  void BuildTurb(Grid &grid, DataBlock &data, real sigv, real lmin, real lmax, real Lz, int seed);
  void FillTurb(DataBlock &data, int jb, int je, real zshift);
  real InjectionVolume(DataBlock &data, const real rinj);
  // Model (design notes §5), Mdot = const:
  //   n_cgm(t) = max(n0 exp(-t/dt_ncgm), n_floor)
  //   v_w(t)   = v_f + (v_w0 - v_f) exp(-t/dt_vw)
  real N_CGM(const real t) {
      return std::fmax(ncgm0*std::exp(-t/dt_ncgm), nfloor);
  }
  real Vw_t(const real t) {
      return vwf + (vw0 - vwf)*std::exp(-t/dt_vw);
  }
  // CGM density and pressure (code units) for a number density n (cm^-3) at T_cgm
  real CGMDensity(const real n) {
      return n*mu_gas*idfx::units.m_p/idfx::units.GetDensity();
  }
  real CGMPressure(const real n) {
      real v0 = idfx::units.GetVelocity();
      return n*idfx::units.k_B*Tcgm/(idfx::units.GetDensity()*v0*v0);
  }
}

// ---------------------------------------------------------------------------
// Constructor: read parameters, convert to code units, enroll functions
// ---------------------------------------------------------------------------
Setup::Setup(Input &input, Grid &grid, DataBlock &data, Output &output) {
  // Physical inputs
  real Mbh_         = input.Get<real>("Setup","Mbh",0);      // M_sun
  real vbh_         = input.Get<real>("Setup","vbh",0);      // code units
  real eps_         = input.Get<real>("Setup","epsilon",0);  // code_units
  real ncgm0_       = input.Get<real>("Setup","n_cgm0",0);   // m_p/cc = code density
  real dt_ncgm_     = input.Get<real>("Setup","dt_ncgm",0);
  real nfloor_      = input.Get<real>("Setup","n_floor",0);  // cm^-3
  real Tcgm_        = input.Get<real>("Setup","T_cgm",0);    // K
  real mu_          = input.Get<real>("Cooling","mu",0);
  real gamma_       = input.Get<real>("Hydro","gamma",0);
  real Mdot_        = input.Get<real>("Setup","Mdot",0);     // Msun/yr, constant
  real vw0_         = input.Get<real>("Setup","vw0",0);
  real vwf_         = input.Get<real>("Setup","vw_floor",0);
  real dt_vw_       = input.Get<real>("Setup","dt_vw",0);
  real rinj_        = input.Get<real>("Setup","rinj",0);
  real XH_          = input.GetOrSet<real>("Cooling","X",0,0.7);
  // Turbulence inputs
  real sigv_        = input.Get<real>("Turb","sigv",0);         //km/s
  real Rnorm_       = input.Get<real>("Turb","Rnorm",0); 
  real lmin_        = input.Get<real>("Turb","lmin",0); 
  real lmax_        = input.Get<real>("Turb","lmax",0);
  real Lz_          = input.Get<real>("Turb","Lz",0); 
  int  seed_        = input.Get<int>("Turb","seed",0); 

  // Conversions (idfx::units is initialised from [Units] before Setup is built)
  real L0           = idfx::units.GetLength();
  real v0           = idfx::units.GetVelocity();
  real rho0         = idfx::units.GetDensity();
  real M0           = rho0*L0*L0*L0;
  real t0           = L0/v0;  

  real Mbh_code     = (Mbh_*idfx::units.M_sun)/M0;
  real G_code       = idfx::units.G*rho0*t0*t0;
  real Mdot_cgs     = (Mdot_*idfx::units.M_sun)/idfx::units.s_yr;
  real Mdot_code    = Mdot_cgs*t0/M0;
  real sigv_code    = sigv_*1e5/v0;

  GMbh              = G_code*Mbh_code;
  vbh               = vbh_;
  eps               = eps_;
  ncgm0             = ncgm0_;
  dt_ncgm           = dt_ncgm_;
  nfloor            = nfloor_;
  Tcgm              = Tcgm_;
  Mdot              = Mdot_code;
  rinj              = rinj_;
  vw0               = vw0_;
  vwf               = vwf_;
  dt_vw             = dt_vw_;
  // Actual volume of the cells with r < rinj (design notes §2), so the
  // injected Mdot is exact on the stair-stepped sphere
  Volinj            = InjectionVolume(data, rinj_);
  if(Volinj <= ZERO_F) IDEFIX_ERROR("No cell centre inside rinj: increase rinj or refine the core");
  idfx::cout << "Injection volume: " << Volinj << " (analytic sphere: "
             << 4*M_PI*rinj_*rinj_*rinj_/3 << ")" << std::endl;
  gamma_gas         = gamma_;
  mu_gas            = mu_;
  XH                = XH_;
  BuildTurb(grid, data, sigv_code, lmin_, lmax_, Lz_, seed_);
  data.hydro->EnrollUserDefBoundary(WindZBC);
  data.hydro->EnrollUserSourceTerm(AddSourceTerms);
}

// ---------------------------------------------------------------------------
// Initial condition: uniform wind everywhere+ Outflow
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
        d.Vc(VX3,k,j,i) = ZERO_F;
        d.Vc(PRS,k,j,i) = pres_cgm0;
        d.Vc(TRG,k,j,i) = 0.0;
      }
    }
  }

  d.SyncToDevice();
  // VX1, VX2 are set on the device, after the sync
  FillTurb(data, 0, data.np_tot[JDIR], ZERO_F);
  // Wind inside r < rinj, set after FillTurb so it is not overwritten
  IdefixArray4D<real> Vc = data.hydro->Vc;
  IdefixArray1D<real> x1 = data.x[IDIR];  // R
  IdefixArray1D<real> x2 = data.x[JDIR];  // z
  real rinj_    = rinj;
  real vinj     = Vw_t(ZERO_F)/sqrt(2);
  real rho_w    = Mdot/(4*M_PI*rinj*rinj*vinj);    // free wind at r = rinj
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
  real rinj_    = rinj;
  real gamma_   = gamma_gas;
  real gm1      = gamma_-1;

  real vinj_    = Vw_t(t)/sqrt(2);
  real Srho     = Mdot/Volinj;
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
// m-th positive zero of J1 (Newton from McMahon's expansion)
double BesselJ1Zero(int m) {
    double b = (m + 0.25)*M_PI;
    double x = b - 3.0/(8.0*b);
    for(int it = 0; it < 20; it++) {
        double j1  = std::cyl_bessel_j(1.0, x);
        double dj1 = std::cyl_bessel_j(0.0, x) - j1/x;   // J1' = J0 - J1/x
        x -= j1/dj1;
    }
    return x;
}
//Calculates lattice, A and Phi and stores as arrays
void BuildTurb(Grid &grid, DataBlock &data, real sigv, real lmin, real lmax, real Lz, int seed) {
  const double Rmax = grid.xend[IDIR];
  const double kmin = 2*M_PI/lmax, kmax = 2*M_PI/lmin;

  // Lattice: kR_m at J1 zeros, kz_n = 2pi n/Lz with n >= 1 (no kz = 0 modes)
  std::vector<double> kR;
  for(int m = 1; BesselJ1Zero(m)/Rmax <= kmax; m++) kR.push_back(BesselJ1Zero(m)/Rmax);
  const int M = kR.size();
  const int N = static_cast<int>(kmax*Lz/(2*M_PI));

  // Fixed amplitudes w_mn = k^{-7/3} sqrt(pi kR/Rmax), random phases.
  // A phase is drawn for every lattice point, so all ranks get the same sequence.
  std::mt19937 gen(seed);
  std::uniform_real_distribution<double> unif(0.0, 2*M_PI);
  std::vector<double> amp(M*N, 0.0), phi(M*N, 0.0);
  double S = 0;
  for(int m = 0; m < M; m++) {
    double J0j = std::cyl_bessel_j(0.0, kR[m]*Rmax);
    for(int n = 0; n < N; n++) {
      double kz = 2*M_PI*(n+1)/Lz;
      double k  = std::sqrt(kR[m]*kR[m] + kz*kz);
      phi[m*N+n] = unif(gen);
      if(k < kmin || k > kmax) continue;
      double w = std::pow(k, -7.0/3.0)*std::sqrt(M_PI*kR[m]/Rmax);
      amp[m*N+n] = w;
      S += 0.5*w*w*k*k*J0j*J0j;          // volume-averaged v^2 of this mode (Parseval)
    }
  }
  const double A = sigv/std::sqrt(S);
  idfx::cout << "Turbulence: " << M << " x " << N << " lattice, A = " << A << std::endl;

  // Device arrays
  const int nx1 = data.np_tot[IDIR], nx2 = data.np_tot[JDIR];
  turb = new TurbField;
  turb->M = M;  turb->N = N;
  turb->kR  = IdefixArray1D<real>("turb_kR", M);
  turb->kz  = IdefixArray1D<real>("turb_kz", N);
  turb->amp = IdefixArray2D<real>("turb_amp", M, N);
  turb->phi = IdefixArray2D<real>("turb_phi", M, N);
  turb->J0  = IdefixArray2D<real>("turb_J0", M, nx1);
  turb->J1  = IdefixArray2D<real>("turb_J1", M, nx1);
  turb->C   = IdefixArray2D<real>("turb_C", M, nx2);
  turb->S   = IdefixArray2D<real>("turb_S", M, nx2);

  auto kRH  = Kokkos::create_mirror_view(turb->kR);
  auto kzH  = Kokkos::create_mirror_view(turb->kz);
  auto ampH = Kokkos::create_mirror_view(turb->amp);
  auto phiH = Kokkos::create_mirror_view(turb->phi);
  auto J0H  = Kokkos::create_mirror_view(turb->J0);
  auto J1H  = Kokkos::create_mirror_view(turb->J1);
  auto x1H  = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), data.x[IDIR]);

  for(int n = 0; n < N; n++) kzH(n) = 2*M_PI*(n+1)/Lz;
  for(int m = 0; m < M; m++) {
    kRH(m) = kR[m];
    for(int n = 0; n < N; n++) {
      ampH(m,n) = A*amp[m*N+n];
      phiH(m,n) = phi[m*N+n];
    }
    // Ghost cells at R < 0: J1 is odd, J0 even (cyl_bessel_j needs x >= 0)
    for(int i = 0; i < nx1; i++) {
      double R   = x1H(i);
      double sgn = (R < 0) ? -1.0 : 1.0;
      J0H(m,i) = std::cyl_bessel_j(0.0, kR[m]*std::fabs(R));
      J1H(m,i) = sgn*std::cyl_bessel_j(1.0, kR[m]*std::fabs(R));
    }
  }
  Kokkos::deep_copy(turb->kR, kRH);
  Kokkos::deep_copy(turb->kz, kzH);
  Kokkos::deep_copy(turb->amp, ampH);
  Kokkos::deep_copy(turb->phi, phiH);
  Kokkos::deep_copy(turb->J0, J0H);
  Kokkos::deep_copy(turb->J1, J1H);
}
// Set VX1, VX2 in rows [jb, je) to -vbh z_hat + turbulence at z' = z + zshift
void FillTurb(DataBlock &data, int jb, int je, real zshift) {
  IdefixArray4D<real> Vc = data.hydro->Vc;
  IdefixArray1D<real> x2 = data.x[JDIR];
  auto kR = turb->kR;   auto kz = turb->kz;
  auto amp = turb->amp; auto phi = turb->phi;
  auto J0 = turb->J0;   auto J1 = turb->J1;
  auto C = turb->C;     auto S = turb->S;
  const int M = turb->M, N = turb->N;
  const real vbh_ = vbh;

  // 1) z-sums for each radial mode m and row j:
  //    C_m = sum_n a kz sin(th),  S_m = kR sum_n a cos(th)
  idefix_for("Turb_zsum", jb, je, 0, M,
    KOKKOS_LAMBDA (int j, int m) {
      real zp = x2(j) + zshift;
      real c = ZERO_F, s = ZERO_F;
      for(int n = 0; n < N; n++) {
        real th = fmod(kz(n)*zp, 2*M_PI) + phi(m,n);   // keep the argument small
        c += amp(m,n)*kz(n)*sin(th);
        s += amp(m,n)*cos(th);
      }
      C(m,j) = c;
      S(m,j) = kR(m)*s;
  });

  // 2) radial sums: v_R = sum_m J1 C_m,  v_z = sum_m J0 S_m
  idefix_for("Turb_rsum", 0, data.np_tot[KDIR], jb, je, 0, data.np_tot[IDIR],
    KOKKOS_LAMBDA (int k, int j, int i) {
      real vR = ZERO_F, vz = ZERO_F;
      for(int m = 0; m < M; m++) {
        vR += J1(m,i)*C(m,j);
        vz += J0(m,i)*S(m,j);
      }
      Vc(VX1,k,j,i) = vR;
      Vc(VX2,k,j,i) = -vbh_ + vz;
  });
}

void WindZBC(Hydro *hydro, int dir, BoundarySide side, const real t){
  auto *data = hydro->data;
  if(dir==JDIR && side==right) {
    IdefixArray4D<real> Vc = hydro->Vc;
    // Time-dependent CGM flowing in at z_max
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
        Vc(VX3,k,j,i) = ZERO_F;
        Vc(PRS,k,j,i) = pres_cgm_;
        Vc(TRG,k,j,i) = 0.0;
      });
      // Gas moves at -vbh, so the frozen pattern is f(z + vbh t)
      FillTurb(*data, data->end[JDIR], data->np_tot[JDIR], vbh*t);
  }
}
}//namespace