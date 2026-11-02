// ***********************************************************************************
// Idefix MHD astrophysical code
//
// Source file problems/turb_box_mirror/setup.cpp
// ***********************************************************************************

#include "idefix.hpp"
#include "setup.hpp"
#include "units.hpp"
#include <cmath>
#include <random>
#include <vector>

namespace{
  real vbh, rho_cgm, pres_cgm, gamma_gas, mu_gas, XH;

  // Mirror plane-wave turbulence: A_phi = sum_mn a_mn sin(kR_m R) cos(kz_n z' + phi_mn)
  struct TurbField {
    int M{0}, N{0};
    IdefixArray1D<real> kR, kz;      // kR_m = pi m/Rmax, kz_n = 2pi n/Lz
    IdefixArray2D<real> amp, phi;    // a_mn (0 outside [kmin,kmax]) and phases, M x N
    IdefixArray2D<real> c0, s1;      // s1 = sin(x), c0 = (1/x) d/dx(x s1) at x = kR_m R_i, M x nx1
    IdefixArray2D<real> C, S;        // work arrays: z-sums per (m, j), M x nx2
  };
  TurbField *turb = nullptr;         // pointer, so the views are not destroyed after Kokkos::finalize

  void WindZBC(Hydro *hydro, int dir, BoundarySide side, const real t);
  void BuildTurb(Grid &grid, DataBlock &data, real sigv, real Rnorm,
                 real lmin, real lmax, real Lz, int seed);
  void FillTurb(DataBlock &data, int jb, int je, real zshift);
}

// ---------------------------------------------------------------------------
// Constructor: read parameters, convert to code units, enroll functions
// ---------------------------------------------------------------------------
Setup::Setup(Input &input, Grid &grid, DataBlock &data, Output &output) {
  // Physical inputs
  real vbh_         = input.Get<real>("Setup","vbh",0);         // code units
  real ncgm_        = input.Get<real>("Setup","n_cgm",0);       // cm^{-3}
  real Tcgm_        = input.Get<real>("Setup","T_cgm",0);       // K
  real mu_          = input.Get<real>("Cooling","mu",0);
  real gamma_       = input.Get<real>("Hydro","gamma",0);
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

  real rho_cgm_     = ncgm_*mu_*idfx::units.m_p/rho0;
  real pres_cgm_    = ncgm_*Tcgm_*idfx::units.k_B/(rho0*v0*v0);
  real sigv_code    = sigv_*1e5/v0;

  vbh               = vbh_;
  rho_cgm           = rho_cgm_;
  pres_cgm          = pres_cgm_;
  gamma_gas         = gamma_;
  mu_gas            = mu_;
  XH                = XH_;
  BuildTurb(grid, data, sigv_code, Rnorm_, lmin_, lmax_, Lz_, seed_);
  data.hydro->EnrollUserDefBoundary(WindZBC);
}

// ---------------------------------------------------------------------------
// Initial condition: uniform wind everywhere
// ---------------------------------------------------------------------------
void Setup::InitFlow(DataBlock &data) {
  DataBlockHost d(data);
  real rho_cgm_   = rho_cgm;
  real pres_cgm_  = pres_cgm;
  for(int k = 0; k < d.np_tot[KDIR] ; k++) {
    for(int j = 0; j < d.np_tot[JDIR] ; j++) {
      for(int i = 0; i < d.np_tot[IDIR] ; i++) {
        d.Vc(RHO,k,j,i) = rho_cgm_;
        d.Vc(VX3,k,j,i) = ZERO_F;
        d.Vc(PRS,k,j,i) = pres_cgm_;
      }
    }
  }

  d.SyncToDevice();
  // VX1, VX2 are set on the device, after the sync
  FillTurb(data, 0, data.np_tot[JDIR], ZERO_F);
}

namespace{
//Calculates lattice, A and Phi and stores as arrays
void BuildTurb(Grid &grid, DataBlock &data, real sigv, real Rnorm,
               real lmin, real lmax, real Lz, int seed) {
  const double Rmax = grid.xend[IDIR];
  const double kmin = 2*M_PI/lmax, kmax = 2*M_PI/lmin;

  // Lattice: kR_m = pi m/Rmax (sin(kR Rmax) = 0, so v_R = 0 at Rmax),
  //          kz_n = 2pi n/Lz with n >= 1 (no kz = 0 modes)
  std::vector<double> kR;
  for(int m = 1; M_PI*m/Rmax <= kmax; m++) kR.push_back(M_PI*m/Rmax);
  const int M = kR.size();
  const int N = static_cast<int>(kmax*Lz/(2*M_PI));

  // Fixed amplitudes w_mn = k^{-7/3}, random phases.
  // A phase is drawn for every lattice point, so all ranks get the same sequence.
  std::mt19937 gen(seed);
  std::uniform_real_distribution<double> unif(0.0, 2*M_PI);
  std::vector<double> amp(M*N, 0.0), phi(M*N, 0.0);
  for(int m = 0; m < M; m++) {
    for(int n = 0; n < N; n++) {
      double kz = 2*M_PI*(n+1)/Lz;
      double k  = std::sqrt(kR[m]*kR[m] + kz*kz);
      phi[m*N+n] = unif(gen);
      if(k < kmin || k > kmax) continue;
      amp[m*N+n] = std::pow(k, -7.0/3.0);
    }
  }

  // Device arrays
  const int nx1 = data.np_tot[IDIR], nx2 = data.np_tot[JDIR];
  turb = new TurbField;
  turb->M = M;  turb->N = N;
  turb->kR  = IdefixArray1D<real>("turb_kR", M);
  turb->kz  = IdefixArray1D<real>("turb_kz", N);
  turb->amp = IdefixArray2D<real>("turb_amp", M, N);
  turb->phi = IdefixArray2D<real>("turb_phi", M, N);
  turb->c0  = IdefixArray2D<real>("turb_c0", M, nx1);
  turb->s1  = IdefixArray2D<real>("turb_s1", M, nx1);
  turb->C   = IdefixArray2D<real>("turb_C", M, nx2);
  turb->S   = IdefixArray2D<real>("turb_S", M, nx2);

  auto kRH  = Kokkos::create_mirror_view(turb->kR);
  auto kzH  = Kokkos::create_mirror_view(turb->kz);
  auto ampH = Kokkos::create_mirror_view(turb->amp);
  auto phiH = Kokkos::create_mirror_view(turb->phi);
  auto c0H  = Kokkos::create_mirror_view(turb->c0);
  auto s1H  = Kokkos::create_mirror_view(turb->s1);
  auto x1H  = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), data.x[IDIR]);

  for(int n = 0; n < N; n++) kzH(n) = 2*M_PI*(n+1)/Lz;
  for(int m = 0; m < M; m++) {
    kRH(m) = kR[m];
    for(int n = 0; n < N; n++) {
      ampH(m,n) = amp[m*N+n];
      phiH(m,n) = phi[m*N+n];
    }
    // s1 is odd and c0 even in R, so ghost cells at R < 0 need no special care
    for(int i = 0; i < nx1; i++) {
      double x = kR[m]*x1H(i);
      s1H(m,i) = std::sin(x);
      c0H(m,i) = (std::fabs(x) < 1e-4) ? 2.0 - 2.0*x*x/3.0 : std::sin(x)/x + std::cos(x);
    }
  }
  Kokkos::deep_copy(turb->kR, kRH);
  Kokkos::deep_copy(turb->kz, kzH);
  Kokkos::deep_copy(turb->amp, ampH);
  Kokkos::deep_copy(turb->phi, phiH);
  Kokkos::deep_copy(turb->c0, c0H);
  Kokkos::deep_copy(turb->s1, s1H);

  // Normalise: the sin modes are not orthogonal under R dR, so measure the
  // volume-averaged v^2 (R < Rnorm, turbulent part only) on the grid and rescale.
  // Vc is only used as scratch here: InitFlow or the restart overwrites it.
  FillTurb(data, 0, data.np_tot[JDIR], ZERO_F);
  IdefixArray4D<real> Vc = data.hydro->Vc;
  IdefixArray3D<real> dV = data.dV;
  IdefixArray1D<real> x1 = data.x[IDIR];
  const real vbh_ = vbh;
  real sumV[2] = {ZERO_F, ZERO_F};
  idefix_reduce("Turb_v2",
    data.beg[KDIR], data.end[KDIR],
    data.beg[JDIR], data.end[JDIR],
    data.beg[IDIR], data.end[IDIR],
    KOKKOS_LAMBDA (int k, int j, int i, real &acc) {
      if(x1(i) < Rnorm) {
        real vz = Vc(VX2,k,j,i) + vbh_;
        acc += (Vc(VX1,k,j,i)*Vc(VX1,k,j,i) + vz*vz)*dV(k,j,i);
      }
    },
    Kokkos::Sum<real>(sumV[0]));
  idefix_reduce("Turb_vol",
    data.beg[KDIR], data.end[KDIR],
    data.beg[JDIR], data.end[JDIR],
    data.beg[IDIR], data.end[IDIR],
    KOKKOS_LAMBDA (int k, int j, int i, real &acc) {
      if(x1(i) < Rnorm) acc += dV(k,j,i);
    },
    Kokkos::Sum<real>(sumV[1]));
  #ifdef WITH_MPI
    MPI_Allreduce(MPI_IN_PLACE, sumV, 2, realMPI, MPI_SUM, MPI_COMM_WORLD);
  #endif
  const real A = sigv/std::sqrt(sumV[0]/sumV[1]);
  auto ampD = turb->amp;
  idefix_for("Turb_scale", 0, M, 0, N,
    KOKKOS_LAMBDA (int m, int n) {
      ampD(m,n) *= A;
  });
  idfx::cout << "Turbulence: " << M << " x " << N << " lattice, A = " << A << std::endl;
}
// Set VX1, VX2 in rows [jb, je) to -vbh z_hat + turbulence at z' = z + zshift
void FillTurb(DataBlock &data, int jb, int je, real zshift) {
  IdefixArray4D<real> Vc = data.hydro->Vc;
  IdefixArray1D<real> x2 = data.x[JDIR];
  auto kR = turb->kR;   auto kz = turb->kz;
  auto amp = turb->amp; auto phi = turb->phi;
  auto c0 = turb->c0;   auto s1 = turb->s1;
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

  // 2) radial sums: v_R = sum_m s1 C_m,  v_z = sum_m c0 S_m
  idefix_for("Turb_rsum", 0, data.np_tot[KDIR], jb, je, 0, data.np_tot[IDIR],
    KOKKOS_LAMBDA (int k, int j, int i) {
      real vR = ZERO_F, vz = ZERO_F;
      for(int m = 0; m < M; m++) {
        vR += s1(m,i)*C(m,j);
        vz += c0(m,i)*S(m,j);
      }
      Vc(VX1,k,j,i) = vR;
      Vc(VX2,k,j,i) = -vbh_ + vz;
  });
}

void WindZBC(Hydro *hydro, int dir, BoundarySide side, const real t){
  auto *data = hydro->data;
  if(dir==JDIR && side==right) {
    IdefixArray4D<real> Vc = hydro->Vc;
    real rho_cgm_   = rho_cgm;
    real pres_cgm_  = pres_cgm;
    idefix_for("InflowZmax",
      0, data->np_tot[KDIR],
      data->end[JDIR], data->np_tot[JDIR],   // ghost cells beyond z_max
      0, data->np_tot[IDIR],
      KOKKOS_LAMBDA (int k, int j, int i) {
        Vc(RHO,k,j,i) = rho_cgm_;
        Vc(VX3,k,j,i) = ZERO_F;
        Vc(PRS,k,j,i) = pres_cgm_;
      });
      // Gas moves at -vbh, so the frozen pattern is f(z + vbh t)
      FillTurb(*data, data->end[JDIR], data->np_tot[JDIR], vbh*t);
  }
}
}//namespace