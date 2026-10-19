// ***********************************************************************************
// Idefix MHD astrophysical code
//
// Source file src/fluid/cooling.cpp
//
// Last modified : 10/2026
//
// Copyright(C) by :
// - Geoffroy Lesur <geoffroy.lesur@univ-grenoble-alpes.fr> (IPAG/UGA/CNRS - 2023 - 2025)
// and other code contributors
//
// Licensed under CeCILL 2.1 License, see COPYING for more information
// ***********************************************************************************
#include "cooling.hpp"
#include "dataBlock.hpp"
#include "units.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

void Cooling::ShowConfig() {
  idfx::cout << "Cooling: ENABLED with ISM cooling function (Schure+09/Koyama&Inutsuka02)."
             << std::endl;
  idfx::cout << "Cooling: mu=" << mu << ", X=" << XH
             << " (rate = n_e n_H Lambda, n_e n_H/n^2="
             << 0.5*XH*(1.0+XH)*std::pow(mu*idfx::units.u/idfx::units.m_p, 2) << ")"
             << std::endl;
  if(Tfloor > 0) {
    idfx::cout << "Cooling: temperature floor Tfloor=" << Tfloor << " K." << std::endl;
  } else {
    idfx::cout << "Cooling: no temperature floor." << std::endl;
  }
  if(heatingRate != 0) {
    idfx::cout << "Cooling: constant heating rate=" << heatingRate << " erg/s per particle."
               << std::endl;
  }
  if(haveMask) {
    idfx::cout << "Cooling: masked where tracer " << maskTracer << " < " << maskThreshold
               << "." << std::endl;
  }
  if(subcycle) {
    idfx::cout << "Cooling: operator-split sub-cycling after each hydro step, nsub_max="
               << nsubMax << ", cool_cfl=" << coolCfl << "." << std::endl;
    if(useTable) {
      idfx::cout << "Cooling: tabulated Lambda(T) (float), " << tabN << " entries."
                 << std::endl;
    }
    idfx::cout << "Cooling: timestep capped by nsub_max*cool_cfl*t_cool." << std::endl;
  } else {
    idfx::cout << "Cooling: timestep limited by CFL*min(t_hydro, t_cool)." << std::endl;
  }
}

Cooling::Coefs Cooling::GetCoefs() const {
  // Conversion to code units, same conventions as AthenaK
  // (computed at each call in case the setup changes the units after construction)
  Coefs c;
  c.tempUnit = mu*idfx::units.GetKelvin();   // T(K) = tempUnit * P/rho
  const real nUnit = idfx::units.GetDensity()/(mu*idfx::units.u);
  const real powerUnit = idfx::units.GetEnergy()/idfx::units.GetTime();  // erg/cm^3/s
  const real coolingUnit = powerUnit/(nUnit*nUnit);
  const real heatingUnit = powerUnit/nUnit;
  // n_e n_H = neHfac * n^2, with n = rho*nUnit the total particle density
  const real nRatio = mu*idfx::units.u/idfx::units.m_p;
  const real neHfac = HALF_F*XH*(ONE_F+XH)*nRatio*nRatio;
  c.lambdaUnit = neHfac/coolingUnit;
  c.gheat = heatingRate/heatingUnit;
  c.Tfloor = Tfloor;
  c.haveFloor = (Tfloor > 0);
  return c;
}

void Cooling::AddCooling(const real dt) {
  idfx::pushRegion("Cooling::AddCooling");

  auto Uc = this->Uc;
  auto Vc = this->Vc;
  auto InvDt = this->InvDt;
  auto eos = this->eos;

  const Coefs c = GetCoefs();
  const real tempUnit = c.tempUnit;
  const real lambdaUnit = c.lambdaUnit;
  const real gheat = c.gheat;
  const real Tfloor = c.Tfloor;
  const bool haveFloor = c.haveFloor;
  const bool haveMask = this->haveMask;
  const int maskIndex = this->maskIndex;
  const real maskThreshold = this->maskThreshold;

  idefix_for("AddCooling",
             data->beg[KDIR],data->end[KDIR],
             data->beg[JDIR],data->end[JDIR],
             data->beg[IDIR],data->end[IDIR],
    KOKKOS_LAMBDA (int k, int j, int i) {
      if(haveMask && Vc(maskIndex,k,j,i) < maskThreshold) return;
      // Rates are computed from the primitive variables at the beginning of the stage
      const real rho = Vc(RHO,k,j,i);
      const real prs = Vc(PRS,k,j,i);
      const real temp = tempUnit*prs/rho;
      const real lambda = lambdaUnit*ISMCoolFn(temp);

      // Net energy loss rate (>0 means cooling)
      const real rate = rho*(rho*lambda - gheat);
      real dE = dt*rate;
      bool atFloor = false;

      if(haveFloor) {
        // Internal energy currently stored in Uc (after the flux and other source terms)
        const real rhoU = Uc(RHO,k,j,i);
        real eint = Uc(ENG,k,j,i)
                    - HALF_F/rhoU * (EXPAND( Uc(MX1,k,j,i)*Uc(MX1,k,j,i)  ,
                                           + Uc(MX2,k,j,i)*Uc(MX2,k,j,i)  ,
                                           + Uc(MX3,k,j,i)*Uc(MX3,k,j,i)  ));
        #if MHD == YES
          eint -= HALF_F * (EXPAND( Uc(BX1,k,j,i)*Uc(BX1,k,j,i)  ,
                                  + Uc(BX2,k,j,i)*Uc(BX2,k,j,i)  ,
                                  + Uc(BX3,k,j,i)*Uc(BX3,k,j,i)  ));
        #endif
        const real efloor = eos.GetInternalEnergy(rhoU*Tfloor/tempUnit, rhoU);
        const real dEmax = (temp <= Tfloor) ? ZERO_F       // No cooling below the floor
                                            : FMAX(eint - efloor, ZERO_F);
        // Never cool below the floor (heating is still allowed)
        if(dE > dEmax) {
          dE = dEmax;
          atFloor = true;
        }
      }
      Uc(ENG,k,j,i) -= dE;

      // Timestep constraint: dt_cool = e_int/|dE/dt|, so that dt = CFL*min(dt_hydro, dt_cool)
      // Cells held by the temperature floor do not constrain the timestep.
      if(!atFloor) {
        const real eint0 = eos.GetInternalEnergy(prs, rho);
        const real invDtCool = FABS(rate)/eint0;
        InvDt(k,j,i) = FMAX(InvDt(k,j,i), invDtCool);
      }
    });

  idfx::popRegion();
}

// MPI-global minimum of t_cool = e_int/|n_e n_H Lambda - n Gamma| over the active cells.
// Masked cells and cells within 1% of the cooling floor are excluded: they cannot cool
// further and must not drive the number of sub-steps. Returns max(real) if no cell counts.
real Cooling::MinCoolingTime() {
  auto Vc = this->Vc;
  auto eos = this->eos;
  const Coefs c = GetCoefs();
  const real tempUnit = c.tempUnit;
  const real lambdaUnit = c.lambdaUnit;
  const real gheat = c.gheat;
  const real Tfloor = c.Tfloor;
  const bool haveFloor = c.haveFloor;
  const bool haveMask = this->haveMask;
  const int maskIndex = this->maskIndex;
  const real maskThreshold = this->maskThreshold;
  const bool useTab = useTable;
  auto tab = coolTab;
  const float lt0 = tabLogT0, dinv = tabDinv;
  const int ntab = tabN;

  real tmin = std::numeric_limits<real>::max();
  idefix_reduce("Cooling_MinCoolingTime",
                data->beg[KDIR],data->end[KDIR],
                data->beg[JDIR],data->end[JDIR],
                data->beg[IDIR],data->end[IDIR],
    KOKKOS_LAMBDA (int k, int j, int i, real &localMin) {
      if(haveMask && Vc(maskIndex,k,j,i) < maskThreshold) return;
      const real rho = Vc(RHO,k,j,i);
      const real prs = Vc(PRS,k,j,i);
      const real e = eos.GetInternalEnergy(prs, rho);
      const real efl = haveFloor ? eos.GetInternalEnergy(rho*Tfloor/tempUnit, rho) : ZERO_F;
      if(haveFloor && e <= 1.01*efl) return;
      const real lambda = lambdaUnit*CoolLambda(tempUnit*prs/rho, useTab, tab, lt0, dinv, ntab);
      const real net = FABS(rho*(rho*lambda - gheat)) + std::numeric_limits<float>::min();
      localMin = FMIN(e/net, localMin);
    },
    Kokkos::Min<real>(tmin));

  #ifdef WITH_MPI
    if(idfx::psize>1) {
      MPI_SAFE_CALL(MPI_Allreduce(MPI_IN_PLACE, &tmin, 1, realMPI, MPI_MIN, MPI_COMM_WORLD));
    }
  #endif
  return tmin;
}

// Operator-split explicit cooling over the full hydro step dt.
// (1) t_min = global minimum cooling time of the post-hydro state
// (2) nsub = min(ceil(dt/(cool_cfl*t_min)), nsub_max), identical on every rank
// (3) one kernel: every cell performs exactly nsub explicit sub-steps of dt/nsub. The loop
//     count is uniform, so there is no warp divergence, and e_int stays in a register.
// Must be called after the RK stages, when Uc and Vc are consistent. Only the internal
// energy changes: Vc(PRS) is updated and Uc(ENG) receives the same energy change. Ghost
// cells are not touched: they are refilled by SetBoundaries() at the next stage.
void Cooling::SubcycleCooling(const real dt) {
  idfx::pushRegion("Cooling::SubcycleCooling");

  // (1) global minimum cooling time
  const real tmin = MinCoolingTime();

  // (2) number of sub-steps, same for every cell
  int nsub = 1;
  if(tmin < std::numeric_limits<real>::max()) {
    const real ratio = dt/(coolCfl*tmin);
    // guard the real->int conversion against overflow before clamping
    nsub = (ratio >= static_cast<real>(nsubMax)) ? nsubMax
                                                 : static_cast<int>(std::ceil(ratio));
  }
  nsub = std::max(1, std::min(nsub, nsubMax));
  lastNsub = nsub;
  const real dts = dt/static_cast<real>(nsub);

  // (3) sub-cycled update, fused with the reduction giving the post-cooling t_cool,min
  auto Uc = this->Uc;
  auto Vc = this->Vc;
  auto eos = this->eos;
  const Coefs c = GetCoefs();
  const real tempUnit = c.tempUnit;
  const real lambdaUnit = c.lambdaUnit;
  const real gheat = c.gheat;
  const real Tfloor = c.Tfloor;
  const bool haveFloor = c.haveFloor;
  const bool haveMask = this->haveMask;
  const int maskIndex = this->maskIndex;
  const real maskThreshold = this->maskThreshold;
  const bool useTab = useTable;
  auto tab = coolTab;
  const float lt0 = tabLogT0, dinv = tabDinv;
  const int ntab = tabN;

  real tpost = std::numeric_limits<real>::max();
  idefix_reduce("Cooling_Subcycle",
                data->beg[KDIR],data->end[KDIR],
                data->beg[JDIR],data->end[JDIR],
                data->beg[IDIR],data->end[IDIR],
    KOKKOS_LAMBDA (int k, int j, int i, real &localMin) {
      // masked cells skip cooling entirely and do not limit the next dt
      if(haveMask && Vc(maskIndex,k,j,i) < maskThreshold) return;
      const real rho = Vc(RHO,k,j,i);
      const real e0 = eos.GetInternalEnergy(Vc(PRS,k,j,i), rho);
      const real efl = haveFloor ? eos.GetInternalEnergy(rho*Tfloor/tempUnit, rho) : ZERO_F;
      // Never cool below the floor, and never push a cell already below it up to the floor
      const real elim = FMIN(e0, efl);
      real e = e0;
      real prs = Vc(PRS,k,j,i);
      for(int s = 0 ; s < nsub ; s++) {
        const real lambda = lambdaUnit*CoolLambda(tempUnit*prs/rho, useTab, tab,
                                                  lt0, dinv, ntab);
        e = FMAX(e - dts*rho*(rho*lambda - gheat), elim);
        prs = eos.GetPressure(e, rho);
      }
      Vc(PRS,k,j,i) = prs;               // primitives stay consistent, no extra C2P needed
      Uc(ENG,k,j,i) += e - e0;           // total energy changes by the same amount

      // post-cooling cooling time of this cell (same definition as MinCoolingTime)
      if(haveFloor && e <= 1.01*efl) return;
      const real lambda = lambdaUnit*CoolLambda(tempUnit*prs/rho, useTab, tab, lt0, dinv, ntab);
      const real net = FABS(rho*(rho*lambda - gheat)) + std::numeric_limits<float>::min();
      localMin = FMIN(e/net, localMin);
    },
    Kokkos::Min<real>(tpost));

  #ifdef WITH_MPI
    if(idfx::psize>1) {
      MPI_SAFE_CALL(MPI_Allreduce(MPI_IN_PLACE, &tpost, 1, realMPI, MPI_MIN, MPI_COMM_WORLD));
    }
  #endif
  tcoolMinPost = tpost;

  idfx::popRegion();
}

// Sub-cycled cooling no longer needs dt < t_cool, only nsub <= nsub_max
real Cooling::GetDtCap() const {
  if(tcoolMinPost >= std::numeric_limits<real>::max()) return tcoolMinPost;
  return static_cast<real>(nsubMax)*coolCfl*tcoolMinPost;
}
