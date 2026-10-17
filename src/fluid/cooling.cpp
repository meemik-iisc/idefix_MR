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
  idfx::cout << "Cooling: timestep limited by CFL*min(t_hydro, t_cool)." << std::endl;
}

void Cooling::AddCooling(const real dt) {
  idfx::pushRegion("Cooling::AddCooling");

  auto Uc = this->Uc;
  auto Vc = this->Vc;
  auto InvDt = this->InvDt;
  auto eos = this->eos;

  // Conversion to code units, same conventions as AthenaK
  // (computed here in case the setup changes the units after construction)
  const real tempUnit = mu*idfx::units.GetKelvin();   // T(K) = tempUnit * P/rho
  const real nUnit = idfx::units.GetDensity()/(mu*idfx::units.u);
  const real powerUnit = idfx::units.GetEnergy()/idfx::units.GetTime();  // erg/cm^3/s
  const real coolingUnit = powerUnit/(nUnit*nUnit);
  const real heatingUnit = powerUnit/nUnit;
  // n_e n_H = neHfac * n^2, with n = rho*nUnit the total particle density
  const real nRatio = mu*idfx::units.u/idfx::units.m_p;
  const real neHfac = HALF_F*XH*(ONE_F+XH)*nRatio*nRatio;

  const real gheat = heatingRate/heatingUnit;
  const real Tfloor = this->Tfloor;
  const bool haveFloor = (Tfloor > 0);

  idefix_for("AddCooling",
             data->beg[KDIR],data->end[KDIR],
             data->beg[JDIR],data->end[JDIR],
             data->beg[IDIR],data->end[IDIR],
    KOKKOS_LAMBDA (int k, int j, int i) {
      // Rates are computed from the primitive variables at the beginning of the stage
      const real rho = Vc(RHO,k,j,i);
      const real prs = Vc(PRS,k,j,i);
      const real temp = tempUnit*prs/rho;
      const real lambda = neHfac*ISMCoolFn(temp)/coolingUnit;

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
