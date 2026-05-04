#include "idefix.hpp"
#include "setup.hpp"
#include "analysis.hpp"

real epsilonGlob;
real alphaGlob;
real tiltGlob;
real densityFloorGlob;
real spinGlob;

Analysis *analysis;
void AnalysisFunction(DataBlock &data) {
    analysis->PerformAnalysis(data);
}

void MySoundSpeed(DataBlock &data, const real t, IdefixArray3D<real> &cs) {
    IdefixArray1D<real> r = data.x[IDIR];
    IdefixArray1D<real> th = data.x[JDIR];
    IdefixArray1D<real> phi = data.x[KDIR];
    real epsilon = epsilonGlob;

    idefix_for("MySoundSpeed",0,data.np_tot[KDIR],0,data.np_tot[JDIR],0,data.np_tot[IDIR],
                KOKKOS_LAMBDA (int k, int j, int i) {
                    real R = r(i) * sin(th(j));
                    cs(k,j,i) = epsilon / sqrt(R);
                });
}

void MyViscosity(DataBlock &data, const real t, IdefixArray3D<real> &eta1, IdefixArray3D<real> &eta2) {
    IdefixArray4D<real> Vc = data.hydro->Vc;
    IdefixArray1D<real> r = data.x[IDIR];
    IdefixArray1D<real> th = data.x[JDIR];
    IdefixArray1D<real> phi = data.x[KDIR];
    real epsilon = epsilonGlob;
    real alpha = alphaGlob;

    idefix_for("MyViscosity",0,data.np_tot[KDIR],0,data.np_tot[JDIR],0,data.np_tot[IDIR],
                KOKKOS_LAMBDA (int k, int j, int i) {
                    real R = r(i) * sin(th(j));
                    real cs = epsilon / sqrt(R);
                    eta1(k,j,i) = alpha * cs * epsilon * R * Vc(RHO,k,j,i);
                    eta2(k,j,i) = ZERO_F;
              });
}

void InternalBoundary(Hydro *hydro, const real t) {
    auto *data = hydro->data;
    IdefixArray4D<real> Vc = hydro->Vc;
    real densityFloor = densityFloorGlob;

    idefix_for("InternalBoundary",
                0, data->np_tot[KDIR],
                0, data->np_tot[JDIR],
                0, data->np_tot[IDIR],
                KOKKOS_LAMBDA (int k, int j, int i) {
                    if(Vc(RHO,k,j,i) < densityFloor) {
                        Vc(RHO,k,j,i) = densityFloor;
                    }
                });
}

void GravitomagneticTerm(Hydro *hydro, const real t, const real dtin) {
    auto *data = hydro->data;
    IdefixArray4D<real> Vc = hydro->Vc;
    IdefixArray4D<real> Uc = hydro->Uc;
    IdefixArray1D<real> x1 = data->x[IDIR];
    IdefixArray1D<real> x2 = data->x[JDIR];
    IdefixArray1D<real> x3 = data->x[KDIR];
    real dt = dtin;
    real epsilon = epsilonGlob;
    real alpha = alphaGlob;

    real tilt = tiltGlob * M_PI / 180.0;    // Conversion in radians
    real spin = spinGlob;
    // -tilt so that the disk is "rotated" counterclockwise
    real Sx = spin * sin(-tilt);
    real Sy = ZERO_F;
    real Sz = spin * cos(-tilt);

    idefix_for("GravitomagneticTerm",
        0, data->np_tot[KDIR],
        0, data->np_tot[JDIR],
        0, data->np_tot[IDIR],
        KOKKOS_LAMBDA (int k, int j, int i) {
            real r = x1(i);
            real th = x2(j);
            real phi = x3(k);
            real Vr = Vc(VX1,k,j,i);
            real Vth = Vc(VX2,k,j,i);
            real Vphi = Vc(VX3,k,j,i);

            real Sr = sin(th)*cos(phi)*Sx + sin(th)*sin(phi)*Sy + cos(th)*Sz;
            real Sth = cos(th)*cos(phi)*Sx + cos(th)*sin(phi)*Sy - sin(th)*Sz;
            real Sphi = - sin(phi)*Sx + cos(phi)*Sy;
            real hr = -4*Sr / pow(r,3);
            real hth = 2*Sth / pow(r,3);
            real hphi = 2*Sphi / pow(r,3);
            real Vcrossh_r = Vth*hphi - Vphi*hth;
            real Vcrossh_th = Vphi*hr - Vr*hphi;
            real Vcrossh_phi = Vr*hth - Vth*hr;

            Uc(MX1,k,j,i) += dt * Vc(RHO,k,j,i) * Vcrossh_r;
            Uc(MX2,k,j,i) += dt * Vc(RHO,k,j,i) * Vcrossh_th;
            Uc(MX3,k,j,i) += dt * Vc(RHO,k,j,i) * Vcrossh_phi;
    });
}

void EinsteinPotential(DataBlock &data, const real t, IdefixArray1D<real> &x1, IdefixArray1D<real> &x2, IdefixArray1D<real> &x3, IdefixArray3D<real> &phi) {
    idefix_for("EinsteinPotential",0,data.np_tot[KDIR],0,data.np_tot[JDIR],0,data.np_tot[IDIR],
        KOKKOS_LAMBDA (int k, int j, int i) {
            real r = x1(i);
            phi(k,j,i) = - 1/r - 3/pow(r,2);
    });
}

Setup::Setup(Input &input, Grid &grid, DataBlock &data, Output &output) {
    epsilonGlob = input.Get<real>("Setup", "epsilon", 0);
    alphaGlob = input.Get<real>("Setup", "alpha", 0);
    tiltGlob = input.Get<real>("Setup", "tilt", 0);
    densityFloorGlob = input.Get<real>("Setup", "densityFloor", 0);
    spinGlob = input.Get<real>("Setup", "spin", 0);

    data.hydro->EnrollInternalBoundary(&InternalBoundary);
    data.hydro->EnrollIsoSoundSpeed(&MySoundSpeed);
    data.hydro->viscosity->EnrollViscousDiffusivity(&MyViscosity);
    data.hydro->EnrollUserSourceTerm(&GravitomagneticTerm);
    data.gravity->EnrollPotential(&EinsteinPotential);

    analysis = new Analysis(input, grid, data);
    output.EnrollAnalysis(&AnalysisFunction);
}

void Setup::InitFlow(DataBlock &data) {
    DataBlockHost d(data);
    real epsilon = epsilonGlob;
    real r, th, phi;    

    for(int k = 0; k < d.np_tot[KDIR]; k++) {
        for(int j = 0; j < d.np_tot[JDIR]; j++) {
            for(int i = 0; i < d.np_tot[IDIR]; i++) {                
                r = d.x[IDIR](i);
                th = d.x[JDIR](j);
                phi = d.x[KDIR](k);

                real R = r*sin(th);
                real Vk = 1.0/sqrt(R);
                real cs = epsilon/sqrt(R);

                d.Vc(RHO,k,j,i) = 1.0/(R * sqrt(R)) * exp(1.0/pow(cs,2) * (1/r - 1/R));
                d.Vc(VX1,k,j,i) = ZERO_F;
                d.Vc(VX2,k,j,i) = ZERO_F;
                d.Vc(VX3,k,j,i) = Vk * sqrt(R/r - 2.5*pow(epsilon,2));
            }
        }
    }

    d.SyncToDevice();
}
