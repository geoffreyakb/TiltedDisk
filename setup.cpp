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
                    cs(k,j,i) = epsilon / sqrt(r(i));
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
                    real cs = epsilon / sqrt(r(i));
                    eta1(k,j,i) = alpha * cs * epsilon * r(i) * Vc(RHO,k,j,i);
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

    real spin = spinGlob;
    // -tilt so that the disk is "rotated" counterclockwise
    real Sx = ZERO_F;
    real Sy = ZERO_F;
    real Sz = spin;

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

void PaczynskiWiitaPotential(DataBlock &data, const real t, IdefixArray1D<real> &x1, IdefixArray1D<real> &x2, IdefixArray1D<real> &x3, IdefixArray3D<real> &phi) {
    idefix_for("PaczynskiWiitaPotential",0,data.np_tot[KDIR],0,data.np_tot[JDIR],0,data.np_tot[IDIR],
        KOKKOS_LAMBDA (int k, int j, int i) {
            real r = x1(i);
            phi(k,j,i) = - 1/(r - 2);
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
    // data.hydro->EnrollUserSourceTerm(&GravitomagneticTerm);
    data.gravity->EnrollPotential(&EinsteinPotential);
    // data.gravity->EnrollPotential(&PaczynskiWiitaPotential);

    analysis = new Analysis(input, grid, data);
    output.EnrollAnalysis(&AnalysisFunction);
}

void Setup::InitFlow(DataBlock &data) {
    DataBlockHost d(data);
    real epsilon = epsilonGlob;
    real tilt = tiltGlob * M_PI / 180.0;    // Conversion in radians
    real densityFloor = densityFloorGlob;

    real r, th, phi;
    real x, y, z;
    real xUnt, yUnt, zUnt;
    real rUnt, thUnt, phiUnt;

    real rhoUnt, VrUnt, VthUnt, VphiUnt;
    real er_ex, er_ey, er_ez;
    real eth_ex, eth_ey, eth_ez;
    real ephi_ex, ephi_ey, ephi_ez;
    real VxUnt, VyUnt, VzUnt;

    real Vx, Vy, Vz;
    real ex_er, ex_eth, ex_ephi;
    real ey_er, ey_eth, ey_ephi;
    real ez_er, ez_eth, ez_ephi;
    real Vr, Vth, Vphi;

    for(int k = 0; k < d.np_tot[KDIR]; k++) {
        for(int j = 0; j < d.np_tot[JDIR]; j++) {
            for(int i = 0; i < d.np_tot[IDIR]; i++) {                
                // Spherical coordinates
                r = d.x[IDIR](i);
                th = d.x[JDIR](j);
                phi = d.x[KDIR](k);
                // Cartesian coordinates
                x = r * sin(th) * cos(phi);
                y = r * sin(th) * sin(phi);
                z = r * cos(th);
                // Rotation around the y-axis (the -tilt is for a clockwise rotation around the y-axis if you set a positive angle)
                xUnt = cos(-tilt)*x + sin(-tilt)*z;
                yUnt = y;
                zUnt = -sin(-tilt)*x + cos(-tilt)*z;
                // Back to spherical coordinates
                rUnt = sqrt(xUnt*xUnt + yUnt*yUnt + zUnt*zUnt);
                thUnt = acos(zUnt/rUnt);
                phiUnt = atan2(yUnt,xUnt);

                // Useful parameters
                real R = rUnt*sin(thUnt);
                real Vk = 1.0/sqrt(R);
                real cs = epsilon/sqrt(R);
                // Physical value in the untilted version of the disk
                rhoUnt = 1.0/(R * sqrt(R)) * exp(1.0/pow(cs,2) * (1/rUnt - 1/R));
                VrUnt = ZERO_F;
                VthUnt  = ZERO_F;
                if (rhoUnt > densityFloorGlob) {
                    VphiUnt = Vk * sqrt(sin(thUnt) - 2.5*pow(epsilon, 2));
                }
                else {
                    rhoUnt = densityFloorGlob;
                    VphiUnt = ZERO_F;
                }

                // Expressing spherical unit vectors as cartesian ones (dot products)
                er_ex = sin(thUnt)*cos(phiUnt);
                er_ey = sin(thUnt)*sin(phiUnt);
                er_ez = cos(thUnt);
                eth_ex = cos(thUnt)*cos(phiUnt);
                eth_ey = cos(thUnt)*sin(phiUnt);
                eth_ez = -sin(thUnt);
                ephi_ex = -sin(phiUnt);
                ephi_ey = cos(phiUnt);
                ephi_ez = ZERO_F;
                // Cartesian untilted velocity
                VxUnt = VrUnt*er_ex + VthUnt*eth_ex + VphiUnt*ephi_ex;
                VyUnt = VrUnt*er_ey + VthUnt*eth_ey + VphiUnt*ephi_ey;
                VzUnt = VrUnt*er_ez + VthUnt*eth_ez + VphiUnt*ephi_ez;
                // Cartesian tilted velocity
                Vx = cos(tilt)*VxUnt + sin(tilt)*VzUnt;
                Vy = VyUnt;
                Vz = -sin(tilt)*VxUnt + cos(tilt)*VzUnt;      

                // Expressing cartesian unit vectors as spherical ones (dot products)
                ex_er = sin(th)*cos(phi);
                ex_eth = cos(th)*cos(phi);
                ex_ephi = -sin(phi);
                ey_er = sin(th)*sin(phi);
                ey_eth = cos(th)*sin(phi);
                ey_ephi = cos(phi);
                ez_er = cos(th);
                ez_eth = -sin(th);
                ez_ephi = ZERO_F;
                // Final spherical velocity
                Vr = Vx*ex_er + Vy*ey_er + Vz*ez_er;
                Vth = Vx*ex_eth + Vy*ey_eth + Vz*ez_eth;
                Vphi = Vx*ex_ephi + Vy*ey_ephi + Vz*ez_ephi;

                d.Vc(RHO,k,j,i) = rhoUnt;
                d.Vc(VX1,k,j,i) = Vr;
                d.Vc(VX2,k,j,i) = Vth;
                d.Vc(VX3,k,j,i) = Vphi;
	        }
        }
    }

    d.SyncToDevice();
}
