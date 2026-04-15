#include "idefix.hpp"
#include "setup.hpp"
#include "analysis.hpp"

real epsilonGlob;
real alphaGlob;
real tiltGlob;
real densityFloorGlob;

Analysis *analysis;
void AnalysisFunction(DataBlock &data) {
    analysis->PerformAnalysis(data);
}

void MySoundSpeed(DataBlock &data, const real t, IdefixArray3D<real> &cs) {
    IdefixArray1D<real> r = data.x[IDIR];
    IdefixArray1D<real> th = data.x[JDIR];
    IdefixArray1D<real> phi = data.x[KDIR];
    real epsilon = epsilonGlob;
    real tilt = tiltGlob * M_PI / 180.0;    // Conversion in radians

    idefix_for("MySoundSpeed",0,data.np_tot[KDIR],0,data.np_tot[JDIR],0,data.np_tot[IDIR],
                KOKKOS_LAMBDA (int k, int j, int i) {
                    // Cartesian coordinates
                    real x = r(i) * sin(th(j)) * cos(phi(k));
                    real y = r(i) * sin(th(j)) * sin(phi(k));
                    real z = r(i) * cos(th(j));
                    // Rotation around the x-axis (the -tilt is for a clockwise rotation around the x-axis if you set a positive angle)
                    real xUnt = x;
                    real yUnt = cos(-tilt)*y - sin(-tilt)*z;
                    real zUnt = sin(-tilt)*y + cos(-tilt)*z;
                    // Back to spherical coordinates
                    real rUnt = sqrt(xUnt*xUnt + yUnt*yUnt + zUnt*zUnt);
                    real thUnt = acos(zUnt/rUnt);

                    real R = rUnt * sin(thUnt);
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
    real tilt = tiltGlob * M_PI / 180.0;

    idefix_for("MyViscosity",0,data.np_tot[KDIR],0,data.np_tot[JDIR],0,data.np_tot[IDIR],
                KOKKOS_LAMBDA (int k, int j, int i) {
                    // Cartesian coordinates
                    real x = r(i) * sin(th(j)) * cos(phi(k));
                    real y = r(i) * sin(th(j)) * sin(phi(k));
                    real z = r(i) * cos(th(j));
                    // Rotation around the x-axis (the -tilt is for a clockwise rotation around the x-axis if you set a positive angle)
                    real xUnt = x;
                    real yUnt = cos(-tilt)*y - sin(-tilt)*z;
                    real zUnt = sin(-tilt)*y + cos(-tilt)*z;
                    // Back to spherical coordinates
                    real rUnt = sqrt(xUnt*xUnt + yUnt*yUnt + zUnt*zUnt);
                    real thUnt = acos(zUnt/rUnt);

                    real R = rUnt * sin(thUnt);
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
                        Vc(RHO,k,j,i)=densityFloor;
                    }
                });
}

Setup::Setup(Input &input, Grid &grid, DataBlock &data, Output &output) {
    epsilonGlob = input.Get<real>("Setup", "epsilon", 0);
    alphaGlob = input.Get<real>("Setup", "alpha", 0);
    tiltGlob = input.Get<real>("Setup", "tilt", 0);
    densityFloorGlob = input.Get<real>("Setup", "densityFloor", 0);

    data.hydro->EnrollInternalBoundary(&InternalBoundary);
    data.hydro->EnrollIsoSoundSpeed(&MySoundSpeed);
    data.hydro->viscosity->EnrollViscousDiffusivity(&MyViscosity);

    analysis = new Analysis(input, grid, data);
    output.EnrollAnalysis(&AnalysisFunction);
}

void Setup::InitFlow(DataBlock &data) {
    DataBlockHost d(data);
    real epsilon = epsilonGlob;
    real tilt = tiltGlob * M_PI / 180.0;    // Conversion in radians

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
                // Rotation around the x-axis (the -tilt is for a clockwise rotation around the x-axis if you set a positive angle)
                // xUnt = x;
                // yUnt = cos(-tilt)*y - sin(-tilt)*z;
                // zUnt = sin(-tilt)*y + cos(-tilt)*z;
                xUnt = cos(-tilt)*x + sin(-tilt)*z;
                yUnt = y;
                zUnt = -sin(-tilt)*x + cos(-tilt)*z;
                // Back to spherical coordinates
                rUnt = sqrt(xUnt*xUnt + yUnt*yUnt + zUnt*zUnt);
                thUnt = acos(zUnt/rUnt);
                phiUnt = atan2(yUnt,xUnt);

                // Useful parameters
                real R = rUnt * sin(thUnt);
                real Vk = 1.0 / sqrt(R);
                real cs2 = pow(epsilon / sqrt(R), 2);
                // Physical value in the untilted version of the disk
                rhoUnt = 1.0/(R * sqrt(R)) * exp(1.0/cs2 * (1/rUnt - 1/R));
                VrUnt = ZERO_F;
                VthUnt = ZERO_F;
                VphiUnt = Vk * sqrt(R/rUnt - 2.5*epsilon*epsilon);

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
                // Vx = VxUnt;
                // Vy = cos(tilt)*VyUnt - sin(tilt)*VzUnt;
                // Vz = sin(tilt)*VyUnt + cos(tilt)*VzUnt;    
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
