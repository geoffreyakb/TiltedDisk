#include "analysis.hpp"
#include "idefix.hpp"
#include "fluid.hpp"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <chrono>
#include <sstream>

Analysis::Analysis(Input &input, Grid &grid, DataBlock &data) : grid(grid), gh(grid), d(data) {
    // This hack ensures that d.Vc is an array distinct from data.hydro->Vc, even on CPUs
    this->d.Vc = Kokkos::create_mirror(data.hydro->Vc);
    gh.SyncFromDevice();
    // Input parameters
    this->epsilon = input.Get<real>("Setup","epsilon",0);
    this->alpha = input.Get<real>("Setup","alpha",0);
    // Output formatting
    this->countAverage = 0;
    data.dump->RegisterVariable(&countAverage, "analysis_count_average");
    this->precision = 10;
    this->column_width = 2*precision;
    this->pathAnalysisFolder = "output/analysis/";
    // Radial averages (to update according to the number of radial profiles you want)
    this->radial_NVARS = 11;
    this->Sigma = 0;
    this->Lx = 1;
    this->Ly = 2;
    this->Lz = 3;
    this->Vr = 4;
    this->Vth = 5;
    this->Vphi = 6;
    this->rho = 7;
    this->rho_Vr = 8;
    this->rho_Vth = 9;
    this->rho_Vphi = 10;
    this->radialAverage = IdefixHostArray2D<real> ("radialAverage", radial_NVARS, grid.np_int[IDIR]);
    // Global averages (to update according to the number of global diagnosis you want)
    this->global_NVARS = 1;
    this->Mtot = 0;
    this->globalAverage = IdefixHostArray1D<real> ("globalAverage", global_NVARS);
}

void Analysis::ComputeRadialAverage(IdefixHostArray4D<real> Vin) {
    IdefixHostArray2D<real> loc_radialAverage("loc_radialAverage", radial_NVARS, grid.np_int[IDIR]);

    for(int k = d.beg[KDIR]; k < d.end[KDIR] ; k++) {
        for(int j = d.beg[JDIR]; j < d.end[JDIR] ; j++) {
            for (int i = d.beg[IDIR]; i < d.end[IDIR]; i++){
                real r = d.x[IDIR](i);
                real th = d.x[JDIR](j);
                real dth = d.dx[JDIR](j);
                real phi = d.x[KDIR](k);
                real dphi = d.dx[KDIR](k);
                // Angular momentum in spherical coordinates
                real rCROSSv_r = ZERO_F;
                real rCROSSv_th = - r * Vin(VX3,k,j,i);
                real rCROSSv_phi = r * Vin(VX2,k,j,i);
                real Lr = Vin(RHO,k,j,i) * rCROSSv_r * pow(r,2) * sin(th) * dth * dphi;
                real Lth = Vin(RHO,k,j,i) * rCROSSv_th * pow(r,2) * sin(th) * dth * dphi;
                real Lphi = Vin(RHO,k,j,i) * rCROSSv_phi * pow(r,2) * sin(th) * dth * dphi;
                // Transforming in cartesian coordinates
                real er_ex = sin(th)*cos(phi);
                real er_ey = sin(th)*sin(phi);
                real er_ez = cos(th);
                real eth_ex = cos(th)*cos(phi);
                real eth_ey = cos(th)*sin(phi);
                real eth_ez = -sin(th);
                real ephi_ex = -sin(phi);
                real ephi_ey = cos(phi);
                real ephi_ez = ZERO_F;
                // Filling the local radial averages arrays
                int glob_i = i + d.gbeg[IDIR] - 2*grid.nghost[IDIR];    // -nghost for the global ghosts -nghost for the local ones (i does not start at the "local 0")
                loc_radialAverage(Sigma, glob_i) += Vin(RHO,k,j,i) * r * sin(th) * dth * dphi/(2*M_PI);      // Should be the same definition as Kimmig and Dullemond (2024)
                loc_radialAverage(Lx, glob_i)    += Lr*er_ex + Lth*eth_ex + Lphi*ephi_ex;
                loc_radialAverage(Ly, glob_i)    += Lr*er_ey + Lth*eth_ey + Lphi*ephi_ey;
                loc_radialAverage(Lz, glob_i)    += Lr*er_ez + Lth*eth_ez + Lphi*ephi_ez;

                // Getting the velocity in cartesian coordinates
                real Vx = Vin(VX1,k,j,i)*er_ex + Vin(VX2,k,j,i)*eth_ex + Vin(VX3,k,j,i)*ephi_ex;
                real Vy = Vin(VX1,k,j,i)*er_ey + Vin(VX2,k,j,i)*eth_ey + Vin(VX3,k,j,i)*ephi_ey;
                real Vz = Vin(VX1,k,j,i)*er_ez + Vin(VX2,k,j,i)*eth_ez + Vin(VX3,k,j,i)*ephi_ez;

                real norm = sqrt(pow(Lx,2) + pow(Ly,2) + pow(Lz,2));
                real tilt_loc = acos(Lz / norm);
                // Un-precess the disk only if the value is trust-worthy and not random due to numerical errors (ONLY IN 3D)
                // real precession_loc = atan2(Ly, Lx);
                // if (tilt_loc > 1e-9) {
                    // ...
                // }

                // Unrotating the disk
                real VxUnt = cos(-tilt_loc)*Vx + sin(-tilt_loc)*Vz;
                real VyUnt = Vy;
                real VzUnt = -sin(-tilt_loc)*Vx + cos(-tilt_loc)*Vz;
                // Cartesian coordinates
                real x = r * sin(th) * cos(phi);
                real y = r * sin(th) * sin(phi);
                real z = r * cos(th);
                // Rotation around the y-axis (the -tilt is for a clockwise rotation around the y-axis if you set a positive angle)
                real xUnt = cos(-tilt_loc)*x + sin(-tilt_loc)*z;
                real yUnt = y;
                real zUnt = -sin(-tilt_loc)*x + cos(-tilt_loc)*z;
                // Back to spherical coordinates
                real rUnt = sqrt(xUnt*xUnt + yUnt*yUnt + zUnt*zUnt);
                real thUnt = acos(zUnt/rUnt);
                real phiUnt = atan2(yUnt,xUnt);
                
                real ex_er = sin(thUnt)*cos(phiUnt);
                real ex_eth = cos(thUnt)*cos(phiUnt);
                real ex_ephi = -sin(phiUnt);
                real ey_er = sin(thUnt)*sin(phiUnt);
                real ey_eth = cos(thUnt)*sin(phiUnt);
                real ey_ephi = cos(phiUnt);
                real ez_er = cos(thUnt);
                real ez_eth = -sin(thUnt);
                real ez_ephi = ZERO_F;
                // Passing the velocity in the untilted version of the local blob of the disk
                real VrUnt = VxUnt*ex_er + VyUnt*ey_er + VzUnt*ez_er;
                real VthUnt = VxUnt*ex_eth + VyUnt*ey_eth + VzUnt*ez_eth;
                real VphiUnt = VxUnt*ex_ephi + VyUnt*ey_ephi + VzUnt*ez_ephi;

                loc_radialAverage(Vr, glob_i)       += VrUnt * sin(th) * dth * dphi / (4*M_PI);
                loc_radialAverage(Vth, glob_i)      += VthUnt * sin(th) * dth * dphi / (4*M_PI);
                loc_radialAverage(Vphi, glob_i)     += VphiUnt * sin(th) * dth * dphi / (4*M_PI);
                loc_radialAverage(rho, glob_i)      += Vin(RHO,k,j,i) * sin(th) * dth * dphi / (4*M_PI);  
                loc_radialAverage(rho_Vr, glob_i)    += Vin(RHO,k,j,i) * VrUnt * sin(th) * dth * dphi / (4*M_PI);
                loc_radialAverage(rho_Vth, glob_i)   += Vin(RHO,k,j,i) * VthUnt * sin(th) * dth * dphi / (4*M_PI);
                loc_radialAverage(rho_Vphi, glob_i)  += Vin(RHO,k,j,i) * VphiUnt * sin(th) * dth * dphi / (4*M_PI);
            }
        }
    }

    IdefixHostArray2D<real> glob_radialAverage("glob_radialAverage", radial_NVARS, grid.np_int[IDIR]);
    #ifdef WITH_MPI
        MPI_Allreduce(loc_radialAverage.data(), glob_radialAverage.data(), radial_NVARS*grid.np_int[IDIR], realMPI, MPI_SUM, MPI_COMM_WORLD);
    #else
        glob_radialAverage = loc_radialAverage;
    #endif
    radialAverage = glob_radialAverage;
}

void Analysis::ComputeGlobalAverage(IdefixHostArray4D<real> Vin) {
    IdefixHostArray1D<real> loc_globalAverage("loc_globalAverage", global_NVARS);

    for(int k = d.beg[KDIR]; k < d.end[KDIR] ; k++) {
        for(int j = d.beg[JDIR]; j < d.end[JDIR] ; j++) {
            for (int i = d.beg[IDIR]; i < d.end[IDIR]; i++){
                real r = d.x[IDIR](i);
                real dr = d.dx[IDIR](i);
                real th = d.x[JDIR](j);
                real dth = d.dx[JDIR](j);
                real phi = d.x[KDIR](k);
                real dphi = d.dx[KDIR](k);

                loc_globalAverage(Mtot) += Vin(RHO,k,j,i) * pow(r,2) * sin(th) * dr * dth * dphi;    
            }
        }
    }

    IdefixHostArray1D<real> glob_globalAverage("glob_globalAverage", global_NVARS);
    #ifdef WITH_MPI
        MPI_Allreduce(loc_globalAverage.data(), glob_globalAverage.data(), global_NVARS, realMPI, MPI_SUM, MPI_COMM_WORLD);
    #else
        glob_globalAverage = loc_globalAverage;
    #endif
    globalAverage = glob_globalAverage;
}

void Analysis::WriteGlobalAverage(DataBlock &data) {
    if (idfx::prank == 0) {
        fileGlobalAverage.open(pathAnalysisFolder+"globalAverage.dat", std::ios::app);
        fileGlobalAverage.precision(precision);

        fileGlobalAverage << std::scientific << std::setw(column_width) << data.t;
        for (int VAR = 0; VAR < global_NVARS; VAR++) {
            fileGlobalAverage << std::scientific << std::setw(column_width) << globalAverage(VAR);
        }
        fileGlobalAverage << std::endl;
        fileGlobalAverage.close();
    }
}

void Analysis::WriteRadialAverage() {
    if (idfx::prank == 0) {
        std::stringstream current_number;
        current_number << std::setfill('0') << std::setw(4) << countAverage;
        fileRadialAverage.open(pathAnalysisFolder+"radialAverage_"+current_number.str()+".dat", std::ios::trunc);
        fileRadialAverage.precision(precision);

        fileRadialAverage << std::setw(column_width) << "r";
        fileRadialAverage << std::setw(column_width) << "Sigma";
        fileRadialAverage << std::setw(column_width) << "Lx";
        fileRadialAverage << std::setw(column_width) << "Ly";
        fileRadialAverage << std::setw(column_width) << "Lz";
        fileRadialAverage << std::setw(column_width) << "Vr";
        fileRadialAverage << std::setw(column_width) << "Vth";
        fileRadialAverage << std::setw(column_width) << "Vphi";
        fileRadialAverage << std::setw(column_width) << "rho";
        fileRadialAverage << std::setw(column_width) << "rho_Vr";
        fileRadialAverage << std::setw(column_width) << "rho_Vth";
        fileRadialAverage << std::setw(column_width) << "rho_Vphi";
        fileRadialAverage << std::endl;

        for (int i = 0 ; i < grid.np_int[IDIR] ; i++) {
            fileRadialAverage << std::scientific << std::setw(column_width) << gh.x[IDIR](i + grid.nghost[IDIR]);
            for (int VAR = 0; VAR < radial_NVARS; VAR++) {
                fileRadialAverage << std::scientific << std::setw(column_width) << radialAverage(VAR, i);
            }
            fileRadialAverage << std::endl;
        }
        fileRadialAverage.close();
    }
}

void Analysis::ResetAnalysis() {
    if (idfx::prank == 0) {
        fileGlobalAverage.open(pathAnalysisFolder+"globalAverage.dat", std::ios::trunc);
        fileGlobalAverage << std::setw(column_width) << "t";
        fileGlobalAverage << std::setw(column_width) << "Mtot";
        fileGlobalAverage << std::endl;
        fileGlobalAverage.close();
    }
}

void Analysis::PerformAnalysis(DataBlock &data) {
    idfx::pushRegion("Analysis::PerformAnalysis");
    if(data.t == 0.0) {
        this->ResetAnalysis();
    }
    d.SyncFromDevice();

    // Computing quantities
    ComputeRadialAverage(d.Vc);
    ComputeGlobalAverage(d.Vc);

    // Writing averages
    WriteRadialAverage();
    WriteGlobalAverage(data);

    if (idfx::prank == 0) {
        std::cout << "Analysis: Write average files n°" << std::to_string(this->countAverage) << std::endl;
    }
    // Updating count
    this->countAverage++;

    idfx::popRegion();
}
