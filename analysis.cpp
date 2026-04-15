#include "analysis.hpp"
#include "idefix.hpp"
#include "fluid.hpp"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <chrono>
#include <sstream>

Analysis::Analysis(Input &input, Grid &grid, DataBlock &data) : grid(grid), d(data) {
    // This hack ensures that d.Vc is an array distinct from data.hydro->Vc, even on CPUs
    this->d.Vc = Kokkos::create_mirror(data.hydro->Vc);
    // Input parameters
    this->epsilon = input.Get<real>("Setup","epsilon",0);
    this->alpha = input.Get<real>("Setup","alpha",0);
    // Output formatting
    this->countAverage = 0;
    this->precision = 10;
    this->column_width = 2*precision;
    this->pathAnalysisFolder = "output/analysis/";
    // Radial averages (to update according to the number of radial profiles you want)
    this->radial_NVARS = 6;
    this->Sigma = 0;
    this->Tilt = 1;
    this->Precession = 2;
    this->Lx = 3;
    this->Ly = 4;
    this->Lz = 5;
    this->radialAverage = IdefixHostArray2D<real> ("radialAverage", radial_NVARS, grid.np_int[IDIR]);
    // Global averages (to update according to the number of global diagnosis you want)
    this->global_NVARS = 0;
    this->globalAverage = IdefixHostArray1D<real> ("globalAverage", global_NVARS);
}

void Analysis::ComputeAngularMomentum(IdefixHostArray4D<real> Vin) {
    for(int i = d.beg[IDIR]; i < d.end[IDIR] ; i++) {
        real r = d.x[IDIR](i);
        real loc_Lx = ZERO_F;
        real loc_Ly = ZERO_F;
        real loc_Lz = ZERO_F;

        for(int k = d.beg[KDIR]; k < d.end[KDIR] ; k++) {
            for(int j = d.beg[JDIR]; j < d.end[JDIR] ; j++) {
                    real th = d.x[JDIR](j);
                    real dth = d.dx[JDIR](j);
                    real phi = d.x[KDIR](k);
                    real dphi = d.dx[KDIR](k);

                    real rCROSSv_r = ZERO_F;
                    real rCROSSv_th = - r * Vin(VX3,k,j,i);
                    real rCROSSv_phi = r * Vin(VX2,k,j,i);

                    real loc_Lr = Vin(RHO,k,j,i) * rCROSSv_r * pow(r,2) * sin(th) * dth * dphi;
                    real loc_Lth = Vin(RHO,k,j,i) * rCROSSv_th * pow(r,2) * sin(th) * dth * dphi;
                    real loc_Lphi = Vin(RHO,k,j,i) * rCROSSv_phi * pow(r,2) * sin(th) * dth * dphi;

                    real er_ex = sin(th)*cos(phi);
                    real er_ey = sin(th)*sin(phi);
                    real er_ez = cos(th);
                    real eth_ex = cos(th)*cos(phi);
                    real eth_ey = cos(th)*sin(phi);
                    real eth_ez = -sin(th);
                    real ephi_ex = -sin(phi);
                    real ephi_ey = cos(phi);
                    real ephi_ez = ZERO_F;
                    loc_Lx += loc_Lr*er_ex + loc_Lth*eth_ex + loc_Lphi*ephi_ex;
                    loc_Ly += loc_Lr*er_ey + loc_Lth*eth_ey + loc_Lphi*ephi_ey;
                    loc_Lz += loc_Lr*er_ez + loc_Lth*eth_ez + loc_Lphi*ephi_ez;
            }
        }

        real glob_Lx, glob_Ly, glob_Lz;
        #ifdef WITH_MPI
            MPI_Reduce(&loc_Lx, &glob_Lx, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
            MPI_Reduce(&loc_Ly, &glob_Ly, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
            MPI_Reduce(&loc_Lz, &glob_Lz, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        #else
            glob_Lx = loc_Lx;
            glob_Ly = loc_Ly;
            glob_Lz = loc_Lz;
        #endif
    
        real norm_L = sqrt(pow(glob_Lx, 2) + pow(glob_Ly, 2) + pow(glob_Lz, 2));
        
        int ip = i - d.beg[IDIR];
        radialAverage(Tilt, ip) = acos(glob_Lz / norm_L) * 180/M_PI;
        // radialAverage(Precession, ip) = atan2(glob_Ly, glob_Lx) * 180/M_PI + 90;           // In Kimmig and Dullemont (2024), they rotate the disk around the y-axis, hence the +90°.
        radialAverage(Precession, ip) = atan2(glob_Ly, glob_Lx) * 180/M_PI;
        radialAverage(Lx, ip) = glob_Lx;
        radialAverage(Ly, ip) = glob_Ly;
        radialAverage(Lz, ip) = glob_Lz;
    }
}

void Analysis::ComputeSurfaceDensity(IdefixHostArray4D<real> Vin) {
    for(int i = d.beg[IDIR]; i < d.end[IDIR] ; i++) {
        real r = d.x[IDIR](i);
        real loc_sigma = ZERO_F;

        for(int j = d.beg[JDIR]; j < d.end[JDIR] ; j++) {
            real th = d.x[JDIR](j);
            real dth = d.dx[JDIR](j);
            
            real rhophi = ZERO_F;
            real sum_dphi = ZERO_F;
            for(int k = d.beg[KDIR]; k < d.end[KDIR] ; k++) {
                real dphi = d.dx[KDIR](k);
                rhophi += Vin(RHO,k,j,i) * dphi;
                sum_dphi += dphi;
            }
            rhophi = rhophi / sum_dphi;     // MPI_REDUCE FOR RHOPHI ??????????????????????????????????

            loc_sigma += rhophi * r * sin(th) * dth;
        }

        real glob_sigma;
        #ifdef WITH_MPI
            MPI_Reduce(&loc_sigma, &glob_sigma, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        #else
            glob_sigma = loc_sigma;
        #endif

        int ip = i - d.beg[IDIR];
        radialAverage(Sigma, ip) = glob_sigma;
    }
}

void Analysis::WriteGlobalAverage(DataBlock &data) {
    fileGlobalAverage.open(pathAnalysisFolder+"globalAverage.dat", std::ios::app);
    fileGlobalAverage.precision(precision);
    if (idfx::prank == 0) {
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
        fileRadialAverage << std::setw(column_width) << "Tilt";
        fileRadialAverage << std::setw(column_width) << "Precession";
        fileRadialAverage << std::setw(column_width) << "Lx";
        fileRadialAverage << std::setw(column_width) << "Ly";
        fileRadialAverage << std::setw(column_width) << "Lz";
        fileRadialAverage << std::endl;

        for (int i = 0 ; i < grid.np_int[IDIR] ; i++) {
            fileRadialAverage << std::scientific << std::setw(column_width) << grid.x[IDIR](i + grid.nghost[IDIR]);
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

    if (idfx::prank == 0) {
        this->start = std::chrono::high_resolution_clock::now();
    }

    // Computing quantities
    ComputeAngularMomentum(d.Vc);
    ComputeSurfaceDensity(d.Vc);

    // Writing averages
    WriteGlobalAverage(data);
    WriteRadialAverage();

    if (idfx::prank == 0) {
        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - this->start);
        std::cout << "Analysis: Write average files n°" << std::to_string(this->countAverage) << "...done in " << std::to_string(duration.count()*1e-9) << " s." << std::endl;
    }

    // Updating count
    this->countAverage++;

    idfx::popRegion();
}
