#include "analysis.hpp"
#include "idefix.hpp"
#include "fluid.hpp"
#include <iostream>
#include <iomanip>
#include <fstream>

Analysis::Analysis(Input &input, Grid &grid, DataBlock &data) : grid(grid), d(data) {
    // This hack ensures that d.Vc is an array distinct from data.hydro->Vc, even on CPUs
    this->d.Vc = Kokkos::create_mirror(data.hydro->Vc);
    // Counting output files
    this->countAverage = 0;
    // Input parameters
    this->epsilon = input.Get<real>("Setup","epsilon",0);
    this->alpha = input.Get<real>("Setup","alpha",0);
}

void Analysis::ComputeTilt(IdefixHostArray4D<real> Vin, std::vector<real> &outCos, std::vector<real> &outSin) {
    std::vector<real> locCos(grid.np_int[IDIR], 0.0);
    std::vector<real> locSin(grid.np_int[IDIR], 0.0);
    std::vector<real> locRho(grid.np_int[IDIR], 0.0);

    // compute int dz dphi rho*th'*cos(phi) and int dz dphi rho*th'*sin(phi)
    // where th' is the latitude angle
    for(int k = d.beg[KDIR]; k < d.end[KDIR] ; k++) {
        real cosphi = cos(d.x[KDIR](k));
        real sinphi = sin(d.x[KDIR](k));
        real dphi = d.dx[KDIR](k);
        for(int j = d.beg[JDIR]; j < d.end[JDIR] ; j++) {
            real thp = M_PI/2-d.x[JDIR](j); // latitude is pi/2-colatitude
            real dth = d.dx[JDIR](j)*cos(thp);
            for(int i = d.beg[IDIR]; i < d.end[IDIR] ; i++) {

                int iglob = i - 2*d.nghost[IDIR] + d.gbeg[IDIR];
                locCos.at(iglob) += thp*Vin(RHO,k,j,i)*dth*dphi*cosphi;
                locSin.at(iglob) += thp*Vin(RHO,k,j,i)*dth*dphi*sinphi;
                locRho.at(iglob) += Vin(RHO,k,j,i)*dth*dphi;
            }
        }
    }

        // Reduce
    #ifdef WITH_MPI
    MPI_Allreduce(locCos.data(), outCos.data(), grid.np_int[IDIR], realMPI ,MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(locSin.data(), outSin.data(), grid.np_int[IDIR], realMPI ,MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, locRho.data(), grid.np_int[IDIR], realMPI ,MPI_SUM, MPI_COMM_WORLD);
    #else
    outCos = locCos;
    outSin = locSin;
    #endif

    // normalisation
    for(int i = 0 ; i < grid.np_int[IDIR] ; i++) {
    outCos[i] /= locRho[i];
    outSin[i] /= locRho[i];
    }
    return ;
}

void Analysis::WriteRadialAverage(std::vector<real> &data, std::string filename, std::string variable_name) {
    if(idfx::prank==0) {
        int precision = 10;
        int col_width = precision + 10; 
        fileRadialAverage.open(filename, std::ios::trunc);
        fileRadialAverage.precision(this->precision);

        fileRadialAverage << std::setw(col_width) << variable_name;

        for (int i = 0; i < grid.np_int[IDIR]; i++) {
            fileRadialAverage << std::scientific << std::setw(col_width) << data[i];
        }

        fileRadialAverage << std::endl;
        fileRadialAverage.close();
    }
}

void Analysis::ResetAnalysis() {
    
}

void Analysis::PerformAnalysis(DataBlock &data) {
    idfx::pushRegion("Analysis::PerformAnalysis");
    if(data.t == 0) {
        this->ResetAnalysis();
    }
    d.SyncFromDevice();

    // Computing quantities
    std::vector<real> tiltCos(grid.np_int[IDIR]);
    std::vector<real> tiltSin(grid.np_int[IDIR]);
    ComputeTilt(d.Vc, tiltCos, tiltSin);

    // Current file names
    std::stringstream current_filename, current_number;
    current_number << std::setfill('0') << std::setw(4) << this->countAverage;
    current_filename << "output/analysis/radialAverages_" << current_number.str() << ".dat";
    std::cout << "Analysis: Write average file " << std::to_string(this->countAverage) << "..." << std::endl;

    // Writing averages
    WriteRadialAverage(tiltCos, current_filename.str(), "tilt_cos");
    WriteRadialAverage(tiltSin, current_filename.str(), "tilt_sin");
    // Updating count
    this->countAverage++;

    idfx::popRegion();
}
