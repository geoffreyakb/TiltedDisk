#ifndef ANALYSIS_HPP_
#define ANALYSIS_HPP_

#include "idefix.hpp"
#include "input.hpp"
#include "output.hpp"
#include "grid.hpp"
#include "dataBlock.hpp"
#include "dataBlockHost.hpp"
#include <vector>
#include <string>
#include <iostream>
#include <iomanip>
#include <fstream>

class Analysis {
    public:
        Analysis(Input& input, Grid& grid, DataBlock& data);
        void ResetAnalysis();
        void PerformAnalysis(DataBlock& );

    private:
        void ComputeTilt(IdefixHostArray4D<real> Vin, std::vector<real> &outCos, std::vector<real> &outSin);
        void WriteRadialAverage(std::vector<real> &data, std::string filename, std::string variable_name);

        DataBlockHost d;
        Grid &grid;

        int countAverage;
        int precision;
        std::ofstream fileRadialAverage;

        real epsilon;
        real alpha;
};

#endif // ANALYSIS_HPP__
