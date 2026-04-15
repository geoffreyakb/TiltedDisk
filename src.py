import os
import sys
sys.path.append(os.getenv("IDEFIX_DIR"))
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib.ticker as tkr
from matplotlib.ticker import MaxNLocator
from matplotlib.ticker import LogLocator, AutoMinorLocator
from matplotlib.colors import LogNorm
from matplotlib.ticker import LogLocator, LogFormatter
from matplotlib.ticker import NullFormatter
from pytools.vtk_io import readVTK
import inifix
import cv2
import numpy.fft as fft
from scipy.stats import binned_statistic
import csv

def READ_BOX_AVERAGES():
    fid = open("output/analysis/globalAverage.dat", "r")
    varnames = fid.readline().split()
    fid.close()
    data = np.loadtxt("output/analysis/globalAverage.dat",skiprows=1)
    V = {}
    i = 0
    for name in varnames:
        V[name] = data[:,i]
        i += 1

    return V

def READ_VTK(vtk_path, t):
    NVAR = 4
    RHO = 0
    VX1 = 1
    VX2 = 2
    VX3 = 3

    if t >= 1000:
        current_number = str(t)
    elif t >= 100:
        current_number = '0' + str(t)
    elif t >= 10:
        current_number = '00' + str(t)
    else:
        current_number = '000' + str(t)
    current_VTK = readVTK(vtk_path + 'data.' + current_number + '.vtk', geometry='spherical')    
    # print(vars(current_VTK))
    # Coordinates
    r_vtk = current_VTK.r
    theta_vtk = current_VTK.theta
    phi_vtk = current_VTK.phi
    # x,y,z aussi


    vtk = np.zeros((NVAR, phi_vtk.size, theta_vtk.size, r_vtk.size), dtype=float)
    vtk[RHO,:,:,:] = np.moveaxis(current_VTK.data['RHO'], [0, 2], [2, 0])
    vtk[VX1,:,:,:] = np.moveaxis(current_VTK.data['VX1'], [0, 2], [2, 0])
    vtk[VX2,:,:,:] = np.moveaxis(current_VTK.data['VX2'], [0, 2], [2, 0])
    vtk[VX3,:,:,:] = np.moveaxis(current_VTK.data['VX3'], [0, 2], [2, 0])

    rho = vtk[RHO,:,:,:]
    v_r = vtk[VX1,:,:,:]
    v_theta = vtk[VX2,:,:,:]
    v_phi = vtk[VX3,:,:,:]

    return rho, v_r, v_theta, v_phi