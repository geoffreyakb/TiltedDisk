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

def READ_BOX_AVERAGE():
    fid = open("output/analysis/globalAverage.dat", "r")
    varnames = fid.readline().split()
    fid.close()
    data = np.loadtxt("output/analysis/globalAverage.dat",skiprows=1)
    V = {}
    i = 0
    for name in varnames:
        # V[name] = data[:,i]           # TO CHANGE WHENEVER GLOBAL QUANTITIES ARE ADDED
        V[name] = data[:]
        i += 1

    return V["t"]

def READ_RADIAL_AVERAGE(n_average, n_r):
    Sigma = np.zeros((n_average, n_r))
    Tilt = np.zeros((n_average, n_r))
    Precession = np.zeros((n_average, n_r))
    L = np.zeros((n_average, n_r, 3))

    current_number = 0
    for i in range(n_average):
        if i < 10:
            current_number = f"000{i}"
        elif i < 100:
            current_number = f"00{i}"
        elif i < 1000:
            current_number = f"0{i}"
        else:
            current_number = i

        fid=open(f"output/analysis/radialAverage_{current_number}.dat","r")
        varnames=fid.readline().split()
        fid.close()
        data=np.loadtxt(f"output/analysis/radialAverage_{current_number}.dat",skiprows=1)
        V={}
        j=0
        for name in varnames:
            V[name]=data[:,j]
            j=j+1

        Sigma[i,:] = V['Sigma']
        Tilt[i,:] = V['Tilt']
        Precession[i,:] = V['Precession']
        L[i,:,0] = V['Lx']
        L[i,:,1] = V['Ly']
        L[i,:,2] = V['Lz']

    return V["r"], Sigma, Tilt, Precession, L

def READ_VTK(n_vtk):
    NVAR = 4
    RHO = 0
    VX1 = 1
    VX2 = 2
    VX3 = 3

    if n_vtk >= 1000:
        current_number = str(n_vtk)
    elif n_vtk >= 100:
        current_number = '0' + str(n_vtk)
    elif n_vtk >= 10:
        current_number = '00' + str(n_vtk)
    else:
        current_number = '000' + str(n_vtk)
    current_VTK = readVTK('output/vtk/data.' + current_number + '.vtk', geometry='spherical')    
    
    r_vtk = current_VTK.r
    theta_vtk = current_VTK.theta
    phi_vtk = current_VTK.phi
    # x,y,z aussi !!!!!!!!!!!!!!!!!!!!!!!

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
