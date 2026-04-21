import os
import sys
sys.path.append(os.getenv("IDEFIX_DIR"))
import csv
import numpy as np
import matplotlib.pyplot as plt
from pytools.vtk_io import readVTK
import inifix
import cv2

def READ_BOX_AVERAGE():
    fid = open("output/analysis/globalAverage.dat", "r")
    varnames = fid.readline().split()
    fid.close()
    data = np.loadtxt("output/analysis/globalAverage.dat",skiprows=1)
    V = {}
    i = 0
    for name in varnames:
        V[name] = data[:,i]           
        i += 1

    return V["t"], V["Mtot"]

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
        L[i,:,0] = V['Lx']
        L[i,:,1] = V['Ly']
        L[i,:,2] = V['Lz']

        norm = np.sqrt(L[i,:,0]**2 + L[i,:,1]**2 + L[i,:,2]**2)
        Tilt[i,:] = np.arccos(L[i,:,2] / norm) * 180/np.pi
        Precession[i,:] = np.arctan2(L[i,:,1], L[i,:,0]) * 180/np.pi
        
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

def PLOT_PROFILE(x1, y1, x2, y2, params):
    w, l = 1.25, 10
    fig, axs = plt.subplots(1, 2, figsize=(8, 4))

    ax = axs[0]
    ax.plot(x1, y1, color=params["color1"])
    ax.set_xlim((params["xmin1"], params["xmax1"]))
    ax.set_xlabel(params["xlabel1"])
    ax.set_ylim((params["ymin1"], params["ymax1"]))
    ax.set_ylabel(params["ylabel1"])
    ax.tick_params(axis='y', which='both', direction='in', right=True, width=w, length=l)
    ax.tick_params(axis='x', which='both', direction='in', top=True, width=w, length=l)
    for spine in ax.spines.values():
            spine.set_linewidth(w)
    ax.grid()

    ax = axs[1]
    ax.plot(x2, y2, color=params["color2"])
    ax.set_xlim((params["xmin2"], params["xmax2"]))
    ax.set_xlabel(params["xlabel2"])
    ax.set_ylim((params["ymin2"], params["ymax2"]))
    ax.set_ylabel(params["ylabel2"])
    ax.tick_params(axis='y', which='both', direction='in', right=True, width=w, length=l)
    ax.tick_params(axis='x', which='both', direction='in', top=True, width=w, length=l)
    for spine in ax.spines.values():
            spine.set_linewidth(w)
    ax.grid()

    fig.suptitle(params["title"], x=0.515, y=0.925)
    fig.tight_layout()
    plt.savefig(params["savepath"], bbox_inches='tight', dpi=250)
    plt.close()

def MOVIE(plots, name):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = 60
    images = plots
    height, width, _ = cv2.imread(images[0]).shape
    video_summary = cv2.VideoWriter(filename=f"./plots/{name}.mp4", fourcc=fourcc, fps=fps, frameSize=(width, height))
    for image in images:
        video_summary.write(cv2.imread(image))
    cv2.destroyAllWindows()
    video_summary.release()