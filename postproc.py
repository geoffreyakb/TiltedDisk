from src import *
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------------
# Plotting parameters
# ----------------------------------------------------------------------------------
plt.rcParams.update({
    "text.usetex": True,
    'text.latex.preamble':r'\usepackage{amsmath}',
    "font.family": "Fourier"
})
w = 1.25
l = 10
l_log = 6

# ----------------------------------------------------------------------------------
# Getting the input data
# ----------------------------------------------------------------------------------
conf = inifix.load("idefix.ini")
t_max = conf["TimeIntegrator"]["tstop"]
n_orbit = 500   # This is an input !
n_average = int(t_max / conf["Output"]["analysis"]) + 1
print(n_average)
r_min = conf["Grid"]["X1-grid"][1]
r_0 = 2*r_min   # This is a convention !
r_max = conf["Grid"]["X1-grid"][-1]
n_r = conf["Grid"]["X1-grid"][2]
alpha = conf["Setup"]["alpha"]

# ----------------------------------------------------------------------------------
# Reading the analysis files
# ----------------------------------------------------------------------------------
t, M_tot = READ_BOX_AVERAGE()
r, Sigma, Tilt, Precession, L = READ_RADIAL_AVERAGE(n_average, n_r)

TiltMean = np.zeros(t.size)
PrecessionMean = np.zeros(t.size)
for i in range(t.size):
    TiltMean[i] = np.mean(Tilt[i,:])
    PrecessionMean[i] = np.mean(Precession[i,:])

# ----------------------------------------------------------------------------------
# Calculating the normalizators
# ----------------------------------------------------------------------------------
epsilon_0 = conf["Setup"]["epsilon"]
Sigma_0 = Sigma[0,np.where(r >= r_0)[0][0]]
t_orbit = 2*np.pi*r_0 / np.sqrt(1/r_0 - 2.5*epsilon_0**2)
wh_t_final = np.where(t/t_orbit >= n_orbit)[0][0]

# ----------------------------------------------------------------------------------
# Comparing our profiles to the theoretical Kimmig and Dullemond (2024) ones
# ----------------------------------------------------------------------------------
Sigma_th = Sigma_0 * (r/r_0)**(-1)
epsilon_th = epsilon_0 * (r/r_0)**0.5

fig, axs = plt.subplots(1, 2, figsize=(8, 4))
ax = axs[0]
ax.plot(r, Sigma[0,:]/Sigma_0, color='tab:blue', label=r"Initial profile")
ax.plot(r, Sigma_th/Sigma_0, color='tab:red', label=r"Theoretical profile")
ax.set_xlim((r_min, r_max))
min_graph = 0.0
max_graph = 2.0
ax.set_ylim((min_graph, max_graph)) 
ax.tick_params(axis='y', which='both', direction='in', right=True, width=w, length=l)
ax.tick_params(axis='x', which='both', direction='in', top=True, width=w, length=l)
for spine in ax.spines.values():
        spine.set_linewidth(w)
ax.set_xlabel(r"$r$ [Code Units]")
ax.set_ylabel(r"$\Sigma/\Sigma_0$ [-]")
ax.grid()
ax.legend()

ax = axs[1]
ax.hlines(epsilon_0/epsilon_0, r_min, r_max, color='tab:blue', label=r"Initial profile")
ax.plot(r, epsilon_th/epsilon_0, color='tab:red', label=r"Theoretical profile")
ax.set_xlim((r_min, r_max))
min_graph = 0.6
max_graph = 2.4
ax.set_ylim((min_graph, max_graph))
ax.tick_params(axis='y', which='both', direction='in', right=True, width=w, length=l)
ax.tick_params(axis='x', which='both', direction='in', top=True, width=w, length=l)
for spine in ax.spines.values():
        spine.set_linewidth(w)
ax.set_xlabel(r"$r$ [Code Units]")
ax.set_ylabel(r"$\varepsilon/\varepsilon_0$ [-]")
ax.grid()
ax.legend()

fig.tight_layout()
plt.savefig("./plots/theoretical_profiles.pdf")
plt.close()

# ----------------------------------------------------------------------------------
# Plotting global mass
# ----------------------------------------------------------------------------------
fig, axs = plt.subplots(1, 1, figsize=(4, 4))
ax = axs
ax.plot(t/t_orbit, M_tot, color='tab:blue')
ax.set_xlim((0, n_orbit))
min_graph = 0.14
max_graph = 0.26
# ax.set_ylim((min_graph, max_graph))
ax.tick_params(axis='y', which='both', direction='in', right=True, width=w, length=l)
ax.tick_params(axis='x', which='both', direction='in', top=True, width=w, length=l)
for spine in ax.spines.values():
        spine.set_linewidth(w)
ax.set_xlabel(r"$t/t_\text{orbit}$ [-]")
ax.set_ylabel(r"$M/M_\odot$ [-]")
ax.grid()

fig.tight_layout()
plt.savefig("./plots/M_tot.pdf")
plt.close()

# ----------------------------------------------------------------------------------
# Plotting inclination and precession
# ----------------------------------------------------------------------------------
fig, axs = plt.subplots(1, 2, figsize=(8, 4))
ax = axs[0]
ax.plot(t/t_orbit, TiltMean, color='tab:blue')
ax.set_xlim((0, n_orbit))
min_graph = 14
max_graph = 15
# ax.set_ylim((min_graph, max_graph))
ax.tick_params(axis='y', which='both', direction='in', right=True, width=w, length=l)
ax.tick_params(axis='x', which='both', direction='in', top=True, width=w, length=l)
for spine in ax.spines.values():
        spine.set_linewidth(w)
ax.set_xlabel(r"$t/t_\text{orbit}$ [-]")
ax.set_ylabel(r"Mean radial inclination [°]")
ax.grid()

ax = axs[1]
ax.plot(t/t_orbit, PrecessionMean, color='tab:green')
ax.set_xlim((0, n_orbit))
min_graph = 0
max_graph = 2
# ax.set_ylim((min_graph, max_graph))
ax.tick_params(axis='y', which='both', direction='in', right=True, width=w, length=l)
ax.tick_params(axis='x', which='both', direction='in', top=True, width=w, length=l)
for spine in ax.spines.values():
        spine.set_linewidth(w)
ax.set_xlabel(r"$t/t_\text{orbit}$ [-]")
ax.set_ylabel(r"Mean radial precession [°]")
ax.grid()

fig.tight_layout()
plt.savefig("./plots/angles.pdf")
plt.close()

# ----------------------------------------------------------------------------------
# Plotting surface density evolution
# ----------------------------------------------------------------------------------
fig, axs = plt.subplots(1, 1, figsize=(4, 4))
ax = axs
ax.plot(r, Sigma[0,:]/Sigma_0, color='tab:blue', label=f"{t[0]/t_orbit:0.1f}" + r" orbits")
ax.plot(r, Sigma[t.size//4,:]/Sigma_0, color='tab:purple', label=f"{t[t.size//4]/t_orbit:0.1f}" + r" orbits")
ax.plot(r, Sigma[t.size//2,:]/Sigma_0, color='tab:red', label=f"{t[t.size//2]/t_orbit:0.1f}" + r" orbits")
ax.plot(r, Sigma[3*t.size//4,:]/Sigma_0, color='tab:orange', label=f"{t[3*t.size//4]/t_orbit:0.1f}" + r" orbits")
ax.plot(r, Sigma[t.size-1,:]/Sigma_0, color='tab:green', label=f"{t[wh_t_final]/t_orbit:0.1f}" + r" orbits")
ax.set_xlim((r_min, r_max))
min_graph = 0.25
max_graph = 1.75
ax.set_ylim((min_graph, max_graph))
ax.tick_params(axis='y', which='both', direction='in', right=True, width=w, length=l)
ax.tick_params(axis='x', which='both', direction='in', top=True, width=w, length=l)
for spine in ax.spines.values():
        spine.set_linewidth(w)
ax.set_xlabel(r"$r$ [Code Units]")
ax.set_ylabel(r"$\Sigma/\Sigma_0$ [-]")
ax.grid()
ax.legend()

fig.tight_layout()
plt.savefig("./plots/sigma.pdf")
plt.close()
