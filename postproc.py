from src import *
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Fourier"
})
w = 1.25
l = 10
l_log = 6

# jobname = os.path.basename(os.getcwd())
# conf = inifix.load(f'./plots/ini_file_{jobname}.ini')
conf = inifix.load("idefix.ini")
n_average = int(conf["TimeIntegrator"]["tstop"] / conf["Output"]["analysis"]) + 1
n_vtk = int(conf["TimeIntegrator"]["tstop"] / conf["Output"]["vtk"]) + 1
n_r = conf["Grid"]["X1-grid"][2]
r_min = conf["Grid"]["X1-grid"][1]
r_max = conf["Grid"]["X1-grid"][-1]

t = READ_BOX_AVERAGE()      # REMEMBER TO CHANGE THE FUNCTION
r, Sigma, Tilt, Precession, L = READ_RADIAL_AVERAGE(n_average, n_r)

TiltMean = np.zeros(t.size)
PrecessionMean = np.zeros(t.size)
for i in range(t.size):
    TiltMean[i] = np.mean(Tilt[i,:])
    PrecessionMean[i] = np.mean(Precession[i,:])

fig, axs = plt.subplots(1, 2, figsize=(13, 6))

ax = axs[0]
ax.plot(t, TiltMean, color='tab:blue')
ax.set_xlim((0, 100))
min_graph = 12
max_graph = 15
ax.set_ylim((min_graph, max_graph))
ax.tick_params(axis='y', which='both', direction='in', right=True, width=w, length=l)
ax.tick_params(axis='x', which='both', direction='in', top=True, width=w, length=l)
for spine in ax.spines.values():
        spine.set_linewidth(w)
ax.set_xlabel(r"$t$ [Code Units]")
ax.set_ylabel(r"Inclination [°]")
ax.grid()

ax = axs[1]
ax.plot(t, PrecessionMean, color='tab:green')
ax.set_xlim((0, 100))
min_graph = 0
max_graph = 10
ax.set_ylim((min_graph, max_graph))
ax.tick_params(axis='y', which='both', direction='in', right=True, width=w, length=l)
ax.tick_params(axis='x', which='both', direction='in', top=True, width=w, length=l)
for spine in ax.spines.values():
        spine.set_linewidth(w)
ax.set_xlabel(r"$t$ [Code Units]")
ax.set_ylabel(r"Precession [°]")
ax.grid()

fig.tight_layout()
plt.savefig("./plots/angles.pdf")
plt.close()


fig, axs = plt.subplots(1, 1, figsize=(6, 4))
ax = axs
ax.plot(r, Sigma[0,:], color='tab:blue', label=r"$t = $" + f" {t[0]:0.2e}")
ax.plot(r, Sigma[t.size//4,:], color='tab:red', label=r"$t = $" + f" {t[t.size//4]:0.2e}")
ax.plot(r, Sigma[t.size//2,:], color='tab:green', label=r"$t = $" + f" {t[t.size//2]:0.2e}")
ax.plot(r, Sigma[3*t.size//4,:], color='tab:orange', label=r"$t = $" + f" {t[3*t.size//4]:0.2e}")
ax.plot(r, Sigma[t.size-1,:], color='tab:purple', label=r"$t = $" + f" {t[t.size-1]:0.2e}")
ax.set_xlim((r_min, r_max))
min_graph = 12
max_graph = 15
# ax.set_ylim((min_graph, max_graph))
ax.tick_params(axis='y', which='both', direction='in', right=True, width=w, length=l)
ax.tick_params(axis='x', which='both', direction='in', top=True, width=w, length=l)
for spine in ax.spines.values():
        spine.set_linewidth(w)
ax.set_xlabel(r"$r$ [Code Units]")
ax.set_ylabel(r"Surface Density [Code Units]")
ax.grid()
ax.legend()

fig.tight_layout()
plt.savefig("./plots/sigma.pdf")
plt.close()
