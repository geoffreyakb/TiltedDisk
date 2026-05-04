from src import *
# ----------------------------------------------------------------------------------
# Plotting parameters
# ----------------------------------------------------------------------------------
# plt.rcParams.update({
#     "text.usetex": True,
#     'text.latex.preamble':r'\usepackage{amsmath}',
#     "font.family": "Fourier"
# })
w, l, l_log = 1.25, 10, 6

# ----------------------------------------------------------------------------------
# Getting the input data
# ----------------------------------------------------------------------------------
conf = inifix.load("idefix.ini")
t_max = conf["TimeIntegrator"]["tstop"]
n_average = int(t_max / conf["Output"]["analysis"]) + 1         # t_max must be not divisible by the output rate in order for the +1 to work
r_min = conf["Grid"]["X1-grid"][1]
r_max = conf["Grid"]["X1-grid"][-1]
n_r = conf["Grid"]["X1-grid"][2]
epsilon_0 = conf["Setup"]["epsilon"]
alpha = conf["Setup"]["alpha"]

# Reading the analysis files
t, M_tot = READ_BOX_AVERAGE()
r, Sigma, Tilt, Precession, L = READ_RADIAL_AVERAGE(n_average, n_r)

TiltMean = np.zeros(t.size)
PrecessionMean = np.zeros(t.size)
for i in range(t.size):
    TiltMean[i] = np.mean(Tilt[i,:])
    PrecessionMean[i] = np.mean(Precession[i,:])

# Calculating the normalizators
r_0 = 2*r_min   # This is an input !
Sigma_0 = Sigma[0,np.where(r >= r_0)[0][0]]     # This is not an input, it would not affect the dynamics
t_orbit = 2*np.pi*r_0 / np.sqrt(1/r_0 - 2.5*epsilon_0**2)       # This is an input
n_orbit = 500   # This is an input !
wh_t_final = np.where(t/t_orbit >= n_orbit)[0][0]

# ----------------------------------------------------------------------------------
# Plotting mean inclination and precession
# ----------------------------------------------------------------------------------
params = {
            "color1": "tab:blue",
            "xmin1": 0,
            "xmax1": n_orbit,
            "xlabel1": r"$t/t_\text{orbit}$ [-]",
            "ymin1": TiltMean.min()*0.95,
            "ymax1": TiltMean.max()*1.05,
            "ylabel1": r"Mean radial inclination [°]",
            "color2": "tab:red",
            "xmin2": 0,
            "xmax2": n_orbit,
            "xlabel2": r"$t/t_\text{orbit}$ [-]",
            "ymin2": PrecessionMean.min()*0.95,
            "ymax2": PrecessionMean.max()*1.05,
            "ylabel2": r"Mean radial precession [°]",
            "title": None,
            "savetype": "pdf",
            "savepath": f"./plots/angles.pdf"
}
PLOT(t/t_orbit, TiltMean, t/t_orbit, PrecessionMean, params)

# ----------------------------------------------------------------------------------
# Radial inclination and mass movies
# ----------------------------------------------------------------------------------
inclination_plots = []
mass_plots = []
for k in range(t.size):
    if ((k%20 == 0) or (k == wh_t_final)) and (k <= wh_t_final):
        params = {
                    "color1": "tab:blue",
                    "xmin1": 1,
                    "xmax1": 10,
                    "xlabel1": r"$r$ [Code Units]",
                    "ymin1": Tilt.min()*0.95,
                    "ymax1": Tilt.max()*1.05,
                    "ylabel1": r"Inclination [°]",
                    "color2": "tab:red",
                    "xmin2": 0,
                    "xmax2": n_orbit,
                    "xlabel2": r"$t/t_\text{orbit}$ [-]",
                    "ymin2": PrecessionMean.min()*0.95,
                    "ymax2": PrecessionMean.max()*1.05,
                    "ylabel2": r"Mean radial precession [°]",
                    "title": r"$t/t_\text{orbit} =$ " + f"{t[k]/t_orbit:.2f}",
                    "savetype": "png",
                    "savepath": f"./output/plots/inclination_{k}.png"
        }
        PLOT(r, Tilt[k,:], t[0:k]/t_orbit, PrecessionMean[0:k], params)
        inclination_plots.append(params["savepath"])

        params = {
                    "color1": "tab:green",
                    "xmin1": 1,
                    "xmax1": 10,
                    "xlabel1": r"$r$ [Code Units]",
                    "ymin1": Sigma.min()*0.95/Sigma_0,
                    "ymax1": Sigma.max()*1.05/Sigma_0,
                    "ylabel1": r"$\Sigma/\Sigma_0$ [-]",
                    "color2": "tab:purple",
                    "xmin2": 0,
                    "xmax2": n_orbit,
                    "xlabel2": r"$t/t_\text{orbit}$ [-]",
                    "ymin2": M_tot.min()*0.95/M_tot[0],
                    "ymax2": M_tot.max()*1.05/M_tot[0],
                    "ylabel2": r"$M/M_0$ [-]",
                    "title": r"$t/t_\text{orbit} =$ " + f"{t[k]/t_orbit:.2f}",
                    "savetype": "png",
                    "savepath": f"./output/plots/mass_{k}.png"
        }
        PLOT(r, Sigma[k,:]/Sigma_0, t[0:k]/t_orbit, M_tot[0:k]/M_tot[0], params)        
        mass_plots.append(params["savepath"])

MOVIE(inclination_plots, "inclination")
MOVIE(mass_plots, "mass")
