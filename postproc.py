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
n_vtk = int(t_max / conf["Output"]["vtk"]) + 1
r_min = conf["Grid"]["X1-grid"][1]
r_max = conf["Grid"]["X1-grid"][-1]
n_r = conf["Grid"]["X1-grid"][2]
epsilon_0 = conf["Setup"]["epsilon"]
alpha = conf["Setup"]["alpha"]
Tilt_init = conf["Setup"]["tilt"]
a = conf["Setup"]["spin"]

# Reading the analysis files
t, M_tot = READ_BOX_AVERAGE()
r, Sigma, Tilt, Precession, L,  V_r, V_theta, V_phi, rho_mean, rho_V_r, rho_V_theta, rho_V_phi = READ_RADIAL_AVERAGE(n_average, n_r)
# Tilt += Tilt_init

TiltMean = np.zeros(t.size)
PrecessionMean = np.zeros(t.size)
for i in range(t.size):
    TiltMean[i] = np.mean(Tilt[i,:])
    PrecessionMean[i] = np.mean(Precession[i,:])

# Calculating the normalizators
r_0 = 2*r_min   # This is an input !
Sigma_0 = Sigma[0,np.where(r >= r_0)[0][0]]     # This is a simple convention
t_orbit = 2*np.pi*(r_0**1.5) / np.sqrt(1 - 2.5*epsilon_0**2)       # This is an input
n_orbit = 10  # This is an input !
wh_t_final = np.where(t/t_orbit >= n_orbit)[0][0]

# ----------------------------------------------------------------------------------
# Plotting mean inclination and precession
# ----------------------------------------------------------------------------------
params = {
            "color1": "tab:blue",
            "xmin1": 0,
            "xmax1": n_orbit,
            "xlabel1": r"$t/t_\text{orbit}$ [-]",
            "ymin1": TiltMean.min()*(1 - 0.05*np.sign(TiltMean.min())),
            "ymax1": TiltMean.max()*(1 + 0.05*np.sign(TiltMean.max())),
            "ylabel1": r"Mean radial inclination [°]",
            "color2": "tab:red",
            "xmin2": 0,
            "xmax2": n_orbit,
            "xlabel2": r"$t/t_\text{orbit}$ [-]",
            "ymin2": PrecessionMean.min()*(1 - 0.05*np.sign(PrecessionMean.min())),
            "ymax2": PrecessionMean.max()*(1 + 0.05*np.sign(PrecessionMean.max())),
            "ylabel2": r"Mean radial precession [°]",
            "title": None,
            "savetype": "pdf",
            "savepath": f"./plots/angles.pdf",
            "yscale": None
}
PLOT(t/t_orbit, TiltMean, t/t_orbit, PrecessionMean, params)

# ----------------------------------------------------------------------------------
# Radial inclination and mass movies
# ----------------------------------------------------------------------------------
tilt_plots = []
precession_plots = []
mass_plots = []
for k in range(t.size):
    if ((k%int(conf["Output"]["vtk"]) == 0) or (k == wh_t_final)) and (k <= wh_t_final):
        params = {
                    "color1": "tab:blue",
                    "xmin1": r_min,
                    "xmax1": r_max,
                    "xlabel1": r"$r$ [Code Units]",
                    "ymin1": Tilt.min()*(1 - 0.05*np.sign(Tilt.min())),
                    "ymax1": Tilt.max()*(1 + 0.05*np.sign(Tilt.max())),
                    "ylabel1": r"Tilt [°]",
                    "color2": "tab:red",
                    "xmin2": 0,
                    "xmax2": n_orbit,
                    "xlabel2": r"$t/t_\text{orbit}$ [-]",
                    "ymin2": TiltMean.min()*(1 - 0.05*np.sign(TiltMean.min())),
                    "ymax2": TiltMean.max()*(1 + 0.05*np.sign(TiltMean.max())),
                    "ylabel2": r"Mean radial tilt [°]",
                    "title": r"$t/t_\text{orbit} =$ " + f"{t[k]/t_orbit:.2f}",
                    "savetype": "png",
                    "savepath": f"./output/plots/tilt_{k}.png",
                    "yscale": None
        }
        PLOT(r, Tilt[k,:], t[0:k]/t_orbit, TiltMean[0:k], params)
        tilt_plots.append(params["savepath"])

        params = {
                    "color1": "tab:orange",
                    "xmin1": r_min,
                    "xmax1": r_max,
                    "xlabel1": r"$r$ [Code Units]",
                    "ymin1": Precession.min()*(1 - 0.05*np.sign(Precession.min())),
                    "ymax1": Precession.max()*(1 + 0.05*np.sign(Precession.max())),
                    "ylabel1": r"Precession [°]",
                    "color2": "tab:brown",
                    "xmin2": 0,
                    "xmax2": n_orbit,
                    "xlabel2": r"$t/t_\text{orbit}$ [-]",
                    "ymin2": PrecessionMean.min()*(1 - 0.05*np.sign(PrecessionMean.min())),
                    "ymax2": PrecessionMean.max()*(1 + 0.05*np.sign(PrecessionMean.max())),
                    "ylabel2": r"Mean radial precession [°]",
                    "title": r"$t/t_\text{orbit} =$ " + f"{t[k]/t_orbit:.2f}",
                    "savetype": "png",
                    "savepath": f"./output/plots/precession_{k}.png",
                    "yscale": None
        }
        PLOT(r, Precession[k,:], t[0:k]/t_orbit, PrecessionMean[0:k], params)
        precession_plots.append(params["savepath"])

        params = {
                    "color1": "tab:green",
                    "xmin1": r_min,
                    "xmax1": r_max,
                    "xlabel1": r"$r$ [Code Units]",
                    "ymin1": Sigma.min()*(1 - 0.05*np.sign(Sigma.min()))/Sigma_0,
                    "ymax1": Sigma.max()*(1 + 0.05*np.sign(Sigma.max()))/Sigma_0,
                    "ylabel1": r"$\Sigma/\Sigma_0$ [-]",
                    "vline_color": "black",
                    "vline_label": r"Theoretical ISCO",
                    "color2": "tab:purple",
                    "xmin2": 0,
                    "xmax2": n_orbit,
                    "xlabel2": r"$t/t_\text{orbit}$ [-]",
                    "ymin2": M_tot.min()*(1 - 0.05*np.sign(M_tot.min()))/M_tot[0],
                    "ymax2": M_tot.max()*(1 + 0.05*np.sign(M_tot.max()))/M_tot[0],
                    "ylabel2": r"$M/M_0$ [-]",
                    "title": r"$t/t_\text{orbit} =$ " + f"{t[k]/t_orbit:.2f}",
                    "savetype": "png",
                    "savepath": f"./output/plots/mass_{k}.png",
                    "yscale": None
        }
        PLOT(r, Sigma[k,:]/Sigma_0, t[0:k]/t_orbit, M_tot[0:k]/M_tot[0], params, a=None)        
        mass_plots.append(params["savepath"])

MOVIE(tilt_plots, "tilt")
MOVIE(precession_plots, "precession")
MOVIE(mass_plots, "mass")

rotationCurve_plots = []
for k in range(n_vtk):
    time = int(conf["Output"]["vtk"] // conf["Output"]["analysis"] * k)
    r_vtk, theta_vtk, phi_vtk, rho, v_r, v_theta, v_phi = READ_VTK(k)
    
    omega = np.zeros(r_vtk.size)
    for i in range(r_vtk.size):
        midplane = np.argmax(rho[0,:,i])
        # ROTATIOOOOOOOOON
        Vr = v_r[0,midplane,i]
        Vth = v_theta[0,midplane,i]
        Vphi = v_phi[0,midplane,i]
        th = theta_vtk[midplane]

        er_ex = np.sin(th)*np.cos(0)
        er_ey = np.sin(th)*np.sin(0)
        er_ez = np.cos(th)
        eth_ex = np.cos(th)*np.cos(0)
        eth_ey = np.cos(th)*np.sin(0)
        eth_ez = -np.sin(th)
        ephi_ex = -np.sin(0)
        ephi_ey = np.cos(0)
        ephi_ez = 0.0
        Vx = Vr*er_ex + Vth*eth_ex + Vphi*ephi_ex
        Vy = Vr*er_ey + Vth*eth_ey + Vphi*ephi_ey
        Vz = Vr*er_ez + Vth*eth_ez + Vphi*ephi_ez

        tilt_loc = Tilt[time,i]
        VxUnt = np.cos(-tilt_loc)*Vx + np.sin(-tilt_loc)*Vz
        VyUnt = Vy
        VzUnt = -np.sin(-tilt_loc)*Vx + np.cos(-tilt_loc)*Vz

        ex_ephi = -np.sin(0)
        ey_ephi = np.cos(0)
        ez_ephi = 0.0
        VphiUnt = VxUnt*ex_ephi + VyUnt*ey_ephi + VzUnt*ez_ephi

        omega[i] = VphiUnt / r_vtk[i]
    kappa_2 = 4*omega**2 + 2*r_vtk*omega*np.gradient(omega, r_vtk)

    omega_V = V_phi[time,:] / r * 4*np.pi       # BECAUSE IN 2D YOU DON'T NEED THIS NORMALIZATION !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    kappa_2_V = 4*omega_V**2 + 2*r*omega_V*np.gradient(omega_V, r)
    omega_rho_V = rho_V_phi[time,:] / (rho_mean[time,:] * r)
    kappa_2_rho_V = 4*omega_rho_V**2 + 2*r*omega_rho_V*np.gradient(omega_rho_V, r)

    omega_th, kappa_2_th = THEORETICAL_ROTATION_CURVE(r_vtk, Tilt_init, gravity="Kepler", source_term=False)

    w, l, l_log = 1.25, 10, 5
    fig, axs = plt.subplots(1, 2, figsize=(8, 4))

    ax = axs[0]
    ax.plot(r_vtk, omega, color="tab:red", label=r"Idefix (VTK)")
    ax.plot(r, omega_V, color="tab:blue", label=r"Idefix ($\overline{V}$)")
    ax.plot(r, omega_rho_V, color="tab:green", label=r"Idefix ($\overline{\rho V}$)")
    ax.plot(r_vtk, omega_th, color="black", linestyle="dashed", label=r"Theory")
    ax.set_xlim((r_min-1, r_max))
    ax.set_xlabel(r"$r$")
    ax.set_yscale("log")
    ax.set_ylim((1e-3, 3))
    ax.set_ylabel(r"$\Omega$")
    ax.tick_params(axis='y', which='both', direction='in', right=True, width=w, length=l)
    ax.tick_params(axis='x', which='both', direction='in', top=True, width=w, length=l)
    ax.tick_params(axis='y', which='minor', length=l_log, width=w)
    for spine in ax.spines.values():
            spine.set_linewidth(w)
    ax.grid()
    ax.legend()
    
    ax = axs[1]
    ax.plot(r_vtk, kappa_2, color="tab:red", label=r"Idefix (VTK)")
    ax.plot(r, kappa_2_V, color="tab:blue", label=r"Idefix ($\overline{V}$)")
    ax.plot(r, kappa_2_rho_V, color="tab:green", label=r"Idefix ($\overline{\rho V}$)")
    ax.plot(r_vtk, kappa_2_th, color="black", linestyle="dashed", label=r"Theory")
    ax.set_xlim((r_min-1, r_max))
    ax.set_xlabel(r"$r$")
    ax.set_yscale("log")
    ax.set_ylim((1e-5, 1))
    ax.set_ylabel(r"$\kappa^2$")
    ax.tick_params(axis='y', which='both', direction='in', right=True, width=w, length=l)
    ax.tick_params(axis='x', which='both', direction='in', top=True, width=w, length=l)
    ax.tick_params(axis='y', which='minor', length=l_log, width=w)
    for spine in ax.spines.values():
            spine.set_linewidth(w)
    ax.grid()
    ax.legend()

    fig.suptitle(r"$t/t_\text{orbit} =$ " + f"{t[time]/t_orbit:.2f}", x=0.515, y=0.925)
    fig.tight_layout()
    plt.savefig(f"./output/plots/rotationCurve_{k}.png", bbox_inches='tight', dpi=300)
    plt.close()
    rotationCurve_plots.append(f"./output/plots/rotationCurve_{k}.png")

MOVIE(rotationCurve_plots, "rotationCurve")
