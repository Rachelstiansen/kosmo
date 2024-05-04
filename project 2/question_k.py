import matplotlib.pyplot as plt
from BBN import BBN
import numpy as np
from scipy.interpolate import interp1d
from tqdm import tqdm

plt.rcParams.update({"font.size": 12}) # For all the plots to have text size 12

# Computing the relic abundances:
N_eff = np.logspace(0, np.log10(5)/np.log10(10), 10)

# Initializing arrays to keep values of the relative number densities
Y_D = np.zeros(len(N_eff)); Y_p = np.zeros(len(N_eff))
Y_He4 = np.zeros(len(N_eff)); Y_Li7 = np.zeros(len(N_eff)); Y_He3 = np.zeros(len(N_eff))

for i in tqdm(range(len(N_eff))): # tqdm makes progress bar
    bbn = BBN(NR_interacting_species=8, N_eff=N_eff[i])
    bbn.solve_BBN(T_init=100e9, T_end=0.01e9) # solve the system until end temperature 0.1e9 K
    Y_i = bbn.Y_i

    # Set lower bound for mass fractions
    Y_min = 1e-20
    Y_i = np.where(Y_i < Y_min, Y_min, Y_i)

    # Decays:
    Y_i[4, -1] += Y_i[3, -1] # T -> He^3
    Y_i[6, -1] += Y_i[7, -1] # Be^7 -> Li^7

    Y_D[i] = Y_i[2, -1]; Y_p[i] = Y_i[1, -1]
    Y_He4[i] = Y_i[5, -1]; Y_Li7[i] = Y_i[6, -1]; Y_He3[i] = Y_i[4, -1]


# Interpolate:
def interp(x_in, y_in):
    x_in = np.log(x_in)
    y_in = np.log(y_in)
    f = interp1d(x_in, y_in, kind="cubic")
    return f

# Interpolated functions
YD_Yp_func = interp(N_eff, Y_D/Y_p)
YLi7_Yp_func = interp(N_eff, Y_Li7/Y_p)
YHe3_Yp_func = interp(N_eff, Y_He3/Y_p)
Y_He4_func = interp(N_eff, Y_He4)

# More resolved x_array for N_eff:
x_new = np.linspace(N_eff[0], N_eff[-1], 1000)

# Observed values
Y_He4 = 0.254
Y_He4_upper = 0.254 + 0.003
Y_He4_lower = 0.254 - 0.003

YD_Yp = 2.57e-5
YD_Yp_upper = 2.57e-5 + 0.03e-5
YD_Yp_lower = 2.57e-5 - 0.03e-5

YLi7_Yp = 1.6e-10
YLi7_Yp_upper = 1.6e-10 + 0.3e-10
YLi7_Yp_lower = 1.6e-10 - 0.3e-10

def chi_2(model, data, error):
    return np.sum((model - data)**2 / error**2)

def Bayesian(model, data, error):
    return 1 / (np.sqrt(2 * np.prod(error**2))) * np.exp(- chi_2(model, data, error))

model = np.array([4 * np.exp(Y_He4_func(np.log(x_new))), np.exp(YD_Yp_func(np.log(x_new))), np.exp(YLi7_Yp_func(np.log(x_new)))])
data = np.array([Y_He4, YD_Yp, YLi7_Yp])
error = np.array([0.003, 0.03e-5, 0.3e-10])

# Initialize arrays for probabilities
Bayesian_prob = np.zeros(len(x_new))
chi2 = np.zeros(len(x_new))

for i in range(len(x_new)):
    Bayesian_prob[i] = (Bayesian(np.transpose(model[:, i]), data, error))
    chi2[i] = (chi_2(np.transpose(model[:, i]), data, error))


# Plotting relic abundances:
fig, ax = plt.subplots(4, 1, figsize=(7, 10), sharex=True)

# Plot He4:
ax[0].plot(x_new, 4 * np.exp(Y_He4_func(np.log(x_new))), label=r"He$^4$",  color="brown")
ax[0].axvline(x_new[np.argmin(chi2)], ls="dotted", color="black")
ax[0].fill_between(N_eff, Y_He4_upper, Y_He4_lower, color="brown", alpha=0.3)
ax[0].set_ylabel(r"$4 Y_{He^4}$")
ax[0].set_ylim(0.2, 0.3)
ax[0].legend()

# Plot D, He^3:
ax[1].plot(x_new, np.exp(YD_Yp_func(np.log(x_new))), label="D", color="green")
ax[1].axvline(x_new[np.argmin(chi2)], ls="dotted", color="black")
ax[1].fill_between(N_eff, YD_Yp_upper, YD_Yp_lower, color="green", alpha=0.3)

ax[1].plot(x_new, np.exp(YHe3_Yp_func(np.log(x_new))), label=r"He$^3$", color="purple")
ax[1].set_ylabel(r"$Y_i / Y_p$")
ax[1].set_ylim(1e-5, 4e-5)
ax[1].legend()

# Plot Li^7:
ax[2].plot(x_new, np.exp(YLi7_Yp_func(np.log(x_new))), label=r"Li$^7$", color="pink")
ax[2].fill_between(N_eff, YLi7_Yp_upper, YLi7_Yp_lower, color="pink", alpha=0.3)
ax[2].axvline(x_new[np.argmin(chi2)], ls="dotted", color="black")
ax[2].set_ylabel(r"$Y_i / Y_p$")
ax[2].set_ylim(1e-10, 5e-10)
ax[2].legend()

# Plot Bayesian likelihood
ax[3].plot(x_new, Bayesian_prob, color="black")
ax[3].axvline(x_new[np.argmin(chi2)], ls="dotted", color="black", label=r"$\chi^2$ =" + f"{np.min(chi2):.3f} \n" + r"$N_{eff} =$" + f"{x_new[np.argmin(chi2)]:.3f}")
ax[3].set_xlabel(r"$N_{eff}$")
ax[3].set_ylabel("Bayesian \n likelihood")
ax[3].legend()
ax[3].set_xlim(1, 5)

fig.tight_layout()
fig.savefig("k_relic_abundances.png")


