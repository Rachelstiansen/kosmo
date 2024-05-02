import matplotlib.pyplot as plt
from BBN import BBN
import numpy as np
from scipy.interpolate import interp1d
from tqdm import tqdm

plt.rcParams.update({"font.size": 12}) # For all the plots to have text size 12
# Computing the relic abundances:

Omega_b0 = np.logspace(-2, 0, 10)
Y_D = np.zeros(len(Omega_b0)); Y_p = np.zeros(len(Omega_b0))
Y_He4 = np.zeros(len(Omega_b0)); Y_Li7 = np.zeros(len(Omega_b0)); Y_He3 = np.zeros(len(Omega_b0))

for i in tqdm(range(len(Omega_b0))): # tqdm makes progress bar
    bbn = BBN(NR_interacting_species=8, Omega_b0=Omega_b0[i])
    bbn.solve_BBN(T_init=100e9, T_end=0.01e9) # solve the system until end temperature 0.1e9 K
    Y_i = bbn.Y_i

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
YD_Yp_func = interp(Omega_b0, Y_D/Y_p)
YLi7_Yp_func = interp(Omega_b0, Y_Li7/Y_p)
YHe3_Yp_func = interp(Omega_b0, Y_He3/Y_p)
Y_He4_func = interp(Omega_b0, Y_He4)

# More resolved x_array for Omega_b0:
x_new = np.geomspace(Omega_b0[0], Omega_b0[-1], 100)

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

model = np.array([np.exp(YD_Yp_func(np.log(x_new))), np.exp(YHe3_Yp_func(np.log(x_new))), np.exp(YLi7_Yp_func(np.log(x_new)))])
data = np.array([Y_He4, YD_Yp, YLi7_Yp])
error = np.array([0.003, 0.03e-5, 0.3e-10])

# Need to initialize arrays?
Bayesian_prob = np.zeros(len(x_new))
chi2 = np.zeros(len(x_new))

for i in range(len(x_new)):
    Bayesian_prob[i] = np.transpose(Bayesian((model[:, i]), data, error))
    chi2[i] = np.transpose(chi_2((model[:, i]), data, error))

print(np.shape(model))
print(np.shape(Bayesian_prob))

fig, ax = plt.subplots(3, 1, figsize=(11, 7), sharex=True)

# Plot He4:
ax[0].plot(x_new, np.exp(Y_He4_func(np.log(x_new))) * 4, label=r"He$^4$",  color="brown")
ax[0].axvline(x_new[np.argmin(chi2)], ls="dotted", color="black")
ax[0].fill_between(Omega_b0, Y_He4_upper, Y_He4_lower, color="brown", alpha=0.3)
ax[0].set_ylabel(r"$4 Y_{He^4}$")
ax[0].set_ylim(0.2, 0.3)
ax[0].legend()

# Plot D, He^3, Li^7, 
ax[1].loglog(x_new, np.exp(YD_Yp_func(np.log(x_new))), label="D", color="green")
ax[1].axvline(x_new[np.argmin(chi2)], ls="dotted", color="black")
ax[1].fill_between(Omega_b0, YD_Yp_upper, YD_Yp_lower, color="green", alpha=0.3)

ax[1].loglog(x_new, np.exp(YHe3_Yp_func(np.log(x_new))), label=r"He$^3$", color="purple")

ax[1].loglog(x_new, np.exp(YLi7_Yp_func(np.log(x_new))), label=r"Li$^7$", color="pink")
ax[1].fill_between(Omega_b0, YLi7_Yp_upper, YLi7_Yp_lower, color="pink", alpha=0.3)
ax[1].set_ylabel(r"$Y_i / Y_p$")
ax[1].set_ylim(1e-11, 1e-3)
ax[1].legend()

# Plot Bayesian likelihood
ax[2].plot(x_new, Bayesian_prob, color="black")
ax[2].axvline(x_new[np.argmin(chi2)], ls="dotted", color="black", label=r"$\chi^2$ =" + f"{np.min(chi2):.3f} \n" + r"$\Omega_{b0} =$" + f"{Omega_b0[np.argmin(chi2)]}")
ax[2].set_xlabel(r"$\Omega_{b0}$")
ax[2].set_ylabel("Bayesian \n likelihood")
ax[2].legend()
ax[2].set_xlim(1e-2, 1)

fig.savefig("j_relic_abundances.png")

