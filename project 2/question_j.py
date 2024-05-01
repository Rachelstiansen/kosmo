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

YD_Yp_func = interp(Omega_b0, Y_D/Y_p)

x_new = np.geomspace(Omega_b0[0], Omega_b0[-1], 100)

plt.loglog(x_new, np.exp(YD_Yp_func(np.log(x_new))))
plt.show()



YD_Yp_upper = 2.57e-5 + 0.03e-5
YD_Yp_lower = 2.57e-5 - 0.03e-5

Y_He4_upper = 0.254 + 0.003
Y_He4_lower = 0.254 - 0.003

YLi7_Yp_upper = 1.6e-10 + 0.3e-10
YLi7_Yp_lower = 1.6e-10 - 0.3e-10

def chi_2(model, data, error):
    return np.sum((model - data)**2 / (error**2))
