import matplotlib.pyplot as plt
from BBN import BBN
import numpy as np

plt.rcParams.update({"font.size": 12}) # For all the plots to have text size 12

bbn = BBN(NR_interacting_species=8)
bbn.solve_BBN(T_init=100e9, T_end=0.01e9) # solve the system until end temperature 0.1e9 K

Omega_b0 = np.logspace(-2, 0, 10)