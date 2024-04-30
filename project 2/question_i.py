import matplotlib.pyplot as plt
from BBN import BBN
import numpy as np

bbn = BBN(NR_interacting_species = 8)
bbn.solve_BBN(T_init=100e9, T_end=0.01e9) # solve the system until end temperature 0.1e9 K

# Plot the mass fraction for each species :
colors = ["orange", "blue", "green", "red", "purple", "brown", "pink", "gray"]
fig, ax = plt.subplots()
for i, y in enumerate(bbn.Y_i):
    ax.loglog(bbn.T, bbn.mass_number[i] * y, label=bbn.species_labels[i], color=colors[i])

ax.legend()
ax.invert_xaxis()
ax.set_ylim([1e-3, 1.5])
ax.set_xlim([1e11, 1e8])
ax.set_ylabel("Mass fraction")
ax.set_xlabel("T [K]")
fig.savefig("i_mass_frac.png")
plt.show()
