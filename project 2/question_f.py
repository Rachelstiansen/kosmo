import matplotlib.pyplot as plt
from BBN import BBN


# Initiate the system including 2 species (neutrons and protons):
bbn = BBN(NR_interacting_species = 2)
bbn.solve_BBN(T_init=100e9, T_end=0.1e9) # solve the system until end temperature 0.1e9 K

# Plot the mass fraction for each species :
colors = ["orange", "blue"]
fig, ax = plt.subplots()
for i, y in enumerate(bbn.Y_i):
    ax.loglog(bbn.T, bbn.mass_number[i] * y, label=bbn.species_labels[i], color=colors[i])

ax.loglog(bbn.T, bbn.get_np_equil(bbn.T)[0] * bbn.mass_number[0], ls="dotted", color="orange")
ax.loglog(bbn.T, bbn.get_np_equil(bbn.T)[1] * bbn.mass_number[1], ls="dotted", color="blue")

ax.legend()
ax.invert_xaxis()
ax.set_ylim([1e-3, 1.5])
ax.set_xlim([1e11, 1e8])
ax.set_ylabel("$Y_i$")
ax.set_xlabel("T [K]")
fig.savefig("f_mass_frac.png")
plt.show()

"""
Question: overflow....
også spør om BBN 64-66
"""
