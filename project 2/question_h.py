import matplotlib.pyplot as plt
from BBN import BBN

plt.rcParams.update({"font.size": 12}) # For all the plots to have text size 12

# Initiate the system including 3 species (neutrons, protons and deuterium):
bbn = BBN(NR_interacting_species = 3)
bbn.solve_BBN(T_init=100e9, T_end=0.1e9) # solve the system until end temperature 0.1e9 K

# Plot the mass fraction for each species :
colors = ["blue", "orange", "green"]
fig, ax = plt.subplots()

sum_mass_frac = 0
for i, y in enumerate(bbn.Y_i):
    ax.loglog(bbn.T, bbn.mass_number[i] * y, label=bbn.species_labels[i], color=colors[i])
    sum_mass_frac += bbn.mass_number[i] * y

ax.loglog(bbn.T, sum_mass_frac, ls="dotted", color="black", label=r"$\sum A_i Y_i$")
ax.loglog(bbn.T, bbn.get_np_equil(bbn.T)[0] * bbn.mass_number[0], ls="dotted", color="blue")
ax.loglog(bbn.T, bbn.get_np_equil(bbn.T)[1] * bbn.mass_number[1], ls="dotted", color="orange")

ax.legend()
ax.invert_xaxis()
ax.set_ylim([1e-3, 1.5])
ax.set_xlim([1e11, 1e8])
ax.set_ylabel("Mass fraction $[A_i Y_i]$")
ax.set_xlabel("T [K]")
ax.grid()
fig.tight_layout()
fig.savefig("h_mass_frac.png")
plt.show()


