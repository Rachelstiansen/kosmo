import numpy as np
from astropy import units as au, constants as ac

N_eff = 3
T_0 = 2.725 # [K]
h = 0.7

def t(T):
    H_0 = (100 * h * au.km / (au.second * au.Mpc))
    Omega_r0 = 8 * np.pi**3 * ac.G * (ac.k_B * T_0)**4 * (1 + (7/8) * N_eff * (4/11)**(4/3)) / (45 * ac.hbar**3 * ac.c**5 * H_0**2)
    t = (T_0 / T)**2 / (2 * H_0 * np.sqrt(Omega_r0)).cgs.value
    return t

print(f"t(T = 10^10 K) = {t(1e10):.2f} s")
print(f"t(T = 10^9 K) = {t(1e9):.2f} s")
print(f"t(T = 10^8 K) = {t(1e8):.2f} s")