import matplotlib.pyplot as plt
import numpy as np
from astropy import units as au, constants as ac

class Background:
    def __init__(self) -> None:

        

        self.N_eff = 3
        self.Omega_b0 = 0.05
        self.Omega_r0 = 8 * np.pi**3 * ac.G * (ac.k_B * T_0)**4 * (1 + self.N_eff * (7/8) * (4/11)**(4/3)) / (45 * H_0**2 * ac.hbar**3 * ac.c**5)

    def get_Hubble(T):
        return ...

    def get_rho_b(T):
        ...