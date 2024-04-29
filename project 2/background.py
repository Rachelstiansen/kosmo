import matplotlib.pyplot as plt
import numpy as np
from astropy import units as au, constants as ac

class Background:
    def __init__(self) -> None:
        self.N_eff = 3
        self.T_0 = 2.725 # [K]
        self.h = 0.7

        self.Omega_b0 = 0.05
        self.H_0 = (100 * self.h * au.km / (au.second * au.Mpc))
        self.Omega_r0 = 8 * np.pi**3 * ac.G * (ac.k_B * self.T_0)**4 * (1 + self.N_eff * (7/8) * (4/11)**(4/3)) / (45 * self.H_0**2 * ac.hbar**3 * ac.c**5)

    def cgs(self):
        k_B = ac.k_B.cgs.value
        G = ac.G.cgs.value
        h_bar = ac.hbar.cgs.value
        c = ac.c.cgs.value
        m_n = ac.m_n.value
        m_p = ac.m_p.value

        self.H_0 = self.H_0.cgs.value
        self.Omega_r0 = self.Omega_r0.cgs.value

    def T_nu(T):
        return (4/11)**(1/3) * T

    def get_Hubble(T):
        return ...

    def get_rho_b(T):
        ...