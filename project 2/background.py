import matplotlib.pyplot as plt
import numpy as np
from astropy import units as au, constants as ac

class Background:
    def __init__(self, Omega_b0 = 0.05, N_eff = 3) -> None:
        self.N_eff = N_eff  # Effective number of neutrino species
        self.T_0 = 2.725    # Todays temperature of Uni [K]
        self.h = 0.7

        self.Omega_b0 = Omega_b0 # Baryon density parameter (today)
        self.H_0 = (100 * self.h * au.km / (au.second * au.Mpc)).cgs.value
        
        # Density parameter for radiation (today):
        self.Omega_r0 = (8 * np.pi**3 * ac.G * (ac.k_B * self.T_0)**4 * (1 + self.N_eff * (7/8) * (4/11)**(4/3)) / (45 * self.H_0**2 * ac.hbar**3 * ac.c**5)).cgs.value

        self.rho_co = 3 * self.H_0**2 / (8 * np.pi * ac.G.cgs.value) # Present value of the critical density
        self.rho_b0 = self.Omega_b0 * self.rho_co # Present value of the baryon density

    def T_nu(self, T):
        return (4/11)**(1/3) * T # Neutrino temperature
    
    def a(self, T):
        return self.T_0 / T # Scale factor
    
    def get_Hubble(self, T):
        return self.H_0 * np.sqrt(self.Omega_r0) / self.a(T)**2 # Hubble constant

    def get_rho_b(self, T):
        return self.Omega_b0 * self.rho_co / self.a(T)**3 # Baryon mass density

class cgs:
    def __init__(self):    
        self.k_B = ac.k_B.cgs.value         # the Boltzmann constant
        self.G = ac.G.cgs.value             # Gravitational constant
        self.h_bar = ac.hbar.cgs.value      # Reduced Planck constant h/2pi
        self.c = ac.c.cgs.value             # Speed of light
        self.m_n = ac.m_n.cgs.value         # Mass of neutron
        self.m_p = ac.m_p.cgs.value         # Mass of proton
        
