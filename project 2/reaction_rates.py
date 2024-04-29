import matplotlib.pyplot as plt
import numpy as np
from astropy import units as au, constants as ac
from scipy.integrate import quad
from background import Background

class ReactionRates:
    def __init__(self):
        pass


    def get_rate_weak(self, T9):
        """
        T = T9
        """
        T_nu = (4/11)**(1/3) * T9
        tau = 1700 # [s] free neutron decay time
        q = 2.53 # (m_n - m_p) / m_e

        Z = 5.93 / T9
        Z_nu = 5.93 / T_nu

        def I_n_p(x):
            return (x + q)**2 * (x**2 - 1)**(1/2) * x / ((1 + np.exp(x * Z)) * (1 + np.exp(-(x + q) * Z_nu))) + (x - q)**2 * (x**2 - 1)**(1/2) * x / ((1 + np.exp(-x * Z)) * (1 + np.exp((x - q) * Z_nu)))

        def I_p_n(x):
            return (x - q)**2 * (x**2 - 1)**(1/2) * x / ((1 + np.exp(x * Z)) * (1 + np.exp(-(x - q) * Z_nu))) + (x + q)**2 * (x**2 - 1)**(1/2) * x / ((1 + np.exp(-x * Z)) * (1 + np.exp((x + q) * Z_nu)))

        lambda_n = quad(I_n_p, 1, np.inf)[0] / tau
        lambda_p = quad(I_p_n, 1, np.inf)[0] / tau
        
        return lambda_n, lambda_p
    
    # Strong reactions:

    def get_np_to_D(self, T9, rho_b):
        rate_np = 2.5e4 * rho_b
        rate_D = 4.68e9 * rate_np * T9**(3/2) * np.exp(-25.82/T9) / rho_b

        return rate_np, rate_D

    def get_nD_to_T(self, T9, rho_b):
        rate_nD = rho_b * (75.5 + 1250 * T9)
        rate_T = 1.63e10 * rate_nD * T9**(3/2) * np.exp(-72.62/T9) / rho_b

        return rate_nD, rate_T
    
    def get_DD_to_pT(self, T9, rho_b):
        ...

