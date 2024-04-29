import matplotlib.pyplot as plt
import numpy as np
from astropy import units as au, constants as ac
from scipy.integrate import quad

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

    def get_np_to_D(T, rho):
        ...

    def get_nD_to_T(T, rho):
        ...
    
    def get_DD_to_pT(T, rho):
        ...

    # Strong reactions:
