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
        n + nu_e <-> p + e-
        n + e+ <-> p + anti nu_e
        n <-> p + e- + anti nu_e
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
        """
        p + n <-> D + gamma (b.1)
        """
        rate_np = 2.5e4 * rho_b
        rate_D = 4.68e9 * rate_np * T9**(3/2) * np.exp(-25.82/T9) / rho_b

        return rate_np, rate_D
    
    def get_pD_to_He3(self, T9, rho_b):
        """
        p + D <-> He^3 + gamma (b.2)
        """
        rate_pD = 2.23e3 * rho_b * T9**(-2/3) * np.exp(-3.72 * T9**(-1/3)) * (1 + 0.112 * T9**(1/3) + 3.38 * T9**(2/3) + 2.65 * T9)
        rate_He3 = 1.63e10 * rate_pD * T9**(3/2) * np.exp(-63.75 / T9) / rho_b

        return rate_pD, rate_He3

    def get_nD_to_T(self, T9, rho_b):
        """
        n + D <-> T + gamma (b.3)
        """
        rate_nD = rho_b * (75.5 + 1250 * T9)
        rate_T = 1.63e10 * rate_nD * T9**(3/2) * np.exp(-72.62/T9) / rho_b

        return rate_nD, rate_T
    
    def get_nHe3_to_pT(self, T9, rho_b):
        """
        n + He^3 <-> p + T (b.4)
        """
        rate_nHe3 = 7.06e8 * rho_b
        rate_pT = rate_nHe3 * np.exp(-8.864 / T9)
        
        return rate_nHe3, rate_pT
    
    def get_pT_to_He4(self, T9, rho_b):
        """
        p + T <-> He^4 + gamma (b.5)
        """
        rate_pT = 2.87e4 * rho_b * T9**(-2/3) * np.exp(-3.87 * T9**(-1/3)) * (1 + 0.108 * T9**(1/3) + 0.466 * T9**(2/3) + 0.352 * T9 + 0.3 * T9**(4/3) + 0.576 * T9**(5/3))
        rate_He4 = 2.59e10 * rate_pT * T9**(3/2) * np.exp(-229.9 / T9) / rho_b
        
        return rate_pT, rate_He4
    
    def get_nHe3_to_He4(self, T9, rho_b):
        """
        n + He^3 <-> He^4 + gamma (b.6)
        """
        rate_nHe3 = 6e3 * rho_b * T9
        rate_He4 = 2.6e10 * rate_nHe3 * T9**(3/2) * np.exp(-238.8 / T9) / rho_b
        
        return rate_nHe3, rate_He4
    
    def get_DD_to_nHe3(self, T9, rho_b):
        """
        D + D <-> n + He^3 (b.7)
        """
        rate_DD = 3.9e8 * rho_b * T9**(-2/3) * np.exp(-4.26 * T9**(-1/3)) * (1 + 0.0979 * T9**(1/3) + 0.642 * T9**(2/3) + 0.44 * T9)
        rate_nHe3 = 1.73 * rate_DD + 0.44 * T9
        
        return rate_DD, rate_nHe3
    
    def get_DD_to_pT(self, T9, rho_b):
        """
        D + D <-> p + T (b.8)
        """
        rate_DD = 3.9e8 * rho_b * T9**(-2/3) * np.exp(-4.26 * T9**(-1/3)) * (1 + 0.0979 * T9**(1/3) + 0.642 * T9**(2/3) + 0.44 * T9)
        rate_n_He3 = 1.73 * rate_DD * np.exp(-46.8 / T9)

        return rate_DD, rate_n_He3
    
    def get_DD_to_He4(self, T9, rho_b):
        """
        D + D <-> He^4 + gamma (b.9)
        """
        rate_DD = 24.1 * rho_b * T9**(-2/3) * np.exp(-4.26 * T9**(-1/3)) * (T9**(2/3) + 0.685 * T9 + 0.152 * T9**(4/3) + 0.265 * T9**(5/3))
        rate_He4 = 4.5e10 * rate_DD * T9**(3/2) * np.exp(-276.7 / T9) / rho_b
        
        return rate_DD, rate_He4
    
    def get_DHe3_to_He4p(self, T9, rho_b):
        """
        D + He^3 <-> He^4 + p (b.10)
        """
        rate_DHe3 = 2.6e9 * rho_b * T9**(-3/2) * np.exp(-2.99 / T9)
        rate_He4p = 5.5 * rate_DHe3 * np.exp(-213 / T9)
        
        return rate_DHe3, rate_He4p
    
    def get_DT_to_He4n(self, T9, rho_b):
        """
        D + T <-> He^4 + n (b.11)
        """
        rate_DT = 1.38e9 * rho_b * T9**(-3/2) * np.exp(-0.745 / T9)
        rate_He4n = 5.5 * rate_DT * np.exp(-204.1 / T9)
        
        return rate_DT, rate_He4n

    def get_He3T_to_He4D(self, T9, rho_b):
        """
        He^3 + T <-> He^4 + D (b.15)
        """ 
        rate_He3T = 3.88e9 * rho_b * T9**(-2/3) * np.exp(-7.72 * T9**(-1/3)) * (1 + 0.054 * T9**(1/3))
        rate_He4D = 1.59 * rate_He3T * np.exp(-166.2 / T9)
        
        return rate_He3T, rate_He4D
    
    def get_He3He4_to_Be7(self, T9, rho_b):
        """
        He^3 + He^4 <-> Be^7 + gamma (b.16)
        """
        rate_He3He4 = 4.8e6 * rho_b * T9**(-2/3) * np.exp(-12.8 * T9**(-1/3)) * (1 + 0.0326 * T9**(1/3) - 0.219 * T9**(2/3) - 0.0499 * T9 + 0.0258 * T9**(4/3) + 0.015 * T9**(5/3))
        rate_Be7 = 1.12e10 * rate_He3He4 * T9**(3/2) * np.exp(-18.42 / T9) / rho_b
        
        return rate_He3He4, rate_Be7
    
    def get_THe4_to_Li7(self, T9, rho_b):
        """
        T + He^4 <-> Li^7 + gamma (b.17)
        """
        rate_THe4 = 5.28e5 * rho_b * T9**(-2/3) * np.exp(-8.08 * T9**(-1/3)) * (1 + 0.0516 * T9**(1/3))
        rate_Li7 = 1.12e10 * rate_THe4 * T9**(3/2) * np.exp(-28.63 / T9) / rho_b
        
        return rate_THe4, rate_Li7
        
    def get_nBe7_to_pLi7(self, T9, rho_b):
        """
        n + Be^7 <-> p + Li^7 (b.18)
        """
        rate_nBe7 = 6.74e9 * rho_b
        rate_pLi7 = rate_nBe7 * np.exp(-19.07 / T9)
        
        return rate_nBe7, rate_pLi7
    
    def get_pLi7_to_2He4(self, T9, rho_b):
        """
        p + Li^7 <-> He^4 + He^4 (b.20)
        """
        rate_pLi7 = 1.42e9 * rho_b * T9**(-2/3) * np.exp(-8.47 * T9**(-1/3)) * (1 + 0.0493 * T9**(1/3))
        rate_2He4 = 4.64 * rate_pLi7 * np.exp(-201.3 / T9)
        
        return rate_pLi7, rate_2He4
    
    def get_nBe7_to_2He4(self, T9, rho_b):
        """
        n + Be^7 <-> He^4 + He^4 (b.21)
        """
        rate_nBe7 = 1.2e7 * rho_b * T9
        rate_2He4 = 4.64 * rate_nBe7 * np.exp(-220.4 / T9)
        
        return rate_nBe7, rate_2He4