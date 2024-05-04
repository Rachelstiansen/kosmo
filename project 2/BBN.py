import matplotlib.pyplot as plt
import numpy as np
from astropy import units as au, constants as ac
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline

# Own imports
from reaction_rates import ReactionRates
from background import Background, cgs

class BBN:
    """
    General Big Bang nucleosynthesis class solving the Boltzmann equations for a collection
    of particle species in the early universe.
    """
    def __init__(self, NR_interacting_species: int = 2, **background_kwargs) -> None:
        """
        The class takes NR_interacting_species as input on initialization. This is a number
        between 2 and 8, determining the collection of particles to be considered :
        
        Keyword Arguments :
            NR_interacting_species {int} -- The number of particle species (default : {2}), max = 8
            
            background_kwargs {dict} -- Arguments passed to the background , e.g. Neff and Omega_b0
        """
        self.NR_species = NR_interacting_species # The number of particle species
        self.species_labels = ["n", "p", "D", "T", "He3 ", "He4 ", "Li7 ", "Be7"] # The particle names
        self.mass_number = [1, 1, 2, 3, 3, 4, 7, 7] # The particle atomic numbers
        self.NR_points = 1000 # the number of points for our solution arrays
        self.RR = ReactionRates() # Initiate the set of reaction rates equations
        
        # Initiate the background equations , e.g. equations for the Hubble parameter and rho_b :
        self.background = Background(**background_kwargs)
        self.cgs = cgs()

    def get_ODE(self, lnT: float, Y: np.ndarray) -> np.ndarray:
        """
        Computes the right hand side of the ODE for Boltzmanns equation
        for elements / particle species in Y

        Arguments :
            lnT {float} -- natural log of temperature
            
            Y {np.ndarray} -- array of relative abundance for each species.
                With all included species Y takes the form: n, p, D, T, He3, He4, Li7, Be7 = Y
        """
        T = np.exp(lnT)
        T9 = T / 1e9
        Hubble = self.background.get_Hubble(T)
        rho_b = self.background.get_rho_b(T)

        dY = np.zeros_like(Y) # differential for each species following the shape of Y
        # e.i. dY[0] corresponds to dY_n. Initialized to zero for all species .
        
        # Weak interactions, always included ~ (n <-> p) (a 1-3)
        Y_n, Y_p = Y[0], Y[1]
        lambda_n, lambda_p = self.RR.get_rate_weak(T9)

        # the change to particles on the left hand side:
        change_LHS = (Y_p * lambda_p - Y_n * lambda_n)
        dY[0] += change_LHS # Update the change to neutron fraction
        dY[1] -= change_LHS # Update the change to proton fraction (opposite sign)

        if self.NR_species > 2: # Include deuterium
            Y_D = Y[2]
            # (n + p <-> D + gamma) (b.1)
            Y_np = Y_n * Y_p
            rate_np_D, rate_D_np = self.RR.get_np_to_D(T9, rho_b)
            change_LHS = Y_D * rate_D_np - Y_np * rate_np_D # left hand side changes
            dY[0] += change_LHS # Update the change to neutron fraction
            dY[1] += change_LHS # Update the change to proton fraction
            dY[2] -= change_LHS # Update the change to deuterium fraction
        
        if self.NR_species > 3: # include tritium
            Y_T = Y[3]
            # (n + D <-> T + gamma) (b.3)
            Y_nD = Y_n * Y_D
            rate_nD_T, rate_T_nD = self.RR.get_nD_to_T(T9, rho_b)
            change_LHS = Y_T * rate_T_nD - Y_nD * rate_nD_T
            dY[0] += change_LHS
            dY[2] += change_LHS
            dY[3] -= change_LHS

            # (D + D <-> p + T) (b.8)
            Y_DD = Y_D * Y_D
            Y_pT = Y_p * Y_T
            rate_DD_pT, rate_pT_DD = self.RR.get_DD_to_pT(T9, rho_b)
            change_LHS = 2 * Y_pT * rate_pT_DD - Y_DD * rate_DD_pT
            change_RHS = 0.5 * Y_DD * rate_DD_pT - Y_pT * rate_pT_DD
            dY[2] += change_LHS
            dY[1] += change_RHS
            dY[3] += change_RHS
        
        if self.NR_species > 4: # Include He3
            Y_He3 = Y[4]
            # (p + D <->  He^3 + gamma) (b.2)
            Y_pD = Y_p * Y_D
            rate_pD_He3, rate_He3_pD = self.RR.get_pD_to_He3(T9, rho_b)
            change_LHS = Y_He3 * rate_He3_pD - Y_pD * rate_pD_He3
            dY[1] += change_LHS
            dY[2] += change_LHS
            dY[4] -= change_LHS

            # (n + He^3 <-> p + T) (b.4)
            Y_nHe3 = Y_n * Y_He3
            rate_nHe3_pT, rate_pT_nHe3 = self.RR.get_nHe3_to_pT(T9, rho_b)
            change_LHS = Y_pT * rate_pT_nHe3 - Y_nHe3 * rate_nHe3_pT
            dY[0] += change_LHS
            dY[4] += change_LHS
            dY[1] -= change_LHS
            dY[3] -= change_LHS

            # (D + D <-> n + He^3) (b.7)
            rate_DD_nHe3, rate_nHe3_DD = self.RR.get_DD_to_nHe3(T9, rho_b)
            change_LHS = 2 * Y_nHe3 * rate_nHe3_DD - Y_DD * rate_DD_nHe3
            change_RHS = 0.5 * Y_DD * rate_DD_nHe3 - Y_nHe3 * rate_nHe3_DD
            dY[2] += change_LHS
            dY[0] += change_RHS
            dY[4] += change_RHS            
        
        if self.NR_species > 5: # Include He4
            Y_He4 = Y[5]
            # (p + T <-> He^4 + gamma) (b.5)
            rate_pT_He4, rate_He4_pT = self.RR.get_pT_to_He4(T9, rho_b)
            change_LHS = Y_He4 * rate_He4_pT - Y_pT * rate_pT_He4
            dY[1] += change_LHS
            dY[3] += change_LHS
            dY[5] -= change_LHS

            # (n + He^3 <-> He^4 + gamma) (b.6)
            rate_nHe3_He4, rate_He4_nHe3 = self.RR.get_nHe3_to_He4(T9, rho_b)
            change_LHS = Y_He4 * rate_He4_nHe3 - Y_nHe3 * rate_nHe3_He4
            dY[0] += change_LHS
            dY[4] += change_LHS
            dY[5] -= change_LHS

            # (D + D <-> He^4 + gamma) (b.9)
            rate_DD_He4, rate_He4_DD = self.RR.get_DD_to_He4(T9, rho_b)
            change_LHS = 2 * Y_He4 * rate_He4_DD - Y_DD * rate_DD_He4
            change_RHS = 0.5 * Y_DD * rate_DD_He4 - Y_He4 * rate_He4_DD
            dY[2] += change_LHS
            dY[5] += change_RHS

            # (D + He^3 <-> He^4 + p) (b.10)
            Y_DHe3 = Y_D * Y_He3
            Y_He4p = Y_He4 * Y_p
            rate_DHe3_He4p, rate_He4p_DHe3 = self.RR.get_DHe3_to_He4p(T9, rho_b)
            change_LHS = Y_He4p * rate_He4p_DHe3 - Y_DHe3 * rate_DHe3_He4p
            dY[2] += change_LHS
            dY[4] += change_LHS
            dY[5] -= change_LHS
            dY[1] -= change_LHS

            # (D + T <-> He^4 + n) (b.11)
            Y_DT = Y_D * Y_T
            Y_He4n = Y_He4 * Y_n
            rate_DT_He4n, rate_He4n_DT = self.RR.get_DT_to_He4n(T9, rho_b)
            change_LHS = Y_He4n * rate_He4n_DT - Y_DT * rate_DT_He4n
            dY[2] += change_LHS
            dY[3] += change_LHS
            dY[5] -= change_LHS
            dY[0] -= change_LHS

            # (He^3 + T <-> He^4 + D) (b.15)
            Y_He3T = Y_He3 * Y_T
            Y_He4D = Y_He4 * Y_D
            rate_He3T_He4D, rate_He4D_He3T = self.RR.get_He3T_to_He4D(T9, rho_b)
            change_LHS = Y_He4D * rate_He4D_He3T - Y_He3T * rate_He3T_He4D
            dY[4] += change_LHS
            dY[3] += change_LHS
            dY[5] -= change_LHS
            dY[2] -= change_LHS 

        if self.NR_species > 6: # Include Li7
            Y_Li7 = Y[6]
            # (T + He^4 <-> Li^7 + gamma) (b.17)
            Y_THe4 = Y_T * Y_He4
            rate_THe4_Li7, rate_Li7_THe4 = self.RR.get_THe4_to_Li7(T9, rho_b)
            change_LHS = Y_Li7 * rate_Li7_THe4 - Y_THe4 * rate_THe4_Li7
            dY[3] += change_LHS
            dY[5] += change_LHS
            dY[6] -= change_LHS

            # (p + Li^7 <-> He^4 + He^4) (b.20)
            Y_pLi7 = Y_p * Y_Li7
            Y_2He4 = Y_He4 * Y_He4
            rate_pLi7_2He4, rate_2He4_pLi7 = self.RR.get_pLi7_to_2He4(T9, rho_b)
            change_LHS = 0.5 * Y_2He4 * rate_2He4_pLi7 - Y_pLi7 * rate_pLi7_2He4
            change_RHS = 2 * Y_pLi7 * rate_pLi7_2He4 - Y_2He4 * rate_2He4_pLi7
            dY[1] += change_LHS
            dY[6] += change_LHS
            dY[5] += change_RHS
        
        if self.NR_species > 7: # Include Be7
            Y_Be7 = Y[7]
            # (He^3 + He^4 <-> Be^7 + gamma) (b.16)
            Y_He3He4 = Y_He3 * Y_He4
            rate_He3He4_Be7, rate_Be7_He3He4 = self.RR.get_He3He4_to_Be7(T9, rho_b)
            change_LHS = Y_Be7 * rate_Be7_He3He4 - Y_He3He4 * rate_He3He4_Be7
            dY[4] += change_LHS
            dY[5] += change_LHS
            dY[7] -= change_LHS

            # (n + Be^7 <-> p + Li^7) (b.18)
            Y_nBe7 = Y_n * Y_Be7
            Y_pLi7 = Y_p * Y_Li7
            rate_nBe7_pLi7, rate_pLi7_nBe7 = self.RR.get_nBe7_to_pLi7(T9, rho_b)
            change_LHS = Y_pLi7 * rate_pLi7_nBe7 - Y_nBe7 * rate_nBe7_pLi7
            dY[0] += change_LHS
            dY[7] += change_LHS
            dY[1] -= change_LHS
            dY[6] -= change_LHS

            # (n + Be^7 <-> He^4 + He^4) (b.21)
            rate_nBe7_2He4, rate_2He4_nBe7 = self.RR.get_nBe7_to_2He4(T9, rho_b)
            change_LHS = 0.5 * Y_2He4 * rate_2He4_nBe7 - Y_nBe7 * rate_nBe7_2He4
            change_RHS = 2 * Y_nBe7 * rate_nBe7_2He4 - Y_2He4 * rate_2He4_nBe7
            dY[0] += change_LHS
            dY[7] += change_LHS
            dY[5] += change_RHS
        
        return - dY / Hubble

    def get_np_equil(self, T_i):
        """
        Calculates the initial values of Y_n and Y_p using equation (16) and (17) in the project description
        """
        Y_n = (1 + np.exp((self.cgs.m_n - self.cgs.m_p) * self.cgs.c**2 / (self.cgs.k_B * T_i)))**(-1)
        Y_p = 1 - Y_n

        return Y_n, Y_p

    def get_IC(self, T_init):
        """
        Defines the initial condition array used in solve_BBN. The only nonzero values are for 
        neutrons and protons, but the shape of self.Y_init is determined by the number of particles included.
        
        Arguments :
            T_init {float} -- the initial temperature
        """
        Y_init = np.zeros(self.NR_species) # Initializes all species to zero
        Y_n_init, Y_p_init = self.get_np_equil(T_init) # Solves eq 16-17
        Y_init[0] = Y_n_init
        Y_init[1] = Y_p_init
        
        return Y_init

    
    def solve_BBN(self, T_init: float = 100e9, T_end: float = 0.01e9):
        """
        Solves the BBN - system for a given range of temperature values
        
        Keyword Arguments :
            T_init {float} -- the initial temperature ( default : { 100e9 })
            T_end {float} -- the final temperature ( default : {0. 01e9 })
        """
        sol = solve_ivp(self.get_ODE, [np.log(T_init), np.log(T_end)], y0=self.get_IC(T_init), method="Radau", rtol=1e-12, atol=1e-12, dense_output=True)
        # Define linearly spaced logarithmic temperatures using points from the solver :
        lnT = np.linspace(sol.t[0], sol.t[-1], self.NR_points)
        self.Y_i = sol.sol(lnT) # use ln(T) to extract the solved solutions
        # Use ln(T) to create a corresponding logarithmically spaced T array
        self.T = np.exp(lnT) # array used for plotting , points corresponds to self .Y
    