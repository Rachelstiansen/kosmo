import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as sc
from scipy.integrate import solve_ivp, cumulative_trapezoid
from scipy import interpolate

# Setting all figure text to size 14
plt.rcParams.update({'font.size': 14})

def gamma(potential):
    # Calculating the gamma parameter:
    alpha = 1

    if potential == "power-law":
        return (alpha + 1) / alpha
    
    if potential == "exp":
        return 1


def eqs_of_motion(V, x_1, x_2, x_3, lamda):
    """
    Calculating the equations of motion for a given potential V 
    and for a given value of x_1, x_2, x_3 and lambda
    """
    dx_1 = -3 * x_1 + (np.sqrt(6) * lamda * x_2**2) / 2 + 0.5 * x_1 * (3 + 3 * x_1**2 - 3 * x_2**2 + x_3**2)
    dx_2 = - (np.sqrt(6) * lamda * x_1 * x_2) / 2 + 0.5 * x_2 * (3 + 3 * x_1**2 - 3 * x_2**2 + x_3**2)
    dx_3 = -2 * x_3 + 0.5 * x_3 * (3 + 3 * x_1**2 - 3 * x_2**2 + x_3**2)
    
    dlamda = - np.sqrt(6) * lamda**2  * (gamma(V) - 1) * x_1

    return np.array([dx_1, dx_2, dx_3, dlamda])


def power_law(N, y):
    x_1, x_2, x_3, lamda = y
    V = "power-law"
    return eqs_of_motion(V, x_1, x_2, x_3, lamda)


def exponential(N, y):
    x_1, x_2, x_3, lamda = y
    V = "exp"
    return eqs_of_motion(V, x_1, x_2, x_3, lamda)


def integrator(V, z_max=None, update_init=None, init_recent=None, upper_limit=None):
    """
    Function that integrates the equations of motion for x1, x2, x3 and lambda for both the power-law 
    potential and the exponential potenital. If z_max != None then thw integrator only integrates from
    z=0 to z=2 and calculates new initial conditions. Note that we need the integrator function to first
    be called once with update_init=True to get the new initial conditions. Then again with init_recent as 
    the new initial conditions found. This is done the first time in Hubble_param(). (Upper_limit was an experiment)
    """
    if z_max != None:
        z_max = z_max

    else:
        z_max = 2e7
    
    z_min = 0
    N_min = -np.log(z_min + 1)
    N_max = -np.log(z_max + 1)

    n_points = 1000 # Number of points in making the range of N and z

    if V == "power-law":
        if update_init == False: # if we want to plot z between 0 and 2 and use the updated initial conditions, not find them
            x1_0 = init_recent[0]; x2_0 = init_recent[1]; x3_0 = init_recent[2]
            lamda_0 = init_recent[3]
        else:
            x1_0 = 5e-5; x2_0 = 1e-8; x3_0 = 0.9999
            lamda_0 = 1e9
        init_val = np.array([x1_0, x2_0, x3_0, lamda_0])

        sol = solve_ivp(power_law, [N_max, N_min], init_val, method="DOP853", atol=1e-8, rtol=1e-8, dense_output=True)

    if V == "exp":
        if update_init == False: # if we want to plot z between 0 and 2 and use the updated initial conditions, not find them
            x1_0 = init_recent[0]; x2_0 = init_recent[1]; x3_0 = init_recent[2]
            lamda_0 = init_recent[3]
        else:
            x1_0 = 0; x2_0 = 5e-13; x3_0 = 0.9999
            lamda_0 = 3/2
        init_val = np.array([x1_0, x2_0, x3_0, lamda_0])

        sol = solve_ivp(exponential, [N_max, N_min], init_val, method="DOP853", atol=1e-8, rtol=1e-8, dense_output=True)

    N = np.linspace(sol.t[0], sol.t[-1], n_points)
    z = np.exp(-N) - 1

    solutions = sol.sol(N)

    init_recent = np.zeros(4)

    # Finding what the inital conditions must be when we have z between 0 and 2
    if update_init == True: # and upper_limit != None: 
        idx = np.where(np.abs(z - 2) < 1e-1)[0][0] # replace 2 with upper_limit to make more general
        init_recent = sol.sol(N[idx])
    
    x1 = solutions[0, :]; x2 = solutions[1, :]; x3 = solutions[2, :]  

    omega_phi = x1**2 + x2**2
    omega_r = x3**2
    omega_m = 1 - omega_phi - omega_r
    
    omegas = np.array([omega_phi, omega_r, omega_m])

    EoS = (x1**2 - x2**2) / (x1**2 + x2**2)
    
    return omegas, EoS, N, z, init_recent

# Need to run integrator twice with updated init conditions the second time for it to be right.

def Hubble_param(V, z_max=None):
    """
    Function for calulcating the Hubble parameter given in eq. 7 in the project description. If z_max != None then
    we caluclate the Hubble parameter only for z in range 0 to 2.
    """
    if z_max != None:
        # For when we want z between 0 and 2 we need values in this range:
        init_recent = integrator(V, update_init=True)[4] # retrieving new initial conditions
        omegas, w_phi, N, z, init_recent = integrator(V, z_max, update_init=False, init_recent=init_recent) # resulting values using the new inital conditions
    else:
        omegas, w_phi, N, z, init_recent = integrator(V)
        
    omega_phi, omega_r, omega_m = omegas

    integral = cumulative_trapezoid(3 * (1 + np.flip(w_phi)), N, initial=0)

    h = np.sqrt(omega_m[-1] * ((1 + z)**3) + omega_r[-1] * (1 + z)**4 + omega_phi[-1] * np.exp(np.flip(integral))) # h = H / H0
    
    return h

def Hubble_lamdaCDM(omega_m0=None, z_max=None):
    """
    Function for calculating the hubble parameter for the lambda CDM model. We use omega_m0 != None when
    we call on this function in omega_m0_best_fit() to be able to find a value for h for all values of omega_m0 (for problem 14).
    """
    if z_max != None:
        # For when we want z between 0 and 2 we need values in this range:
        init_recent = integrator("exp", update_init=True)[4] # retrieving new initial conditions
        z = integrator("exp", z_max, update_init=False, init_recent=init_recent)[3] # resulting z values using the new inital conditions
    else:
        z = integrator("exp")[3]

    # Used for problem 14 in omega_m0_best_fit() to get h value for several values of omega_m0
    if omega_m0 != None:
        omega_m0 = omega_m0
    else:
        omega_m0 = 0.3
    
    h = np.sqrt(omega_m0 * (1 + z)**3 + (1 - omega_m0))  # h = H / H0

    return h

def age(V):
    """
    Function for calulating the dimensionless age H_0t_0 of the Universe for the two
    potential models and the lambda CDM model.
    """
    N = integrator(V)[2]

    h = Hubble_param(V)
    h_lCDM = Hubble_lamdaCDM()
    
    t0 = cumulative_trapezoid(1 / (h), N, initial=0)
    t0_lamdaCDM = cumulative_trapezoid(1 / (h_lCDM), N, initial=0)

    return t0, t0_lamdaCDM


def luminosity_dist(model, omega_m0=None, z_max=None):
    """
    Function for calcualating the luminosity distance
    """
    if model == "exp":
        h = Hubble_param(model, z_max)
    if model == "power-law":
        h = Hubble_param(model, z_max)
    if model == "lCDM":
        h = Hubble_lamdaCDM(omega_m0, z_max)
        model = "exp" # to get correct values of z when calling integrator()

    init_recent = integrator(model, update_init=True)[4] # retrieving new initial conditions for z between 0 and 2
    omegas, w_phi, N, z, init_recent = integrator(model, z_max, update_init=False, init_recent=init_recent) # resulting values using the new inital conditions

    integrand = np.exp(-N) / h

    integral_N = cumulative_trapezoid(np.flip(integrand), N, initial=0)

    H0_d_L_over_c = np.exp(-N) * np.flip(integral_N)

    return H0_d_L_over_c, z

def chi_square(model, omega_m0=None):
    z_max = 2 # Setting this value because we do not need the whole interval from 0 to 2e7.
    data = np.loadtxt("sndata.txt", skiprows=5)
    z_data = data[:, 0]

    d_L_data = data[:, 1]; error = data[:, 2]  # [Gpc]
        
    h = 0.7 
    H0 = 100 * h * 1e6 # [m s^-1 Gpc^-1]

    if omega_m0 != None:
        d_L_model, z = luminosity_dist(model, omega_m0, z_max) 
    else:
        d_L_model, z = luminosity_dist(model, z_max) 

    # Interpolating data from exp and power law models:
    f = interpolate.interp1d(z, d_L_model)
    model = f(z_data) * sc.c / H0  # converting to Gpc

    chi2 = np.sum((model - d_L_data)**2 / (error**2))
    
    return data, chi2

def omega_m0_best_fit(plot=None):
    # Finding the value of omega_m0 which gives the lowest chi^2 value
    possible_omega_m0 = np.linspace(0, 1, 100) # Could have had more points here, but takes longer to run
    chi2 = np.zeros(100)

    for i in range(len(possible_omega_m0)):
        chi2[i] = chi_square("lCDM", possible_omega_m0[i])[1]
    
    idx = np.argmin(chi2)
    best_m0 = possible_omega_m0[idx]
    
    print(f"The best chi^2 value for lambda CDM model = {np.min(chi2)}")
    print(f"The value of omega_m0 which gives the best fit {best_m0}\n")
    
    if plot == True:
        plt.figure(figsize=(10, 5))
        plt.plot(possible_omega_m0, chi2)
        plt.plot(best_m0, idx, "x", markersize=10, label="Best fit value")
        plt.title("Best $\chi^2$ fit of $\Omega_{m0}$")
        plt.xlabel("$\Omega_{m0}$")
        plt.ylabel("$\chi^2$")
        plt.legend()
        plt.grid()
        plt.savefig("best_omega_m0.png")
        plt.show()

"""
-------------- Plotting ---------------------
"""

def plotting_omegas():

    z = integrator("exp")[3]

    omegas_power = integrator("power-law")[0]
    omegas_exp = integrator("exp")[0]
    labels = np.array(["$\Omega_{\phi}$", "$\Omega_r$", "$\Omega_m$"])

    plt.figure(figsize=(10, 6))
    plt.suptitle("Density parameters $\Omega(z)$")

    plt.subplot(2, 1, 1)
    for i in range(len(omegas_power)):
        plt.plot(z, omegas_power[i], label=labels[i])
    plt.plot(z, omegas_power[0] + omegas_power[1] + omegas_power[2], label="$\sum \Omega_i$", linestyle="--")
        
    plt.title("$V(\phi) = M^{4 + \\alpha} \phi^{- \\alpha}$")
    plt.xlabel("z")
    plt.ylabel("$\Omega$")
    plt.xscale("log")
    plt.gca().invert_xaxis()
    plt.grid()
    plt.legend()

    plt.subplot(2, 1, 2)
    for i in range(len(omegas_exp)):
        plt.plot(z, omegas_exp[i], label=labels[i])
    plt.plot(z, omegas_exp[0] + omegas_exp[1] + omegas_exp[2], label="$\sum \Omega$", linestyle="--")
    
    plt.title("$V(\phi) = V_0 e^{-\kappa \zeta \phi}$")
    plt.xlabel("z")
    plt.ylabel("$\Omega$")
    plt.xscale("log")
    plt.gca().invert_xaxis()
    plt.grid()
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f"omegas.png")
    plt.show()

def plot_EOS():

    z = integrator("exp")[3]

    EoS_power = integrator("power-law")[1]
    EoS_exp = integrator("exp")[1]

    plt.figure(figsize=(10, 5))
    plt.title("Equation of state parameter")

    plt.plot(z, EoS_power, label="$\omega_{\phi}$, power-law potential")
    plt.plot(z, EoS_exp, label="$\omega_{\phi}$, exponential potential")

    plt.xlabel("z")
    plt.ylabel("$\omega_{\phi}$")
    plt.xscale("log")
    plt.gca().invert_xaxis()
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"EoS.png")
    plt.show()

def plot_Hubble():

    z = integrator("exp")[3]
    
    h_exp = Hubble_param("exp")
    h_power = Hubble_param("power-law")
    h_lCDM = Hubble_lamdaCDM()

    plt.figure(figsize=(10, 5))
    plt.title("Hubble parameters")

    plt.plot(z, h_power, label="$H/H_0$ power-potential")
    plt.plot(z, h_exp, label="$H/H_0$ exponential potential")
    plt.plot(z, h_lCDM, label="$H/H_0$ for $\Lambda CDM$ with $\Omega_{m0} = 0.3$")

    plt.xlabel("z")
    plt.ylabel("$H / H_0$")
    plt.xscale("log")
    plt.yscale("log")
    plt.gca().invert_xaxis()
    plt.tight_layout()
    plt.legend()
    plt.grid()
    plt.savefig(f"hubble.png")
    plt.show()

def plot_d_L():

    plt.figure(figsize=(10,5))

    d_L_power, z = luminosity_dist("power-law", z_max=2) # data[-1, 0]
    d_L_exp, z = luminosity_dist("exp", z_max=2)
    
    
    plt.plot(z, d_L_power, label="$d_L$ power-law")
    plt.plot(z, d_L_exp, label="$d_L$ exponential")
    plt.title("Luminosity distance")
    plt.xlabel("z")
    plt.ylabel("$H_0 d_L /c$")
    plt.gca().invert_xaxis()
    plt.grid()
    plt.tight_layout()
    plt.legend()
    plt.savefig(f"dL.png")
    plt.show()
    
"""
--------------- Function calls ------------------
"""

plotting_omegas()

plot_EOS()

plot_Hubble()
    
plot_d_L()

t0_power_law = age("power-law")[0]
t0_exp = age("exp")[0]
t0_LamdaCDM = age("exp")[1]

print(f"Age of the Universe t_0 = {t0_power_law[-1]} for power-law potential")
print(f"Age of the Universe t_0 = {t0_exp[-1]} for exponential potential ")
print(f"Age of the Universe t_0 = {t0_LamdaCDM[-1]} for lambda CDM \n")
    
chi2_exp = chi_square("exp")[1]
chi2_power = chi_square("power-law")[1]

print(f"Chi squared for exponential model = {chi2_exp:.4f}")
print(f"Chi squared for power-law model = {chi2_power:.4f} \n")

omega_m0_best_fit(plot=True)
