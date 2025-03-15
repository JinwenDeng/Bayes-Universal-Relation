import numpy as np
from scipy.integrate import quad
from scipy.optimize import root_scalar
from astropy import constants as const
from scipy.interpolate import interp1d as sp_interp1d

# cosmological parameters
km = 10**5
c=const.c.cgs.value
H0 = 67.66*km  # cm/s/Mpc
Omega_m = 0.3103
Omega_lambda = 0.6897

def H(z):
    return H0 * np.sqrt(Omega_m * (1+z)**3 + Omega_lambda)

z_list = np.linspace(0, 0.5, 1000)
D_c_list = []  # comoving distance
D_l_list = []  # luminosity distance
for z in z_list:
    integrand = lambda z_prime: c / H(z_prime)
    D_c_list.append(quad(integrand, 0, z)[0])
    D_l_list.append((1+z) * D_c_list[-1])

z2dc = sp_interp1d(z_list, D_c_list)
dc2z = sp_interp1d(D_c_list, z_list)
z2dl = sp_interp1d(z_list, D_l_list)
dl2z = sp_interp1d(D_l_list, z_list)

def z_to_D_c(z):
    return z2dc(z)

def D_c_to_z(D_c):
    return dc2z(D_c)

def z_to_D_l(z):
    return z2dl(z)

def D_l_to_z(D_l):
    return dl2z(D_l)

