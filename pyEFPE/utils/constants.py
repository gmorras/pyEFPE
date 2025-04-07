import numpy as np

# Pre-cached numerical constants 
sqrt_2       = np.sqrt(2)
sqrt_3       = np.sqrt(3)
sqrt_4       = np.sqrt(4)
sqrt_5       = np.sqrt(5)
sqrt_6       = np.sqrt(6)

# Spherical harmonic prefactors
Y2_prefactor = ((1.25/np.pi)**0.5)

# Physical constants
t_sun_s      = 4.92549094831e-6
c_SI         = 299792458.0
G_SI         = 6.67430e-11
M_sun_SI     = 1.988409870698051e30
AU_SI        = 149597870700.0
pc_SI        = 3.0856775814671916e16
yr_SI        = 31557600.0
kpc_SI       = 1.0e3 * pc_SI
Mpc_SI       = 1.0e6 * pc_SI
Gpc_SI       = 1.0e9 * pc_SI

# Geometric constants
piMfISCO     = 6.0**(-3.0/2.0) * np.pi