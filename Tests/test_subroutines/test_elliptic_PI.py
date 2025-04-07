import os
os.environ.update(
    OMP_NUM_THREADS = '1',
    OPENBLAS_NUM_THREADS = '1',
    NUMEXPR_NUM_THREADS = '1',
    MKL_NUM_THREADS = '1',
)

import numpy as np
import scipy.special
from mpmath import ellippi
import time

####################################################################################

#function to compute the elliptic integral of the third kind \Pi(n;\phi;m) using Carlsons symmetric forms
def my_ellipPI_Carlson_old(n, m, phi=None):
	
	#if phi is not given, return complete integral
	if phi is None:
		y = 1 - m
		return scipy.special.elliprf(0, y, 1) + (n/3)*scipy.special.elliprj(0, y, 1, 1-n)
	#otherwise, return the corresponding incomplete integral
	else:
		sphi = np.sin(phi)
		sphi2 = sphi*sphi
		cphi2 = 1 - sphi2
		y = 1 - m*sphi2
		return sphi*(scipy.special.elliprf(cphi2, y, 1) + (n/3)*sphi2*scipy.special.elliprj(cphi2, y, 1, 1-n*sphi2))

#function to compute the elliptic integral of the third kind \Pi(n;\phi;m)
#We assume n, m and phi (if given) to have the same shape
def my_ellipPI_old(n, m, phi=None):
	
	#consider the case in wich n is a number
	if np.asarray(n).ndim == 0:
		#compute only the values for which n<1 to avoid errors
		if n<1: return my_ellipPI_Carlson_old(n, m, phi=phi)
		else:   return 0
	else:
		#initialize result to have the same properties as n
		result = np.zeros_like(n)
		
		#compute only the values for which n<1 to avoid errors
		idxs = (n<1)
		if phi is None: result[idxs] = my_ellipPI_Carlson_old(n[idxs], m[idxs])
		else:           result[idxs] = my_ellipPI_Carlson_old(n[idxs], m[idxs], phi=phi[idxs])
		
		return result

####################################################################################

#function to compute the elliptic integral of the third kind \Pi(n;\phi;m) using Carlsons symmetric form RJ and the elliptic integral of the first kind K(m)
def my_ellipPI_Carlson(n, m, phi=None):
	
	#if phi is not given, return complete integral
	if phi is None:
		return scipy.special.ellipk(m) + (n/3)*scipy.special.elliprj(0, 1-m, 1, 1-n)
	#otherwise, return the corresponding incomplete integral
	else:
		sphi = np.sin(phi)
		sphi2 = sphi*sphi
		nsphi2 = n*sphi2
		return scipy.special.ellipkinc(phi, m) + (nsphi2/3)*sphi*scipy.special.elliprj(1 - sphi2, 1 - m*sphi2, 1, 1-nsphi2)

#function to compute the elliptic integral of the third kind \Pi(n;\phi;m)
#We assume n, m and phi (if given) to have the same shape
def my_ellipPI(n, m, phi=None):
	
	#consider the case in wich n is a number
	if np.asarray(n).ndim == 0:
		#compute only the values for which n<1 to avoid errors
		if n<1: return my_ellipPI_Carlson(n, m, phi=phi)
		else:   return 0
	else:
		#initialize result to have the same properties as n
		result = np.zeros_like(n)
		
		#compute only the values for which n<1 to avoid errors
		idxs = (n<1)
		if phi is None: result[idxs] = my_ellipPI_Carlson(n[idxs], m[idxs])
		else:           result[idxs] = my_ellipPI_Carlson(n[idxs], m[idxs], phi=phi[idxs])
		
		return result


####################################################################################

#ranges to check
n_min = 0
n_max = 1
m_min = 0
m_max = 1
phi_min = -0.5*np.pi
phi_max = 0.5*np.pi

#number to check
N = int(1e5)

#create random realizations
n, m, phi = np.transpose(np.random.uniform(low=[n_min, m_min, phi_min], high=[n_max, m_max, phi_max], size=(N,3)))

#compute my elliptic PI (complete and incomplete)
start_soltime = time.time()
myPI_comp = my_ellipPI(n, m)
myPI_inc = my_ellipPI(n, m, phi=phi)
print("\nTime to evaluate elliptic functions with scipy vectorized:     %s seconds/call "%((time.time() - start_soltime)/N))

#compute my elliptic PI (complete and incomplete)
start_soltime = time.time()
myPI_comp_old = my_ellipPI_old(n, m)
myPI_inc_old = my_ellipPI_old(n, m, phi=phi)
print("Time to evaluate old elliptic functions with scipy vectorized: %s seconds/call \n"%((time.time() - start_soltime)/N))

#compute difference
print('Difference in complete integral:', np.linalg.norm(myPI_comp_old - myPI_comp)/np.linalg.norm(0.5*(myPI_comp + myPI_comp_old)))
print('Difference in incomplete integral:', np.linalg.norm(myPI_inc_old - myPI_inc)/np.linalg.norm(0.5*(myPI_inc + myPI_inc_old)))

#loop over realizations
start_soltime = time.time()
PI_comp = np.zeros(N)
PI_inc = np.zeros(N)
for i in range(N):
	#compute elliptic functions with mp math
	PI_comp[i] = ellippi(n[i],m[i])
	PI_inc[i] = ellippi(n[i],phi[i],m[i])

print("\nTime to evaluate elliptic functions with mpmath: %s seconds/call \n"%((time.time() - start_soltime)/N))

#compute difference
print('Difference in complete integral:', np.linalg.norm(PI_comp - myPI_comp)/np.linalg.norm(0.5*(myPI_comp + PI_comp)))
print('Difference in incomplete integral:', np.linalg.norm(PI_inc - myPI_inc)/np.linalg.norm(0.5*(myPI_inc + PI_inc)))

#loop over realizations also for my_elliptic py
start_soltime = time.time()
for i in range(N):
	#compute elliptic functions with mp math
	my_ellipPI(n[i], m[i])
	#my_ellipPI(n[i], phi[i], m[i])
print("\nTime to evaluate complete integral with scipy and loop:     %s seconds/call"%((time.time() - start_soltime)/N))

#loop over realizations also for my_elliptic_old py
start_soltime = time.time()
for i in range(N):
	#compute elliptic functions with mp math
	my_ellipPI_old(n[i], m[i])
	#my_ellipPI_old(n[i], phi[i], m[i])
print("Time to evaluate old complete integral with scipy and loop: %s seconds/call \n"%((time.time() - start_soltime)/N))
	
