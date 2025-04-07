import numpy as np
import scipy.special
import time
start_runtime = time.time()

########################################################################################################################

#function to compute Newtonian amplitudes N22_p(e**2), as defined in 2402.06804
def N22_Newtonian_old(p, e2):

	#compute common factors of eccentricity that will be needed
	e = np.sqrt(e2)
	sq1me2 = np.sqrt(1-e2)
	p1me2_2 = 1 - 0.5*e2
	Jpm2_fact = 0.5*(sq1me2 + p1me2_2)
	Jpp2_fact = np.where(e2>1e-4, 0.5*(sq1me2 - p1me2_2), -0.0625*e2*e2*(1+0.5*e2))
	
	#compute k that goes into the argument of bessel function
	k = p+2
	ke = k*e

	#return the amplitude
	return k*(-sq1me2*scipy.special.jv(k,ke) + 0.5*e*(scipy.special.jv(k+1,ke) - scipy.special.jv(k-1,ke)) + Jpm2_fact*scipy.special.jv(k-2,ke)+Jpp2_fact*scipy.special.jv(k+2,ke))


#function to compute Newtonian amplitudes N22_p(e**2), as defined in 2402.06804
def N22_Newtonian_new(p, e2):

	#compute common factors of eccentricity that will be needed
	e = np.sqrt(e2)
	sq1me2 = np.sqrt(1-e2)
	Jpm2_fact = 0.5*(sq1me2 + 1 - 0.5*e2)
	Jpp2_fact = -0.0625*e2*e2/Jpm2_fact
	
	#compute k that goes into the argument of bessel function
	k = p+2
	ke = k*e

	#return the amplitude
	return k*(-sq1me2*scipy.special.jv(k,ke) + 0.5*e*(scipy.special.jv(k+1,ke) - scipy.special.jv(k-1,ke)) + Jpm2_fact*scipy.special.jv(k-2,ke) + Jpp2_fact*scipy.special.jv(k+2,ke))

########################################################################################################################

N_test = 1000000
e2_max = 0.64

#compute random array of ps and es
p = np.random.randint(-100, 100, N_test)
e2 = np.random.uniform(0, e2_max, N_test)

#test vectorized performance
start_soltime = time.time()
N2m_old = N22_Newtonian_old(p, e2)
print("\nTime to compute vectorized N22 with old code: %s seconds/call" % ((time.time() - start_soltime)/N_test))

start_soltime = time.time()
N2m_new = N22_Newtonian_new(p, e2)
print("Time to compute vectorized N22 with new code: %s seconds/call" % ((time.time() - start_soltime)/N_test))

N2m_rel_diff = 2*np.abs(N2m_old - N2m_new)/np.maximum(np.abs(N2m_old + N2m_new), 1e-17)
print("Relative error:  MSE:", np.sqrt(np.mean(np.square(N2m_rel_diff))), " max:", np.amax(N2m_rel_diff))

#test loop performance
start_soltime = time.time()
N2m_old = [N22_Newtonian_old(p_i, e2_i) for p_i, e2_i in zip(p, e2)]
print("\nTime to compute N22 in loop with old code: %s seconds/call" % ((time.time() - start_soltime)/N_test))

start_soltime = time.time()
N2m_new = [N22_Newtonian_new(p_i, e2_i) for p_i, e2_i in zip(p, e2)]
print("Time to compute N22 in loop with new code: %s seconds/call" % ((time.time() - start_soltime)/N_test))

N2m_old, N2m_new = np.array(N2m_old), np.array(N2m_new)
N2m_rel_diff = 2*np.abs(N2m_old - N2m_new)/np.maximum(np.abs(N2m_old + N2m_new), 1e-17)
print("Relative error:  MSE:", np.sqrt(np.mean(np.square(N2m_rel_diff))), " max:", np.amax(N2m_rel_diff))



