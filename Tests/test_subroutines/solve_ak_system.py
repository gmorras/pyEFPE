import numpy as np
from scipy.special import factorial, factorial2
import time
start_runtime = time.time()

#we want to solve the system of Eq.(127-128) of arXiv:2106.10291
#we move all factorials to the l.h.s. to have values closer to 1 in the Matrix of linear system
def ak_SUA_hard(kmax, alpha=1):

	#rewritte it as a linear system M*a = b
	M = np.zeros((kmax+1, kmax+1), dtype=np.complex128)
	b = np.ones(kmax+1)

	#compute the matrix M
	q = np.arange(kmax)+1
	q_factor = ((1j**q)/factorial2(2*q-1))[:,np.newaxis]
	M[1:,1:] = q_factor*((alpha*q)[np.newaxis,:]**(2*q[:,np.newaxis]))
	M[0,0] = 0.5
	M[0,1:] = 1

	#now solve the system, we multiply the final result by 0.5 to take into account the 1/2 of Eq.(39) of 1408.5158
	return 0.5*np.linalg.solve(M, b)

#we want to solve the system of Eq.(127-128) of arXiv:2106.10291
def ak_SUA_ez(kmax, alpha=1):

	#rewritte it as a linear system M*a = b
	M = np.zeros((kmax+1, kmax+1))
	ks = np.arange(kmax+1)
	b = (-0.5j)**ks/factorial(ks)

	#compute the matrix M
	k = ks[1:][np.newaxis,:]
	q = ks[1:][:,np.newaxis]
	M[1:,1:] = ((alpha*k)**(2*q))/factorial(2*q)
	M[0,0] = 0.5
	M[0,1:] = 1

	#now solve the system, we multiply the final result by 0.5 to take into account the 1/2 of Eq.(39) of 1408.5158
	return 0.5*np.linalg.solve(M, b)


#solve the system of Eq.(127-128) of arXiv:2106.10291 for the SUA constants a_{k, k_\max}
#we move all factorials to the l.h.s. to have values closer to 1 in the Matrix of linear system
#we allow for $\alpha \neq 0$ in the timeshifts $A(t + \alpha k T)$
def ak_SUA(kmax, alpha=1):

	#if kmax>9, there are problems with numerical precission, throw a warning
	if kmax>9: print('Warning: for SUA_kmax>9, finite precission errors when solving system of equations lead to untrustworthy values of a_{k,kmax}.')

	#rewritte it as a linear system M*a = b
	M = np.zeros((kmax+1, kmax+1), dtype=np.complex128)
	#b is 0.5 instead of 1 to take into account the 1/2 of Eq.(39) of 1408.5158
	b = np.full(kmax+1, 0.5)
	
	#compute k
	k=np.arange(kmax)+1
	ak = alpha*k
	
	#compute the matrix M_{q k} = ((i k^2)^q/(2q-1)!!)
	M[1:,1:] = np.cumprod(((1j*ak*ak)[None,:])/((2*k-1)[:,None]), axis=0)
	M[ 0, 0] = 0.5
	M[ 0,1:] = 1

	#now solve the system
	return np.linalg.solve(M, b)

#use a value of alpha
kmax_max = 12
alpha = 0.5
print('alpha =', alpha, '\n')
for kmax in range(kmax_max+1):
	#compute and time a with first method
	soltime1 = time.time()
	ak_kmax_1 = ak_SUA(kmax, alpha=alpha)
	soltime1 = time.time()- soltime1
	#compute and time a with second method
	soltime2 = time.time()
	ak_kmax_2 = ak_SUA_ez(kmax, alpha=alpha)
	soltime2 = time.time() - soltime2

	print('kmax =', kmax)
	print('Soltime 1: %.3gs   Soltime 2: %.3gs'%(soltime1, soltime2))
	print('|as| =',np.abs(ak_kmax_1))
	print('a0 + 2*sum_k a_k:, ', ak_kmax_1[0] + 2*np.sum(ak_kmax_1[1:]), ak_kmax_2[0] + 2*np.sum(ak_kmax_2[1:]))
	print('rel_err =',np.abs(ak_kmax_1 - ak_kmax_2)/np.abs(ak_kmax_1), '\n')

	#print(ak_kmax_1[:,np.newaxis],'\n')

#Runtime
print("\nRuntime: %s seconds" % (time.time() - start_runtime))

