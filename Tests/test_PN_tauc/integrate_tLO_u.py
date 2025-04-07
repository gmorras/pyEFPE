import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import cumulative_trapezoid
from scipy.special import gamma
plt.rcParams.update({'font.size': 24})
plt.rcParams.update({'lines.linewidth': 2})

#compute the exact integral we want to approximate
u = np.geomspace(1, 100, 1000000)
integrand = ((1 - 1/u**2)**(5/19))*(1 - (121/425)/u**2)**(1181/2299) - 1
F = cumulative_trapezoid(u*integrand, np.log(u), initial=0)

#approximate the integral of ((1 - 1/u**2)**N1)*(1 - a2/u**2)**N2 - 1
def approx_int_large_u(u, N1, N2, a2, kmax=100, qmax=100):
	
	#compute c1 coefficients
	c1 = np.ones(kmax)
	c1[1] = -N1
	for k in range(2, kmax):
		c1[k] = -c1[k-1]*(N1-k+1)/k
	
	#compute c2 coefficients
	c2 = np.ones(qmax)
	c2[1] = -a2*N2 
	for q in range(2, qmax):
		c2[q] = -c2[q-1]*a2*(N2-q+1)/q

	#compute u**(2*n-1)/(2*n - 1)
	u_2 = u**-2
	u_2n_1 = 1/u 
	
	#start suming
	integral = np.zeros_like(u)
	for n in range(2, kmax+qmax):
		
		#compute cn
		if kmax>(n-1):
			cn = np.sum(c1[max(0,n-qmax):n]*c2[(min(n,qmax)-1)::-1]) 
		else:
			cn = np.sum(c1[max(0,n-qmax):kmax]*c2[(min(n,qmax)-1):n-kmax-1:-1])

		#update integral
		integral = integral - cn*u_2n_1/(2*n - 3)	
		
		#update u_2n_1
		u_2n_1 = u_2n_1*u_2
		
	return integral

#approximate the integral ((1 - 1/u**2)**(5/19))*(1 - (121/425)/u**2)**(1181/2299) - 1 at large u
def fast_tNLO_int_large_u(u, nmax=10):
	
	#we are going to approximate the integral as -f0 + sum_{n=1}^{nmax} cn*u**-(2*n-1)
	cns = np.array([0,0.40941176470588236, 0.02286320645905421, 0.008142925951557094, 0.004155878512401501, 0.0024847568765827364, 0.001633290511588348, 0.0011455395465032605, 0.000842577094447224, 0.0006427195446513564, 0.0005045715814217113, 0.00040544477831148924, 0.0003321116545345162, 0.0002764636962467965, 0.00023331895675436905, 0.0001992473575067662, 0.00017190919726595898, 0.00014966654751355843, 0.000131346469484909, 0.00011609207361015848, 0.00010326617525262309, 9.238740896587563e-05, 8.308691874613337e-05, 7.50784086790562e-05, 6.813705814151314e-05, 6.20844345948999e-05, 5.677753690123865e-05, 5.210072977646673e-05, 4.7959732146031274e-05])
	
	#if np.amin(u)<1: raise Exception('u has to be larger than 1, but u_min=%s'%(u_min))
	
	#evaluate polynomial
	x = u**-2
	return u*np.polyval(cns[nmax::-1],x)-0.4555165803216864

#compute list of cn coefficients of approximate the integral of ((1 - 1/u**2)**N1)*(1 - a2/u**2)**N2 - 1
def cn_approx_int_large_u(N1, N2, a2, nmax=50, tol=1e-15):
	
	#find kmax and qmax accordingly
	kmax = 2*nmax + 1
	qmax = 2*nmax + 1
	
	#compute c1 coefficients
	c1 = np.ones(kmax)
	c1[1] = -N1
	for k in range(2, kmax):
		c1[k] = -c1[k-1]*(N1-k+1)/k
	
	#compute c2 coefficients
	c2 = np.ones(qmax)
	c2[1] = -a2*N2 
	for q in range(2, qmax):
		c2[q] = -c2[q-1]*a2*(N2-q+1)/q

	#start suming
	integral = np.zeros_like(u)
	cn = list()
	for n in range(2, nmax):
		
		#compute cn
		if kmax>(n-1):
			cn.append(-np.sum(c1[max(0,n-qmax):n]*c2[(min(n,qmax)-1)::-1])/(2*n - 3))
		else:
			cn.append(-np.sum(c1[max(0,n-qmax):kmax]*c2[(min(n,qmax)-1):n-kmax-1:-1])/(2*n - 3))
	
	print("-------------- cns --------------")
	print(cn)
	
	#find also from which u onwards, tol is reached
	print("----------u | tol<%s ----------"%(tol))
	print((np.array(cn)/tol)**(1/(2*np.arange(len(cn)) + 1)))
	
	return cn


#approximate value of the integral of ((1 - 1/u**2)**N1)*(1 - a2/u**2)**N2 - 1 from u=1 to infty
def approx_val_infty(N1, N2, a2, qmax=30):
		
	#compute c2 coefficients
	c2 = np.ones(qmax)
	c2[1] = -a2*N2 
	for q in range(2, qmax):
		c2[q] = -c2[q-1]*a2*(N2-q+1)/q

	#compute the value of integrating ((1 - 1/u**2)**N1)*(1/u**(2n)), n>=1 from u=1 to infty
	c1_sum = np.zeros(qmax)
	n = np.arange(1, qmax)
	c1_sum[1:] = 0.5*gamma(n-0.5)*gamma(1+N1)/gamma(0.5 + n + N1)
	
	#for the n=0 case compute the integral of ((1 - 1/u**2)**N1) - 1 from u=1 to infty
	c1_sum[0] = 1-np.sqrt(np.pi)*gamma(1+N1)/gamma(0.5 + N1)
	
	return np.sum(c1_sum*c2)

approx_F = approx_int_large_u(u, 5/19, 1181/2299, 121/425)
approx_F_inf = approx_val_infty(5/19, 1181/2299, 121/425)	
approx_F_fast = fast_tNLO_int_large_u(u)

print(approx_F_inf)
cn_approx_int_large_u(5/19, 1181/2299, 121/425)


plt.figure(figsize=(16,10))
plt.plot(u, -integrand, label='integrand')
plt.plot(u, -(-0.4094117647058824/u**2 - 0.06858961937716263/u**4 - 0.04071462975778547/u**6 -0.02909114958681050/u**8), '--', label='O(1/u^8) approximation')
plt.plot(u, -(-1 + 1.010338615633403 *(u-1)**(5/19) + 0.01434344987568197*(u-1)**(24/19) + 0.5514454916730133*(u - 1)**(43/19) + 1.232939532553864*(u - 1)**(62/19)-2.208656163964881*(u - 1)**(81/19)), '-', label='O((u-1)^(81/19)) approximation')
#plt.xscale('log')
#plt.yscale('log')
plt.xlim(1,3)
plt.ylim(0,1)
#plt.xlim(u[0], u[-1])
plt.xlabel('$u$')
plt.grid(True)
plt.tight_layout()
plt.legend()


plt.figure(figsize=(16,10))
plt.plot(1/u, F, label='exact')
plt.plot(1/u, approx_F + approx_F_inf, label='approx')
plt.plot(1/u, approx_F_fast,'--', label='fast approx')
plt.axhline(y=approx_F_inf, ls='--')
#plt.xscale('log')
plt.xlim(1/u[0], 1/u[-1])
plt.xlabel('$1/u$')
plt.grid(True)
plt.tight_layout()
plt.legend()

plt.show()

