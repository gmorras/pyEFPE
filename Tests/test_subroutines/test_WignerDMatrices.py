import numpy as np
import time

#function to compute Wigner (small) d matrices with l=2, from Eq.(A9) of 2106.10291
def single_wigner_d2(m1, m2, th):

	#Use that d_{-m1,-m2} = d_{m2, m1} = (-1)^{m1 + m2} d_{m1,m2} to fix m1>=|m2|>=0
	if abs(m1)<abs(m2):
		return ((-1)**(m1+m2))*single_wigner_d2(m2, m1, th)
	elif m1<0:
		return ((-1)**(m1+m2))*single_wigner_d2(-m1, -m2, th)
	#now hard-code the matrix elements with l=2>=m1>=|m2|>=0
	elif m1==2 and m2==2:  return  (0.5*(1+np.cos(th)))**2
	elif m1==2 and m2==1:  return -0.5*np.sin(th)*(1+np.cos(th))
	elif m1==2 and m2==0:  return  (0.375**0.5)*(np.sin(th)**2)
	elif m1==2 and m2==-1: return -0.5*np.sin(th)*(1-np.cos(th))
	elif m1==2 and m2==-2: return  (0.5*(1-np.cos(th)))**2
	elif m1==1 and m2==1:  return  np.cos(th)**2 + 0.5*(np.cos(th)-1)
	elif m1==1 and m2==0:  return -(0.375**0.5)*np.sin(2*th)
	elif m1==1 and m2==-1: return -np.cos(th)**2 + 0.5*(np.cos(th)+1)
	elif m1==0 and m2==0:  return 0.5*(3*np.cos(th)**2 -1)
	#if the case was not considered, spit an error
	else:
		raise Exception('m1=%s, m2=%s is not valid, m1 and m2 have to be integers between [-2,2]'%(m1,m2))

#function to create a dictionary of Wigner (small) d^{l=2}_{m1,m2} matrix elements with 2>=m1>=|m2|>=0, following conventions in 2106.10291
def compute_Wigner_small_d2(theta):
	
	#initialize dictionary
	d2 = {}
	
	#compute necessary trigonometric functions
	cth = np.cos(theta)
	cth2 = cth*cth
	sth = (1 - cth2)**0.5
	
	#fill dictionary with all Wigner d^l_{m1,m2}-matrix elements with l=2>=m1>=|m2|>=0
	#the rest can be obtained from d_{-m1,-m2} = d_{m2, m1} = (-1)^{m1 + m2} d_{m1,m2}
	d2[(2, 2)] =  (0.5*(1+cth))**2
	d2[(2, 1)] = -0.5*sth*(1+cth)
	d2[(2, 0)] =  (0.375**0.5)*sth*sth
	d2[(2,-1)] = -0.5*sth*(1-cth)
	d2[(2,-2)] =  (0.5*(1-cth))**2
	d2[(1, 1)] =  cth2 + 0.5*(cth-1)
	d2[(1, 0)] =  -(1.5**0.5)*sth*cth
	d2[(1,-1)] = -cth2 + 0.5*(cth+1)
	d2[(0, 0)] =  0.5*(3*cth2-1)
	
	#return this matrix
	return d2

#function to compute an array with all necessary Wigner D^{l=2}_{m1, m2} matrix elements, from Eq.(A9) of 2106.10291
def compute_Wigner_D2(phi, theta, zeta):
	
	#first compute the Wigner small d-matrix
	d2 = compute_Wigner_small_d2(theta)

	#initialize Wigner D-matrix
	if  np.asarray(theta).ndim!=0: D2 = np.zeros((5, 5, len(theta)), dtype=np.complex128)
	elif  np.asarray(phi).ndim!=0: D2 = np.zeros((5, 5,   len(phi)), dtype=np.complex128)
	elif np.asarray(zeta).ndim!=0: D2 = np.zeros((5, 5,  len(zeta)), dtype=np.complex128)
	else: D2 = np.zeros((5, 5), dtype=np.complex128)
	
	#compute exp(i*m*phi)
	if np.asarray(phi).ndim==0: exp_imph = np.cumprod(np.tile(np.exp(1j*phi), 2))
	else: exp_imph = np.cumprod(np.tile(np.exp(1j*phi), (2,1)), axis=0)

	#compute exp(i*m*zeta)
	if np.asarray(zeta).ndim==0: exp_imz = np.cumprod(np.tile(np.exp(1j*zeta), 2))
	else: exp_imz = np.cumprod(np.tile(np.exp(1j*zeta), (2,1)), axis=0)

	#loop over m1 and m2 to fill D^{l=2}_{m1, m2}
	for m1 in range(-2,3):
		#get the index corresponding to m1
		i_m1 = m1+2
		for m2 in range(-2,3):
			#get the index corresponding to m2
			i_m2 = m2 + 2
			#Use that d_{-m1,-m2} = d_{m2, m1} = (-1)^{m1 + m2} d_{m1,m2} to fix m1>=|m2|>=0
			if abs(m1)<abs(m2):
				if m2>0: D2[i_m1, i_m2] = ((-1)**(m1+m2))*d2[(m2,m1)]
				else:    D2[i_m1, i_m2] = d2[(-m2,-m1)]
			else:
				if m1<0: D2[i_m1, i_m2] = ((-1)**(m1+m2))*d2[(-m1,-m2)]
				else:    D2[i_m1, i_m2] = d2[(m1,m2)]

			#Add the exp(-i*m1*phi) phase term to go from Wigner small d to big D
			if   m1>0: D2[i_m1, i_m2] = np.conj(exp_imph[m1-1])*D2[i_m1, i_m2]
			elif m1<0: D2[i_m1, i_m2] = exp_imph[-m1-1]*D2[i_m1, i_m2]

			#Add the exp(-i*m2*zeta) phase term to go from Wigner small d to big D
			if   m2>0: D2[i_m1, i_m2] = np.conj(exp_imz[m2-1])*D2[i_m1, i_m2]
			elif m2<0: D2[i_m1, i_m2] = exp_imz[-m2-1]*D2[i_m1, i_m2]
			
	#return the Wigner D matrix
	return D2

#function to compute an array with all necessary Wigner D^{l=2}_{m1, m2} matrix elements, from Eq.(A9) of 2106.10291
def compute_Wigner_D2_slow(phi, theta, zeta):
	
	#first compute the Wigner small d-matrix
	d2 = compute_Wigner_small_d2(theta)

	#initialize Wigner D-matrix
	if  np.asarray(theta).ndim!=0: D2 = np.zeros((5, 5, len(theta)), dtype=np.complex128)
	elif  np.asarray(phi).ndim!=0: D2 = np.zeros((5, 5,   len(phi)), dtype=np.complex128)
	elif np.asarray(zeta).ndim!=0: D2 = np.zeros((5, 5,  len(zeta)), dtype=np.complex128)
	else: D2 = np.zeros((5, 5), dtype=np.complex128)
	
	#loop over m1 and m2 to fill D^{l=2}_{m1, m2}
	for m1 in range(-2,3):
		#get the index corresponding to m1
		i_m1 = m1 + 2
		for m2 in range(-2,3):
			#get the index corresponding to m2
			i_m2 = m2 + 2
			#Use that d_{-m1,-m2} = d_{m2, m1} = (-1)^{m1 + m2} d_{m1,m2} to fix m1>=|m2|>=0
			if abs(m1)<abs(m2):
				if m2>0: D2[i_m1, i_m2] = ((-1)**(m1+m2))*d2[(m2,m1)]
				else:    D2[i_m1, i_m2] = d2[(-m2,-m1)]
			else:
				if m1<0: D2[i_m1, i_m2] = ((-1)**(m1+m2))*d2[(-m1,-m2)]
				else:    D2[i_m1, i_m2] = d2[(m1,m2)]
			
			#Add the phi and zeta phase terms to go from Wigner small d to big D
			D2[i_m1, i_m2] = np.exp(-1j*(m1*phi + m2*zeta))*D2[i_m1, i_m2]
	
	#return the Wigner D matrix
	return D2


#function to compute the spin weighted spherical harmonics with spin -2 with l=2, i.e. _{-2}Y_{2,m} in Eq.(A10) of 2106.10291
def compute_m2_Y2_slow(theta, phi):

	#first compute the Wigner small d-matrix
	d2 = compute_Wigner_small_d2(theta)
	
	#initialize the spherical harmonics
	if np.asarray(theta).ndim!=0: m2_Y2 = np.zeros((5, len(theta)), dtype=np.complex128)
	elif np.asarray(phi).ndim!=0: m2_Y2 = np.zeros((5,   len(phi)), dtype=np.complex128)
	else: m2_Y2 = np.zeros(5, dtype=np.complex128)

	#compute _{-2}Y_{2,m}(\phi,\theta) = (-1)^m \sqrt{(2 l +1)/(4 \pi)} D^{2}_{-m,-2}(\phi,\theta) = (-1)^m \sqrt{(2 l +1)/(4 \pi)} e^{i m \phi} d^{2}_{2,m}(\theta)
	pref = (1.25/np.pi)**0.5
	for m in range(-2,3):
		i_m = m+2
		m2_Y2[i_m] = ((-1)**i_m)*pref*np.exp(1j*m*phi)*d2[(2,m)]
	
	#return spin -2 spherical harmonics
	return m2_Y2
	
############################################################################################################

#Function to compute arrays with necessary elements of Wigner (small) d matrix, following conventions in 2106.10291
#For our purposes we only need d^{l=2}_{m',2} and (sometimes) d^{l=2}_{m',0}. Here cth=cos(theta)
def compute_necessary_Wigner_small_d2(cth, return_d2_mp0=False, dtype=np.complex128):
	
	#compute necessary trigonometric functions
	cth2 = cth*cth
	sth = (1 - cth2)**0.5

	#compute an array with d^{l=2}_{m',2}
	d2_mp2= np.array([(0.5*(1-cth))**2    ,  #d^{2}_{-2,2}
	                   0.5*sth*(1-cth)    ,  #d^{2}_{-1,2}
	                  (0.375**0.5)*sth*sth,  #d^{2}_{ 0,2}
	                   0.5*sth*(1+cth)    ,  #d^{2}_{ 1,2}
	                  (0.5*(1+cth))**2    ], #d^{2}_{ 2,2}
	                  dtype=dtype)
	
	#if required, compute also d^{l=2}_{m',0}
	if return_d2_mp0:
		
		#compute d^{2}_{1,0}
		d2_10 = -(1.5**0.5)*sth*cth
		
		#now compute d^{l=2}_{0,m}
		d2_mp0 = np.array([d2_mp2[2]     ,  #d^{2}_{-2,0}= d^{2}_{0,2}
		                   -d2_10        ,  #d^{2}_{-1,0}=-d^{2}_{1,0}
		                   0.5*(3*cth2-1),  #d^{2}_{ 0,0}
		                   d2_10         ,  #d^{2}_{ 1,0}
		                   d2_mp2[2]     ], #d^{2}_{ 2,0} = d^{2}_{0,2}
		                   dtype=dtype)
		
		return d2_mp2, d2_mp0
	else:
		return d2_mp2


#Function to compute arrays with necessary elements of Wigner D matrix, following conventions in 2106.10291
#For our purposes we only need D^{l=2}_{m',2} and (sometimes) D^{l=2}_{m',0}
def compute_necessary_Wigner_D2(phi, cos_theta, zeta, return_D2_mp0=True):
	
	#compute necessary elements of Wigner (small) d matrix
	if return_D2_mp0: D2_mp2, D2_mp0 = compute_necessary_Wigner_small_d2(cos_theta, return_d2_mp0=return_D2_mp0, dtype=np.complex128)
	else: D2_mp2 = compute_necessary_Wigner_small_d2(cos_theta, return_d2_mp0=return_D2_mp0, dtype=np.complex128)

	#compute exp(-i*phi)
	exp_miph = np.cos(phi) - 1j*np.sin(phi)
	#compute exp(-i*phi*[1,2])
	exp_mimpphi = np.array([exp_miph, exp_miph*exp_miph])
	
	#add the exp(-i*mp*phi) phase to D^{l=2}_{m',2}
	D2_mp2[:2] = D2_mp2[:2]*np.conj(exp_mimpphi[::-1])
	D2_mp2[3:] = D2_mp2[3:]*exp_mimpphi
	
	#compute the exp(-2*i*zeta) phase
	exp_m2iz = np.cos(2*zeta) - 1j*np.sin(2*zeta)

	#add it to the wigner D matrix taking into account all the possibilities for zeta and th
	if np.asarray(zeta).ndim==0: D2_mp2 = exp_m2iz*D2_mp2
	elif D2_mp2.ndim==2:         D2_mp2 = exp_m2iz[np.newaxis,:]*D2_mp2
	else:                        D2_mp2 = exp_m2iz[np.newaxis,:]*D2_mp2[:, np.newaxis]
	
	#if required, compute also D^{l=2}_{m',0}, with m>=0
	if return_D2_mp0:
		
		#add the exp(-i*mp*phi) phase to D^{l=2}_{m',0}, m'>0
		D2_mp0[3:] = D2_mp0[3:]*exp_mimpphi
		
		#use D^{2}_{-2,0} = (D^{2}_{2,0})*
		D2_mp0[0] =  np.conj(D2_mp0[4])
		#use D^{2}_{-1,0}=-(D^{2}_{1,0})*
		D2_mp0[1] = -np.conj(D2_mp0[3])
		
		return D2_mp2, D2_mp0
	else:
		return D2_mp2

#function to compute the spin weighted spherical harmonics with spin -2 with l=2, i.e. _{-2}Y_{2,m} in Eq.(A10) of 2106.10291
def compute_m2_Y2(cos_theta, phi):

	#Compute (-1)^m \sqrt{(2 l +1)/(4 \pi)} D^{2}_{-m,-2}(\phi,\theta,0) = \sqrt{(2 l +1)/(4 \pi)} conj(D^{2}_{m,2}(\phi,\theta, 0))
	return ((1.25/np.pi)**0.5)*np.conj(compute_necessary_Wigner_D2(phi, cos_theta, 0, return_D2_mp0=False))
	
############################################################################################################


#number of points to plot
Npoints = 1000000

#test the function to compute wigner small d-matrix
cth = np.random.uniform(low=-1, high=1, size=Npoints)
th = np.arccos(cth)

#compute dictionary with relevant wigner directly
start_soltime = time.time()
d2 = compute_Wigner_small_d2(th)
print("\nTime to evaluate Wigner (small) d matrix: %s seconds \n" % (time.time() - start_soltime))

#compute Big Wigner D matrix fast
ph, z = np.random.uniform(low=[0,0], high=[2*np.pi,2*np.pi], size=(Npoints,2)).T
start_soltime = time.time()
D2 = compute_Wigner_D2(ph, th, z)
print("\nTime to evaluate Wigner D matrices fast: %s seconds \n" % (time.time() - start_soltime))

#compute Big Wigner D matrix slow
start_soltime = time.time()
D2_slow = compute_Wigner_D2_slow(ph, th, z)
print("\nTime to evaluate Wigner D matrices slow: %s seconds \n" % (time.time() - start_soltime))

print('Average difference:', np.linalg.norm(D2_slow - D2)/np.sqrt(np.prod(D2.shape)))

#compute the spherical harmonics fast
start_soltime = time.time()
m2_Y2 = compute_m2_Y2(cth, ph)
print("\nTime to evaluate spin -2 spherical harmonics fast: %s seconds \n" % (time.time() - start_soltime))

#compute the spherical harmonics slow
start_soltime = time.time()
m2_Y2_slow = compute_m2_Y2_slow(th, ph)
print("\nTime to evaluate spin -2 spherical harmonics slow: %s seconds" % (time.time() - start_soltime))
print('Average difference:', np.linalg.norm(m2_Y2 - m2_Y2_slow)/np.sqrt(np.prod(m2_Y2.shape)))

#compute necessary elements of Wigner D matrix
start_soltime = time.time()
D2_mp2, D2_mp0 = compute_necessary_Wigner_D2(ph, cth, z, return_D2_mp0=True)
print("\nTime to evaluate necessary Wigner D2: %s seconds \n" % (time.time() - start_soltime))

#compute difference with the other implementations
print('Average differences: 2m:', np.linalg.norm(D2[:,-1] - D2_mp2)/np.sqrt(np.prod(D2_mp2.shape)),' 0m:', np.linalg.norm(D2[:,2] - D2_mp0)/np.sqrt(np.prod(D2_mp0.shape)), '\n')

#test computing the values of D^{2}_{mp,-2} using that D^{2}_{mp,-2} = (-1)^mp (D^{mp,2})^{*}
start_soltime = time.time()
D2_mmp2 = np.conj(D2_mp2[::-1])
D2_mmp2[1::2] = -D2_mmp2[1::2]
print("\nTime to evaluate Wigner D2_{mp,-2} from D2_{mp,2}: %s seconds \n" % (time.time() - start_soltime))

print('Average difference:', np.linalg.norm(D2[:,0] - D2_mmp2)/np.sqrt(np.prod(D2_mmp2.shape)), '\n')

#test orthogonality of Wigner D matrix
norms_D2 = 5*np.mean(np.abs(D2)**2, axis=-1)
print(norms_D2)
#print(np.abs(np.einsum('ijk,lmk->ijlm', np.conj(D2), D2, optimize='greedy')*5/Npoints))

#test completness of Wigner D matrix
delta_m1m2 = np.einsum('ijk,ilk->jlk', np.conj(D2), D2, optimize='greedy')
print('\n', np.abs(np.mean(delta_m1m2, axis=-1)))
#print(np.std(delta_m1m2, axis=-1))

delta_m2m1 = np.einsum('jik,lik->jlk', np.conj(D2), D2, optimize='greedy')
print('\n', np.abs(np.mean(delta_m2m1, axis=-1)))
#print(np.std(delta_m2m1, axis=-1))

#test SU(2) group characters with Wigner D matrix
print()
for i in range(3):
	#compute rotation along single axis	
	if   i==0:  D2_1ax = compute_Wigner_D2(ph, 0, 0)
	elif i==1:  D2_1ax = compute_Wigner_D2(0, ph, 0)
	elif i==2:  D2_1ax = compute_Wigner_D2(0, 0, ph)
	
	chi2 = np.einsum('iij->j', D2_1ax, optimize='greedy')
	chi2_pred = np.sin(2.5*ph)/np.sin(0.5*ph)

	print(np.mean(chi2/chi2_pred), '\pm', np.std(chi2/chi2_pred))

#test orthogonality of spherical harmonics
print(np.abs(np.einsum('ik,jk->ij', np.conj(m2_Y2), m2_Y2, optimize='greedy'))*4*np.pi/Npoints)

