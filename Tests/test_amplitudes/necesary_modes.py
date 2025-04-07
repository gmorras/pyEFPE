import numpy as np
from scipy.special import jv
from matplotlib import pyplot as plt

plot_norms = False

import os
if not os.path.exists('Plots/'): os.makedirs('Plots/')

import time
start_runtime = time.time()

##########################################################################

#function to compute Newtonian amplitudes N20_j, as defined in 2402.06804
def N20_Newtonian(p, e2):
	return np.where(p==0, 0, ((2/3)**0.5)*jv(p,p*np.sqrt(e2)))

#function to compute Newtonian amplitudes N22_j, as defined in 2402.06804
def N22_Newtonian(p, e2):

	#compute common factors of eccentricity that will be needed
	e         = np.sqrt(e2)
	sq1me2    = np.sqrt(1-e2)
	p1me2_2   = 1 - 0.5*e2
	Jpm2_fact = 0.5*(sq1me2 + p1me2_2)
	Jpp2_fact = np.where(e2>1e-4, 0.5*(sq1me2 - p1me2_2), -0.0625*e2*e2*(1+0.5*e2))
	
	#compute k that goes into the argument of bessel function
	k  = p+2
	ke = k*e

	#return the amplitude
	return k*(-sq1me2*jv(k,ke) + 0.5*e*(jv(k+1,ke) - jv(k-1,ke)) + Jpm2_fact*jv(k-2,ke)+Jpp2_fact*jv(k+2,ke))


#function to compute number of coefficients of Fourier series of Newtonian amplitudes N^{2m}_p needed for a given eccentricity
def Newtonian_orders_needed(e2, pmax, tol):

	#for 20 mode, compute 1 <= p <=pmax, since we will use N20_{-p} = N20_p
	p20  = np.arange(1,pmax+1)

	#for 22 mode compute -pmax <= p <=pmax
	p22  = np.arange(-pmax,pmax+1)
	
	#array with both order indexes and their correspondig labels
	ps   = np.append(np.column_stack(( np.zeros_like(p20), p20)),
	               np.column_stack((2*np.ones_like(p22), p22)), axis=0)
	
	#compute fourier coeficients and put them together in array
	Njs  = np.append(N20_Newtonian(p20, e2), N22_Newtonian(p22, e2))
 
	#compute also the norms i.e. \sum_p |N_{p}|^2
	norm = (4/3)*(-1 + 4/np.sqrt(1-e2))

	#compute the normalized amplitudes
	normed_Njs = (Njs**2)/norm
	
	#sort the elements from larger to smaller
	idxs_sort  = np.argsort(-normed_Njs)
	normed_Njs_sorted = normed_Njs[idxs_sort]

	#compute cumulative norm	
	cum_norm   = np.cumsum(normed_Njs_sorted, axis=0)
	
	#find how many orders have to be taken into account
	N_orders_needed = 1 + np.searchsorted(cum_norm, 1-tol, side='right')
	
	#return the indexes
	return ps[idxs_sort[:N_orders_needed]]
	
##########################################################################

#eccentricities to explore
e        = np.linspace(0, 0.9, 2000)

#orders to explore
pmax     = 400

#tolerances to use
tols     = np.array([1e-2,1e-3,1e-4])

tol_name = r'$\epsilon_N$'

##########################################################################

#for 20 mode, compute 1 <= p <=pmax, since we will use N20_{-p} = N20_p
p20 = np.arange(1,pmax+1)

#for 22 mode compute -pmax <= p <=pmax
p22 = np.arange(-pmax,pmax+1)

#compute mesh
e2  = e*e

#compute amplitudes (for 20 mode, add contribution from N20_{-p})
Njs   = {'20': (2**0.5)*N20_Newtonian(*np.meshgrid(p20, e2, indexing='ij')),
       '22':          N22_Newtonian(*np.meshgrid(p22, e2, indexing='ij'))}

#compute theoretical norms \sum_{p=-\inf}^\inf |N2m_p|^2
norms = {'20':(2/3)*(1/np.sqrt(1-e2) - 1), '22':5/np.sqrt(1-e2) - 1}

#save the orders being taken into account
ps    = {'20': p20, '22': p22}

#compute also the joint case
Njs['joint']   = np.append(Njs['20'], (2**0.5)*Njs['22'], axis=0)

#compute the joint norm
norms['joint'] = norms['20'] + 2*norms['22']

#compute joint ps with their corresponding labels
ps['joint']    = np.append(np.column_stack(( np.zeros_like(p20), p20)),
                        np.column_stack((2*np.ones_like(p22), p22)), axis=0)

#loop over modes
num_norms = {}
idxs_sort = {}
N_orders_needed = {}
for lm in ['20', '22', 'joint']:

	#compute sum over order of squared amplitudes
	num_norms[lm] = np.sum(np.abs(Njs[lm])**2, axis=0)

	#compute the normalized amplitudes
	normed_Njs = (np.abs(Njs[lm])**2)/norms[lm][np.newaxis,:]

	#sort the elements from larger to smaller along order axis
	idxs_sort[lm] = np.argsort(-normed_Njs, axis=0)
	
	#compute sorted ps and sorted modes
	p_sorted = ps[lm][idxs_sort[lm]]
	normed_Njs_sorted = np.take_along_axis(normed_Njs, idxs_sort[lm], axis=0)

	#compute cumulative sum along order
	cum_norm = np.cumsum(normed_Njs_sorted, axis=0)
	
	#find how many orders have to be taken into account
	tols = np.asarray(tols)
	N_orders_needed[lm] = 1 + np.apply_along_axis(np.searchsorted, 0, cum_norm, 1-tols)

#time cost of get needed orders
start_soltime = time.time()
for e2c in e2: 
	Newtonian_orders_needed(e2c, 200, 1e-4)
print("\nNewtonian_orders_need runtime: %s seconds/ecc" % np.around(((time.time() - start_soltime)/len(e2)),6))

#compare with the array method
diff = np.zeros(len(e2), dtype=int)
for ie2, e2c in enumerate(e2):
	diff[ie2] = N_orders_needed['joint'][0,ie2] - len(Newtonian_orders_needed(e2c, pmax, tols[0]))

print('Difference in methods:', np.linalg.norm(diff))

if plot_norms:
    #make a plot of the norms
    plt.figure(figsize=(12,8))
    plt.plot(e, num_norms['20'], label=r'$\sum_{p=1}^{%s} |N^{20}_p|^2$'%(pmax))
    plt.plot(e, norms['20'], '--', label=r'$||K^{20}||^2$')
    plt.plot(e, num_norms['22'], label=r'$\sum_{p=-%s}^{%s} |N^{22}_p|^2$'%(pmax, pmax))
    plt.plot(e, norms['22'], '--', label=r'$||K^{22}||^2$')
    plt.xlabel('$e$')
    plt.xlim(e[0], e[-1])
    plt.ylim(bottom=0)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('Plots/Newtonian_norms.pdf')
    
    #make a plot of how many orders we need as a function of eccentricity
    ls = {'20': '--', '22': '-'}
    plt.figure(figsize=(12,8))
    for itol, tol in enumerate(tols):
    	for lm in ['20', '22']:
    		plt.plot(e, N_orders_needed[lm][itol], ls[lm]+'C%.f'%(itol), label=lm+' mode, %s=%.3g'%(tol_name, tol))

    # Plot Newtonian orders needed
    plt.xlabel('$e$')
    plt.xlim(e[0], e[-1])
    plt.ylim(bottom=1)
    plt.yscale('log')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

plt.figure(figsize=(12,8),dpi=100)
for itol, tol in enumerate(tols):
	plt.plot(e, N_orders_needed['joint'][itol], 'C%.f'%(itol), label=r'%s=%.3g'%(tol_name, tol), linewidth=2.75)

plt.xlabel(r'$e$')
plt.ylabel(r'len($\mathbf{p}_0^\mathrm{sel}$) + len($\mathbf{p}_2^\mathrm{sel}$)')
plt.xlim(e[0], e[-1])
plt.yticks([1e0,1e1,1e2])
#plt.ylim(bottom=1)
plt.yscale('log')
#plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig('Plots/Newtonian_orders_needed.pdf')

#Runtime
print("\nRuntime: %s seconds" % np.around((time.time() - start_runtime),6))

plt.show()


