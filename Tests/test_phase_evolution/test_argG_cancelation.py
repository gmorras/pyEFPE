import numpy as np
import time
from tDomain_integrate_RR_funcs import *
import matplotlib.pyplot as plt

start_runtime = time.time()

#################################

#number of tests to do
N_test = int(1e7)
Nbins = 100

#################################

#generate random spins uniform in direction and magnitude
spin1_2 = np.random.uniform(low=[-1,-1,-1], high=[1,1,1], size=(2, N_test, 3))
s_1_2 = np.random.uniform(low=0, high=1, size=(2,N_test))
spin1, spin2 = spin1_2*((s_1_2/np.linalg.norm(spin1_2, axis=-1))[:,:,np.newaxis])
spin1_norm, spin2_norm = s_1_2

#generate random mass ratios
m1 = 1
m2 = np.random.uniform(0, m1, size=N_test)

#generate random PN parameters
y = np.exp(np.random.uniform(np.log(0.001), np.log(6**-0.5), size=N_test))

#now compute mass related stuff
M = m1 + m2           #total mass
mu1, mu2 = m1/M, m2/M #reduced individual masses
nu = mu1*mu2          #symmetric mass ratio 
dmu = mu1 - mu2       #dimensionless mass diference

#compute the reduced spins as defined in Eq.(7). Note that S = mu^2 spin
s0_1, s0_2 = mu1[:, np.newaxis]*spin1, mu2[:, np.newaxis]*spin2

#compute the components of the spin we need
sz_1, sz_2 = s0_1[:,2], s0_2[:,2]
sp2_1, sp2_2 = np.sum(s0_1[:,:2]**2, axis=-1), np.sum(s0_2[:,:2]**2, axis=-1)
DJ2 = 2*(s0_1[:,0]*s0_2[:,0] + s0_1[:,1]*s0_2[:,1])

######### copied from function_pyEFPE.basic_prec_quantities #########

#compute chi_eff and dchi0
chi_eff = sz_1 + sz_2
dchi0 = sz_1 - sz_2

#compute (y^2 q) and (y^3 q) from Eqs.(29,30). We have substituted the coefficients B, C and D from Eqs.(B1-B3) already
#where we have substituted J**2 = L**2 + 2*L*(mu1*sz_1 + mu2*sz_2) + 2*nu*sz_1*sz_2 + (mu1**2)*s2_1 + (mu2**2)*s2_2 + nu*DJ2
#compute stuff that will be used to simplyfy equations
dmu2 = dmu*dmu
j2a = (dmu2/y) + (chi_eff*chi_eff)*y #related to modulus of aligned angular momentum
dChi = dmu*dchi0
bp = y*(sp2_1 + sp2_2 + DJ2)
dp = y*dmu2*(4*sp2_1*sp2_2 - DJ2*DJ2)

#compute part of p (Eq.(29)) that does not vanish in the aligned spin case
pal = (j2a - 2*dChi + bp)/3
pal2 = pal*pal
#compute perpendicular part of p (i.e. that vanishes in the aligned spin case)
pperp = bp*dChi - dmu*(DJ2*dmu + (sp2_1 - sp2_2)*y*chi_eff)
pal_pal2pperp = pal*(pal2 + pperp)
	
#compute (y^2 p) and (y^3 q) from Eqs.(29,30)
py2 = 3*pal2 + 2*pperp
qy3 = -2*pal_pal2pperp + dp
	
#compute y^6 times the cubic discriminant (p/3)**3 - (0.5*q)**2 of Eq.(33). We expand in terms of pal, pperp and dp to avoid numerical errors
discy6 = (pperp*pperp)*(9*pal2 + 8*pperp)/27 + dp*(pal_pal2pperp - 0.25*dp)

#compute arg(G)/3 from Eq.(33). The discriminant has to be larger than 0 since there are three real roots in cubic equation.
argG_3 = np.where(discy6>0, np.arctan2(discy6**0.5, -0.5*qy3)/3, 0)
	
#compute the prefactor that multiplies the Y's in Eqs.(31,32)
Y_pref = 2*(abs(py2)**0.5)/y
	
#compute Y_+ - Y_- using that cos(x - 2*pi/3) - cos(x + 2*pi/3) = sqrt(3) sin(x)
YpmYm = Y_pref*np.sin(argG_3)

#compute Y_+ + Y_- using that cos(x - 2*pi/3) + cos(x + 2*pi/3) = -cos(x)
YppYm = -(3**-0.5)*Y_pref*np.cos(argG_3)

#compute Y_3 - Y_- using that cos(x) - cos(x + 2*pi/3) = sqrt(3) cos(x - pi/6)
Y3mYm = Y_pref*np.cos(argG_3 - (np.pi/6))

#compute B from Eq.(B1)
B = -j2a - dChi - bp
	
#compute dchi_diff=(chi_+ - chi_-)/2 and dchi_av=(chi_+ + chi_-)/2
dY = B/(3*y) #compute dY from Eq.(34)
dchi_av = y*(0.5*YppYm - dY)/dmu
dchi_diff = 0.5*y*YpmYm/dmu

#compute m from Eq.(38)
m = YpmYm/Y3mYm
	
#compute sqrt(Y3 - Y_-), which is what actually appears in equations
sqY3mYm = Y3mYm**0.5
	
#compute the newtonian angular momentum from Eq.(8)
L = nu/y

#Define sum of moduli of perpendicular part of spins S0_perp_1^2 + S0_perp_2^2
Sperp2_1 = (mu1*mu1)*sp2_1
Sperp2_2 = (mu2*mu2)*sp2_2
	
#compute component of J0 parallel to L (J0 \cdot \hat{L})
J0Lh = L + 0.5*(chi_eff + dChi)
J0Lh2 = J0Lh*J0Lh
	
#compute squared perpedicular component of J
Jperp2 = Sperp2_1 + Sperp2_2 + nu*DJ2

#compute J by adding parallel and perpendicular moduli
J = (J0Lh2 + Jperp2)**0.5
	
#compute expansion factor in J = \sqrt{1 + 2*x}
xJ = 0.5*Jperp2/J0Lh2

#compute also J \pm J0Lh, considering the cases where x is small
small_x = (abs(xJ)<1e-6)
dJ_small_x = J0Lh*xJ*(1 - 0.5*xJ*(1-xJ))
J_p_J0Lh = np.where(small_x & (J0Lh<0),-dJ_small_x, J + J0Lh)
J_m_J0Lh = np.where(small_x & (J0Lh>0), dJ_small_x, J - J0Lh)

#compute Np=N_+ and Nm=N_- from Eqs.(41-42)
muSz = 2*(mu1*mu1*sz_1 + mu2*mu2*sz_2)
dmudSp2 = dmu*(Sperp2_1-Sperp2_2)
Np = J_p_J0Lh*(J_p_J0Lh - muSz)-dmudSp2
Nm = J_m_J0Lh*(J_m_J0Lh + muSz)-dmudSp2

#compute Cp and Cm from Eqs.(26,43-46)
Cp =  dchi_diff*dmu
Cm = - Cp
	
#compute dmu \Delta chi_- = dmu(dchi_- - (s1z-s2z)) from dchi_av and dchi_diff
dmuDdchim = dmu*((dchi_av - dchi0) - dchi_diff)
	
#compute (Bp-Cp) and (Bm-Cm) from Eqs.(26,43-46)
Bp_Cp = 2*J_p_J0Lh + dmuDdchim
Bm_Cm = 2*J_m_J0Lh - dmuDdchim

#compute prefactor's to elliptic PI's appearing in Eq.(109) of 2106.10291
PI_fact_p = Np/Bp_Cp
PI_fact_m = Nm/Bm_Cm

#compute also the arguments
PI_arg_p = -2*Cp/Bp_Cp
PI_arg_m = -2*Cm/Bm_Cm

#compute Pp and Pm appearing in Eq.(109) of 2106.10291
Pp = PI_fact_p*my_ellipPI(PI_arg_p, m)
Pm = PI_fact_m*my_ellipPI(PI_arg_m, m)

#####################################################################

from matplotlib import pyplot as plt
plt.rcParams.update({'font.size': 24})
plt.rcParams.update({'lines.linewidth': 2})

#compute relative difference
x = discy6/(np.abs(qy3/2)**2 + np.abs(py2/3)**3)
bins = np.geomspace(np.amin(x), np.amax(x), Nbins+1)

#find number of samples bellow threashold
x_threas = 1e-12
print('Fraction of samples with Delta_G<%.3g: %s \n'%(x_threas, np.sum(x<x_threas)/len(x)))

plt.figure(figsize=(13,8),dpi=100)
plt.hist(x, bins=bins, histtype = 'step', linewidth=3.5, color='C0')
plt.hist(x, bins=bins, histtype = 'stepfilled', linewidth=3.5,alpha=0.25, color='C0')
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'$\Delta_G = \left[\left(\frac{p}{3}\right)^3 - \left(\frac{q}{2}\right)^2 \right]/\left[\left|\frac{p}{3}\right|^3 + \left|\frac{q}{2}\right|^2  \right]$')
plt.xlim(bins[0], bins[-1]);
plt.ylabel(r'Number of samples');
plt.tight_layout();
plt.savefig('Plots/argG_cancelation.pdf')

######################## old way to compute stuff ########################

#compute individual modulus of spins from Eq.(7)
S2_1 = (mu1**2)*(sp2_1 + sz_1*sz_1)
S2_2 = (mu2**2)*(sp2_2 + sz_2*sz_2)

#compare against old way to compute stuff
Np_old = (J+L)*(J+L+2*nu*chi_eff)-dmu*(S2_1-S2_2)
Nm_old = (J-L)*(J-L-2*nu*chi_eff)-dmu*(S2_1-S2_2)

#compute Bp, Cp and Bm, Cm from Eqs.(26,43-46)
Bp_old = 2*(J+L)+chi_eff+dchi_av*dmu
Bm_old = 2*(J-L)-chi_eff-dchi_av*dmu
Cp_old =  dchi_diff*dmu
Cm_old = -dchi_diff*dmu

#########################################################################

#functions to compute absolute and relative errors
def error_in_var(v1, v2, name=None):

	#compute absolute error
	abs_err = np.abs(v1 - v2)
	#compute the root mean square and maximum of this
	abs_err_rms = np.sqrt(np.mean(np.square(abs_err)))
	abs_err_max = np.amax(abs_err)
	
	#compute relative error
	rel_err = 0.5*np.abs(v1 - v2)/(np.abs(v1) + np.abs(v2))
	#compute the root mean square and maximum of this
	rel_err_rms = np.sqrt(np.mean(np.square(rel_err)))
	rel_err_max = np.amax(rel_err)

	if name is not None:
		print('Error in %s: Absolute: RMS: %.3g  Max:%.3g || Relative: RMS: %.3g  Max:%.3g'%(name, abs_err_rms, abs_err_max, rel_err_rms, rel_err_max))
	
	#return errors 
	return abs_err_rms, abs_err_max, rel_err_rms, rel_err_max	

#test that rewritting of Bp/m and Cp/m is correct
error_in_var(Np, Np_old, name='Np')
error_in_var(Nm, Nm_old, name='Nm')
error_in_var(Bp_Cp, Bp_old - Cp_old, name='Bp-Cp')
error_in_var(Bm_Cm, Bm_old - Cm_old, name='Bm-Cm')
error_in_var(Bm_Cm*(Bm_Cm + 2*Cm), Bm_old**2 - Cm_old**2, name='Bm^2-Cm^2')
print()

#########################################################################

#new way to compute dmu*(dchi_av - dchi0) and dmu*dchi_diff
sargG_3, cargG_3 = np.sin(argG_3), np.cos(argG_3)
sqpy2 = py2**0.5
dmudchiav_m_dchi0 = np.where((py2>0) & (pal>0), (pal2*sargG_3*sargG_3 - (2/3)*pperp*cargG_3*cargG_3)/(pal + (3**-0.5)*sqpy2*cargG_3), pal - (3**-0.5)*sqpy2*cargG_3)
dmudchi_diff = sqpy2*sargG_3

#use this to compute Bm_Cm_new
Bm_Cm_new =  2*J_m_J0Lh - (dmudchiav_m_dchi0 - dmudchi_diff)

error_in_var(dmudchiav_m_dchi0, dmu*(dchi_av - dchi0), name='dmu*(dchi_av - dchi0)')
error_in_var(dmudchi_diff, dmu*dchi_diff, name='dmu*dchi_diff')

error_in_var(Bm_Cm_new*(Bm_Cm_new - 2*dmudchi_diff), Bm_Cm*(Bm_Cm + 2*Cm), name='Bm^2-Cm^2')
print()

#compute dchi_av and dchi_diff from dmu*(dchi_av - dchi0) and dmu*dchi_diff
dchi_av_new = dchi0 + (dmudchiav_m_dchi0/dmu)
dchi_diff_new = dmudchi_diff/dmu

#compute the small dmu approximations
sp2_tot = sp2_1 + sp2_2 + DJ2
s2_tot = sp2_tot + chi_eff*chi_eff
dchi_av_new_small_dmu = chi_eff*(sp2_1 - sp2_2 + dchi0*chi_eff)/s2_tot
dchi_diff_new_small_dmu = np.sqrt(sp2_tot*(4*(sp2_2*sz_1*sz_1 + sp2_1*(sz_2*sz_2 + sp2_2))-DJ2*(4*sz_1*sz_2 + DJ2)))/s2_tot

#########################################################################

delta_dchi_av = 0.5*np.abs(dchi_av_new - dchi_av_new_small_dmu)/(np.abs(dchi_av_new)+np.abs(dchi_av_new_small_dmu))
delta_dchi_diff = 0.5*np.abs(dchi_diff_new - dchi_diff_new_small_dmu)/(np.abs(dchi_diff_new)+np.abs(dchi_diff_new_small_dmu))

#make a scatter plot
idxs_dmu_small = ((dmu/y)<1e-3)
plt.figure(figsize=(13,8),dpi=100)
plt.scatter((dmu/y)[idxs_dmu_small], delta_dchi_av[idxs_dmu_small], label=r'$ 2\frac{| \delta\chi_\mathrm{av} - \delta\chi_\mathrm{av}^\mathrm{approx}|}{|\delta\chi_\mathrm{av}| +| \delta\chi_\mathrm{av}^\mathrm{approx}|}$', alpha=0.25)
plt.scatter((dmu/y)[idxs_dmu_small], delta_dchi_diff[idxs_dmu_small], label=r'$ 2\frac{| \delta\chi_\mathrm{diff} - \delta\chi_\mathrm{diff}^\mathrm{approx}|}{|\delta\chi_\mathrm{diff}| +| \delta\chi_\mathrm{diff}^\mathrm{approx}|}$', alpha=0.25)
plt.xlabel(r'$\delta\mu/y$')
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.tight_layout()

#make a histogram
x = np.abs(dmudchiav_m_dchi0)/np.abs(pal) #np.abs(dmudchiav_m_dchi0)/np.abs(dChi)
bins = np.geomspace(max(np.amin(x),1e-20), np.amax(x), 100)
plt.figure(figsize=(13,8),dpi=100)
plt.hist(x, bins = bins, histtype = 'stepfilled')
plt.xscale('log');
plt.yscale('log');
plt.tight_layout()

#Runtime
print("\nRuntime: %s seconds" % (time.time() - start_runtime))
plt.show()

