import os
os.environ.update(
    OMP_NUM_THREADS = '1',
    OPENBLAS_NUM_THREADS = '1',
    NUMEXPR_NUM_THREADS = '1',
    MKL_NUM_THREADS = '1',
)

import numpy as np
from tDomain_integrate_RR_funcs import *
from matplotlib import pyplot as plt
plt.rcParams.update({'font.size': 24})
plt.rcParams.update({'lines.linewidth': 2})
import os
if not os.path.exists('Plots/'): os.makedirs('Plots/')

#universal constants
t_sun = 4.92549094831e-6 #GMsun/c**3 [s]

import time
start_runtime = time.time()

#function to compute the dP- entering in Euler angles
def compute_dPm_parts(bpsip, y, DJ2, m1, m2, sz_1, sz_2, sp2_1, sp2_2):

	#compute mass related stuff
	M, mu1, mu2, nu, dmu = mass_params_from_m1_m2(m1, m2)

	#compute basic quantities coming from solving precession equation in 2106.10291
	m, dchi_av, dchi_diff, chi_eff, J, L, sqY3mYm, Pp, Pm, PI_fact_p, PI_fact_m, PI_arg_p, PI_arg_m = basic_prec_quantities(y, DJ2, m1, m2, sz_1, sz_2, sp2_1, sp2_2, only_for_Dv=False)

	#compute K(m)
	K_m = scipy.special.ellipk(m)

	#compute hbpsip_pi_2 = (2/pi)\hat{\overline{\psi}}_p (after Eq.49)
	hbpsip_pi_2 = np.mod((2/np.pi)*bpsip + 1, 2) - 1
	
	#compute the value of \hat{\psi}_p from \overline{\psi}_p using Eq.(95)
	hpsip = K_m*hbpsip_pi_2

	#compute the Jacobic elliptic functions. These correspond to sn and am of Eq.(A7)
	sn, cn, dn, am = scipy.special.ellipj(hpsip, m)

	#compute the incomplete elliptic integral of the third kind (Eq.(52))
	nusqY3mYm = nu*sqY3mYm
	ellipPI_p_inc = (PI_fact_p*my_ellipPI(PI_arg_p, m, phi=am) - hbpsip_pi_2*Pp)/nusqY3mYm
	ellipPI_m_inc = (PI_fact_m*my_ellipPI(PI_arg_m, m, phi=am) - hbpsip_pi_2*Pm)/nusqY3mYm
	
	#return variation of Euler angles on precession timescales, forcing costhL to be between -1 and 1
	return PI_fact_m*my_ellipPI(PI_arg_m, m, phi=am)/nusqY3mYm, hbpsip_pi_2*Pm/nusqY3mYm, ellipPI_m_inc


#function to compute derivative of Euler angles
def compute_deuler_angles(bpsip, y, e2, DJ2, m1, m2, sz_1, sz_2, sp2_1, sp2_2, use_dJ=True):

	#compute mass related stuff
	M, mu1, mu2, nu, dmu = mass_params_from_m1_m2(m1, m2)
	
	#take into account dJ
	if use_dJ:
		#compute basic quantities coming from solving precession equation in 2106.10291
		m, dchi_av, dchi_diff, chi_eff, J, L, sqY3mYm, Pp, Pm, PI_fact_p, PI_fact_m, PI_arg_p, PI_arg_m = basic_prec_quantities(y, DJ2, m1, m2, sz_1, sz_2, sp2_1, sp2_2, only_for_Dv=False, min_2J1pmcthL=-1)

		#compute E(m) and K(m)
		K_m = scipy.special.ellipk(m)
		E_m = scipy.special.ellipe(m)
		
		#compute hbpsip_pi_2 = (2/pi)\hat{\overline{\psi}}_p (after Eq.49)
		hbpsip_pi_2 = np.mod((2/np.pi)*bpsip + 1, 2) - 1
	
		#compute the value of \hat{\psi}_p from \overline{\psi}_p using Eq.(95)
		hpsip = K_m*hbpsip_pi_2

		#compute the Jacobic elliptic functions. These correspond to sn and am of Eq.(A7)
		sn, cn, dn, am = scipy.special.ellipj(hpsip, m)

		 #compute \delta DJ2 \approx 2 J0 \delta J/nu from Eq.(91). Use 2*dmu*dchi_diff/(m*sqY3mYm) = y*sqY3mYm. There is a factor of 2 missing in Eq.(91)
		DJ2 += (4/3)*nu*sqY3mYm*((32 + 28*e2)/5)*(y*y/(1 - y*chi_eff))*(scipy.special.ellipeinc(am, m) - E_m*hbpsip_pi_2)
			
	#compute basic quantities coming from solving precession equation in 2106.10291
	m, dchi_av, dchi_diff, chi_eff, J, L, sqY3mYm, Pp, Pm, PI_fact_p, PI_fact_m, PI_arg_p, PI_arg_m = basic_prec_quantities(y, DJ2, m1, m2, sz_1, sz_2, sp2_1, sp2_2, only_for_Dv=False)

	#compute also the prefactor of the first term in \delta\zeta of Eq.(52), we simplify
	#2*dmu*dchi_diff/(m*sqY3mYm) = y*sqY3mYm in C=2*dmu*dchi_diff/(3*m*(1-y*chi_eff)*sqY3mYm)
	dzeta_E_fact = y*sqY3mYm/(3*(1-y*chi_eff))
	
	#compute K(m)
	K_m = scipy.special.ellipk(m)
	
	#compute hbpsip_pi_2 = (2/pi)\hat{\overline{\psi}}_p (after Eq.49)
	hbpsip_pi_2 = np.mod((2/np.pi)*bpsip + 1, 2) - 1
	
	#compute the value of \hat{\psi}_p from \overline{\psi}_p using Eq.(95)
	hpsip = K_m*hbpsip_pi_2

	#compute the Jacobic elliptic functions. These correspond to sn and am of Eq.(A7)
	sn, cn, dn, am = scipy.special.ellipj(hpsip, m)

	#compute cos(\theta_L) from Eqs.(15, 26)
	dchi = dchi_av - dchi_diff*(1 - 2*sn*sn)
	costhL = (L + 0.5*(chi_eff + dmu*dchi))/J

	#compute Dphiz from Eq.(40)
	Dphiz = (y**6)*(0.5*J + (3/(4*nu))*(1-y*chi_eff)*((PI_fact_p/(1 - PI_arg_p*sn*sn)) + (PI_fact_m/(1 - PI_arg_m*sn*sn))))
	
	#compute Dzeta using minimal rotation condition
	Dzeta = -costhL*Dphiz

	#compute D \equiv M/(1-e^2)^{3/2} d/dt (Eq.(4))
	D_fact = ((1 - e2)**1.5)/M

	return D_fact*Dphiz, D_fact*Dzeta

################## Inputs #######################

#number of times for integral
Nts   = 1000000

#number of time for plots
Nplot = 100000

#final orbitañ frequency
ff_orb= 1e5

#qi is the quadrupole parameter, where q=1 for BHs (see Appendix A of 1801.08542)
q1=1
q2=1

############ force discontinuity in P- ############
#m1, m2, e0, f0_orb, spin1, spin2, theta_L, phi_L, theta_N, phi_N, phi0, phi_e0  = 4.3750202725600555, 1.85247603570533, 0.37555337745344125, 5, [0.07869126641711355, 0.08571166972247439, 0.022455007920716835], [0.3261674410165153, 0.5449188036724386, -0.3088043941781543], 0, 0, 0, 0, 0, 0

#m1, m2, e0, f0_orb, spin1, spin2, theta_L, phi_L, theta_N, phi_N, phi0, phi_e0  = 2.789252101151268, 1.7744995201380507, 0.5858498991324617, 5, [-0.22801920399972156, -0.016670042646158685, -0.1830257132608543], [-0.5793314528490882, 0.3032271469041688, 0.028379817453835044], 0, 0, 0, 0, 0, 0

#m1, m2, e0, f0_orb, spin1, spin2, theta_L, phi_L, theta_N, phi_N, phi0, phi_e0  = 2.8198071836313647, 1.3127703803692006, 0.4540835132719014, 5, [-0.1377583713633159, 0.12483621224804303, 0.1562235820307167], [-0.7212638818927588, -0.21623076365551594, -0.08256298319648378], 0, 0, 0, 0, 0, 0

#m1, m2, e0, f0_orb, spin1, spin2, theta_L, phi_L, theta_N, phi_N, phi0, phi_e0  = 4.561421015325589, 1.354862432845054, 0.1983870948830915, 5, [-0.03626494101421062, -0.048742653558487364, -0.029863720945828336], [-0.3960815189285696, -0.5947888599603085, -0.29224028550490916], 0, 0, 0, 0, 0, 0

############ force transitional precesion ############

#m1, m2, e0, f0_orb, spin1, spin2, theta_L, phi_L, theta_N, phi_N, phi0, phi_e0  = 20.962836942184314, 2.7834784635411967, 0.23590466234181431, 5, [-0.0006331939883143971, 0.00022612252112697492, -0.4351812901088922], [0.013918612379978462, -0.011514267183162301, -0.007696054724892697], 0, 0, 0, 0, 0, 0

#transitional precession with discontinuity in P+
#m1, m2, e0, f0_orb, spin1, spin2, theta_L, phi_L, theta_N, phi_N, phi0, phi_e0  = 16.253632374076453, 2.914220837703304, 0.6338521295121787, 5, [-0.007373181861752118, 0.001548416640899919, -0.898956726743585], [0.18376194268145338, -0.19209061355192356, -0.672329380136111], 0, 0, 0, 0, 0, 0

m1, m2, e0, f0_orb, spin1, spin2, theta_L, phi_L, theta_N, phi_N, phi0, phi_e0  = 34.428070451062744, 2.068748141854075, 0.4264256595680419, 6.75, [0.004336137223970934, -0.005606030434483803, -0.7934271237539297], [-0.007257895428127617, 0.4781962487733726, 0.025701708721412083], 0, 0, 0, 0, 0, 0


#################################################

#convert solar masses to seconds
m1, m2 = m1*t_sun, m2*t_sun

#compute initial conditions
v_ini, yf, m1, m2, sz_1, sz_2, sp2_1, sp2_2, q1, q2, thJN, phJN, psi_pol = initial_conditions_for_RR_eqs(f0_orb, ff_orb, e0, m1, m2, spin1, spin2, theta_L, phi_L, theta_N, phi_N, phi0, phi_e0, q1, q2)

#compute mass stuff and print info
M, mu1, mu2, nu, dmu = mass_params_from_m1_m2(m1, m2)
print('\ndmu=%.3g, sz_1=%.3g, sz_2=%.3g, sp2_1=%.3g, sp2_2=%.3g'%(dmu, sz_1, sz_2, sp2_1, sp2_2))

#spin related stuff
chi_eff = sz_1 + sz_2
s2_1 = sp2_1 + sz_1*sz_1
s2_2 = sp2_2 + sz_2*sz_1

#initialize class to compute PN derivatives
PN_derivatives = pyEFPE_PN_derivatives(m1, m2, chi_eff, s2_1, s2_2, q1, q2)

#compute sol
start_soltime = time.time()
RR_sol_interp = solve_ivp_RR_eqs_t(v_ini, PN_derivatives, yf, m1, m2, sz_1, sz_2, sp2_1, sp2_2,
rtol=[1e-8,1e-8,1e-12,1e-12,1e-12,1e-12,1e-12,1e-12],
atol=[1e-12,1e-12,1e-8,1e-8,1e-12,1e-8,1e-6,1e-6])
print("\nRR solving runtime: %s seconds \n" % (time.time() - start_soltime))

#make an array of times
ts = -np.geomspace(-RR_sol_interp.all_ts[0], -RR_sol_interp.all_ts[-1], Nts)

#compute solution v=[y, e2, l, dl, DJ2, bpsip, phiz0, zeta0] at these times
start_soltime = time.time()
y, e2, l, dl, DJ2, bpsip, phiz0, zeta0 = RR_sol_interp(ts)
print("\nTime to evaluate solution with class: %s seconds \n" % (time.time() - start_soltime))

#compute the derivatives of the Euler angles
dphiz_dt, dzeta_dt = compute_deuler_angles(bpsip, y, e2, DJ2, m1, m2, sz_1, sz_2, sp2_1, sp2_2)

#integrate Euler angles
from scipy.integrate import cumulative_trapezoid
phiz_int = cumulative_trapezoid(dphiz_dt, x=ts, initial=0)
zeta_int = cumulative_trapezoid(dzeta_dt, x=ts, initial=0)

#thin stuff for plots
diplot = int(Nts/Nplot)
dphiz_dt, dzeta_dt, phiz_int, zeta_int = dphiz_dt[::diplot], dzeta_dt[::diplot], phiz_int[::diplot], zeta_int[::diplot]
y, e2, l, dl, DJ2, bpsip, phiz0, zeta0 = y[::diplot], e2[::diplot], l[::diplot], dl[::diplot], DJ2[::diplot], bpsip[::diplot], phiz0[::diplot], zeta0[::diplot]

#compute Euler angles approximately
dphiz, dzeta, costhL = precesion_Euler_angles(bpsip, *constants_precesion_Euler_angles(y, DJ2, m1, m2, sz_1, sz_2, sp2_1, sp2_2))
phiz_approx = phiz0 + dphiz
zeta_approx = zeta0 + dzeta

#compute thetaL
thL = np.arccos(costhL)

#compute variable proportional to precesion cycles
npc = bpsip/np.pi - 0.5

#make a plot of derivatives of Euler angles
plt.figure(figsize=(16,10))
plt.plot(npc, dphiz_dt, label=r'$d\phi_z/dt$')
plt.plot(npc, dzeta_dt, label=r'$d\zeta/dt$')
plt.xlabel('$\overline{\psi}_p/\pi - 1/2$')
plt.xlim(npc[0], npc[-1])
plt.grid(True)
plt.legend()
plt.tight_layout()

#make a plot of Euler angles
plt.figure(figsize=(16,10))
plt.plot(npc, phiz_int - phiz0, 'C0-', alpha=0.5, label=r'$\phi_z - \phi_{z,0}^\mathrm{approx}$')
plt.plot(npc, dphiz, 'C0--', alpha=0.5, label=r'$\delta\phi_{z,0}^\mathrm{approx}$')
plt.plot(npc, zeta_int - zeta0, 'C1-', alpha=0.5, label=r'$\zeta - \zeta_{0}^\mathrm{approx}$')
plt.plot(npc, dzeta, 'C1--', alpha=0.5, label=r'$\delta\zeta_{0}^\mathrm{approx}$')
plt.plot(npc, phiz_int + zeta_int, 'C2-', alpha=0.5, label=r'$\phi_z + \zeta$')
plt.plot(npc, phiz_approx + zeta_approx, 'C2--', alpha=0.5, label=r'$\phi_z^\mathrm{approx} + \zeta^\mathrm{approx}$')
plt.xlabel('$\overline{\psi}_p/\pi - 1/2$')
plt.xlim(npc[0], npc[-1])
plt.grid(True)
plt.legend()
plt.tight_layout()

#make a plot of different parts of incomplete Pm
Pm_inc, Pm_av, dPm = compute_dPm_parts(bpsip, y, DJ2, m1, m2, sz_1, sz_2, sp2_1, sp2_2)
plt.figure(figsize=(16,10))
plt.plot(npc, Pm_inc, label=r'$P_-(\mathrm{am}(\hat{\psi}_p,m))$')
plt.plot(npc,  Pm_av, label=r'$P_- \hat{\psi}_p/K(m)$')
plt.plot(npc,    dPm, label=r'$\delta P_- = P_-(\mathrm{am}(\hat{\psi}_p,m)) - P_- \hat{\psi}_p/K(m)$')
plt.xlabel('$\overline{\psi}_p/\pi - 1/2$')
plt.xlim(npc[0], npc[-1])
plt.grid(True)
plt.legend()
plt.tight_layout()

#make a plot of quaternion components
plt.figure(figsize=(16,10))
#plt.plot(npc, np.cos(0.5*thL), 'r-', label=r'$\cos\frac{\theta}{2}$')
plt.plot(npc, np.cos(0.5*thL)*np.cos(0.5*(phiz_int + zeta_int)), 'C0-', label=r'$\cos\frac{\theta}{2}\cos{\frac{\phi_z + \zeta}{2}}$')
plt.plot(npc, np.cos(0.5*thL)*np.cos(0.5*(phiz_approx + zeta_approx)), 'k--')
#plt.plot(npc, np.sin(0.5*thL), 'r--', label=r'$\sin\frac{\theta}{2}$')
plt.plot(npc, np.sin(0.5*thL)*np.cos(0.5*(phiz_int - zeta_int)), 'C1-', label=r'$\sin\frac{\theta}{2}\cos{\frac{\phi_z - \zeta}{2}}$')
plt.plot(npc, np.sin(0.5*thL)*np.cos(0.5*(phiz_approx - zeta_approx)), 'k--')
plt.xlabel('$\overline{\psi}_p/\pi - 1/2$')
plt.xlim(npc[0], npc[-1])
plt.grid(True)
plt.legend()
plt.tight_layout()


#Runtime
print("\nRuntime: %s seconds" % (time.time() - start_runtime))

plt.show()
