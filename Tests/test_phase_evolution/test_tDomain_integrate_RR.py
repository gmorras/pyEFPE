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

#number of times for plots
Nts = 100000

#Primary mass
m1 = 1.2*t_sun

#mass ratio
q = 0.5 #np.random.uniform(low=0.2, high=1)
m2 = q*m1

#qi is the quadrupole parameter, where q=1 for BHs (see Appendix A of 1801.08542)
q1 = 1
q2 = 1

#frequencies to consider (Hz)
freqs = np.geomspace(0.5, 100, Nts)

#harmonics to consider
nmax = 6
mmax = 6

#minimum and maximum harmonic to consider for amplitudes
pmin = -3
pmax = 7

#initial conditions
f0_orb= 10
ff_orb= 1e5
e0 = 0.7
theta_L, phi_L, theta_N, phi_N, phi0, phi_e0 = np.random.uniform(low=[0,0,0,0,0,0], high=[np.pi,2*np.pi,np.pi,2*np.pi, 2*np.pi, 2*np.pi])
spin1, spin2 = np.random.uniform(low=[-1,-1,-1], high=[1,1,1], size=(2,3))
s_1, s_2 = np.random.uniform(low=0.2, high=1), np.random.uniform(low=0.2, high=1)
spin1, spin2 = s_1*spin1/np.linalg.norm(spin1), s_2*spin2/np.linalg.norm(spin2)
#parameters to force discontinuity on \delta\zeta and \delta\phi_z
#theta_L, phi_L, theta_N, phi_N, phi0, phi_e0 = [2.8024639,  1.6078873,  0.46666412, 3.54295848, 3.15420125, 3.04625859]
#spin1, spin2 = [0.11624348, 0.0259301, 0.68979351] , [-0.57947872, -0.41343063, 0.5507082]
theta_L, phi_L, theta_N, phi_N, phi0, phi_e0 = [0,  0,  0, 0, 0, 0]
#spin1, spin2 = [1e-3, 1e-4, 0.68979351] , [1e-4, 1e-3, 0.5507082]
print('Mass ratio =', q)
print('Angles =', np.array([theta_L, phi_L, theta_N, phi_N, phi0, phi_e0]))
print('spin1, spin2 =', np.array(spin1), ',', np.array(spin2))

############ force discontinuity in P- ############
#m1, m2, e0, f0_orb, spin1, spin2, theta_L, phi_L, theta_N, phi_N, phi0, phi_e0  = 4.3750202725600555*t_sun, 1.85247603570533*t_sun, 0.37555337745344125, 5, [0.07869126641711355, 0.08571166972247439, 0.022455007920716835], [0.3261674410165153, 0.5449188036724386, -0.3088043941781543], 0, 0, 0, 0, 0, 0

#m1, m2, e0, f0_orb, spin1, spin2, theta_L, phi_L, theta_N, phi_N, phi0, phi_e0  = 2.789252101151268*t_sun, 1.7744995201380507*t_sun, 0.5858498991324617, 5, [-0.22801920399972156, -0.016670042646158685, -0.1830257132608543], [-0.5793314528490882, 0.3032271469041688, 0.028379817453835044], 0, 0, 0, 0, 0, 0

#m1, m2, e0, f0_orb, spin1, spin2, theta_L, phi_L, theta_N, phi_N, phi0, phi_e0  = 2.8198071836313647*t_sun, 1.3127703803692006*t_sun, 0.4540835132719014, 5, [-0.1377583713633159, 0.12483621224804303, 0.1562235820307167], [-0.7212638818927588, -0.21623076365551594, -0.08256298319648378], 0, 0, 0, 0, 0, 0

#m1, m2, e0, f0_orb, spin1, spin2, theta_L, phi_L, theta_N, phi_N, phi0, phi_e0  = 4.561421015325589*t_sun, 1.354862432845054*t_sun, 0.1983870948830915, 5, [-0.03626494101421062, -0.048742653558487364, -0.029863720945828336], [-0.3960815189285696, -0.5947888599603085, -0.29224028550490916], 0, 0, 0, 0, 0, 0

############ force transitional precesion ############

#m1, m2, e0, f0_orb, spin1, spin2, theta_L, phi_L, theta_N, phi_N, phi0, phi_e0  = 20.962836942184314*t_sun, 2.7834784635411967*t_sun, 0.23590466234181431, 5, [-0.0006331939883143971, 0.00022612252112697492, -0.4351812901088922], [0.013918612379978462, -0.011514267183162301, -0.007696054724892697], 0, 0, 0, 0, 0, 0

#transitional precession with discontinuity in P+
#m1, m2, e0, f0_orb, spin1, spin2, theta_L, phi_L, theta_N, phi_N, phi0, phi_e0  = 16.253632374076453*t_sun, 2.914220837703304*t_sun, 0.6338521295121787, 5, [-0.007373181861752118, 0.001548416640899919, -0.898956726743585], [0.18376194268145338, -0.19209061355192356, -0.672329380136111], 0, 0, 0, 0, 0, 0

#m1, m2, e0, f0_orb, spin1, spin2, theta_L, phi_L, theta_N, phi_N, phi0, phi_e0  = 33.61280041274842*t_sun, 2.0869654273660374*t_sun, 0.4611617612760155, 6.75, [-0.0003233866242412857, 0.00011205798053836189, -0.36698364530230576], [0.08491285606611519, -0.08834794153713839, 0.5302918801385069], 0, 0, 0, 0, 0, 0

m1, m2, e0, f0_orb, spin1, spin2, theta_L, phi_L, theta_N, phi_N, phi0, phi_e0  = 34.428070451062744*t_sun, 2.068748141854075*t_sun, 0.4264256595680419, 6.75, [0.004336137223970934, -0.005606030434483803, -0.7934271237539297], [-0.007257895428127617, 0.4781962487733726, 0.025701708721412083], 0, 0, 0, 0, 0, 0


#compute initial conditions
v_ini, yf, m1, m2, sz_1, sz_2, sp2_1, sp2_2, q1, q2, thJN, phJN, psi_pol = initial_conditions_for_RR_eqs(f0_orb, ff_orb, e0, m1, m2, spin1, spin2, theta_L, phi_L, theta_N, phi_N, phi0, phi_e0, q1, q2)

#compute mass and spin related stuff
M = m1 + m2
mu1, mu2 = m1/M, m2/M
nu = mu1*mu2
dmu = mu1 - mu2 
chi_eff = sz_1 + sz_2
s2_1 = sp2_1 + sz_1*sz_1
s2_2 = sp2_2 + sz_2*sz_1

#initialize class to compute PN derivatives
PN_derivatives = pyEFPE_PN_derivatives(m1, m2, chi_eff, s2_1, s2_2, q1, q2)

#compute sol
start_soltime = time.time()
RR_sol_interp = solve_ivp_RR_eqs_t(v_ini, PN_derivatives, yf, m1, m2, sz_1, sz_2, sp2_1, sp2_2)
print("\nRR solving runtime: %s seconds \n" % (time.time() - start_soltime))

#make an array of times
ts = -np.geomspace(-RR_sol_interp.all_ts[0], -RR_sol_interp.all_ts[-1], Nts)

#compute also time to coalescence
ts_coal_M = -ts/M

#compute solution v=[y, e2, l, dl, DJ2, bpsip, phiz0, zeta0] at these times
start_soltime = time.time()
vs = RR_sol_interp(ts)
print("\nTime to evaluate solution with class: %s seconds \n" % (time.time() - start_soltime))

#compute derivatives using PN expressions
PN_derivatives_vec = pyEFPE_PN_derivatives_python(m1, m2, chi_eff, s2_1, s2_2, q1, q2)
start_soltime = time.time()
dv_dt_PN = derivatives_prec_avg(vs[0], vs[1], vs[4], PN_derivatives_vec, m1, m2, sz_1, sz_2, sp2_1, sp2_2)
print("\nTime to evaluate derivatives vectorized:                     %.3g seconds/call \n"%((time.time() - start_soltime)/Nts))

######################### Timings when running things in loop ###########################
#full derivatives
start_soltime = time.time()
for i in range(len(vs[0])):
	derivatives_prec_avg(vs[0,i], vs[1,i], vs[4,i], PN_derivatives, m1, m2, sz_1, sz_2, sp2_1, sp2_2)
print("\nTime to evaluate derivatives in loop:                        %.3g seconds/call"%((time.time() - start_soltime)/Nts))

#basic_prec_quantities
start_soltime = time.time()
for i in range(len(vs[0])):
	basic_prec_quantities(vs[0,i], vs[4,i], m1, m2, sz_1, sz_2, sp2_1, sp2_2, only_for_Dv=True)
print("Time to evaluate basic_prec_quantities in loop:              %.3g seconds/call"%((time.time() - start_soltime)/Nts))
m, dchi_av, dchi_diff, chi_eff, J, L, sqY3mYm, Pp, Pm, PI_fact_p, PI_fact_m, PI_arg_p, PI_arg_m = basic_prec_quantities(vs[0], vs[4], m1, m2, sz_1, sz_2, sp2_1, sp2_2, only_for_Dv=False)
m, dchi_av, dchi_diff, chi_eff, J, L, sqY3mYm, Pp, Pm, dmudchiav_m_dchi0, dmudchi_diff = basic_prec_quantities(vs[0], vs[4], m1, m2, sz_1, sz_2, sp2_1, sp2_2, only_for_Dv=True)

#precesion averaged couplings
start_soltime = time.time()
for i in range(len(vs[0])):
	compute_dchi_dchi2_sperp2_prec_avg(vs[0,i], m[i], dchi_av[i], dchi_diff[i], dmudchiav_m_dchi0[i], dmudchi_diff[i], vs[4,i], sp2_1, sp2_2)
print("Time to evaluate compute_dchi_dchi2_sperp2_prec_avg in loop: %.3g seconds/call"%((time.time() - start_soltime)/Nts))
dchi_prec_avg, dmudchi_prec_avg_m_dchi0, dchi2_prec_avg, sperp2_prec_avg = compute_dchi_dchi2_sperp2_prec_avg(vs[0], m, dchi_av, dchi_diff, dmudchiav_m_dchi0, dmudchi_diff, vs[4], sp2_1, sp2_2)

#PN derivatives
start_soltime = time.time()
for i in range(len(vs[0])):
	PN_derivatives.Dy_De2_Dl_Ddl(vs[0,i], vs[1,i], dchi_prec_avg[i], dchi2_prec_avg[i], sperp2_prec_avg[i])
print("Time to evaluate PN derivatives in loop:                     %.3g seconds/call"%((time.time() - start_soltime)/Nts))

#elliptic k
start_soltime = time.time()
for i in range(len(vs[0])):
	scipy.special.ellipk(m[i])
print("Time to evaluate elliptic K in loop:                         %.3g seconds/call"%((time.time() - start_soltime)/Nts))

#elliptic Pi
start_soltime = time.time()
for i in range(len(vs[0])):
	my_ellipPI(PI_arg_p[i], m[i])
	my_ellipPI(PI_arg_m[i], m[i])
print("\nTime to evaluate elliptic Pi's in loop:                      %.3g seconds/call"%((time.time() - start_soltime)/Nts))


#########################################################################################


#compute derivative of solution differenciating interpolation polynomial
dv_dt_interp = RR_sol_interp(ts, derivative=1)

#compute difference between interpolated and PN derivatives
delta_dv_dt = (dv_dt_PN - dv_dt_interp)/dv_dt_PN

#compute second derivative of solution differenciating interpolation polynomial
ddlambda_interp, dddeltal_interp = RR_sol_interp(ts, derivative=2, idxs=[2,3])

#compute ddlambda_dt2 using PN expression
ddlambda_PN = (1/M)*3*(vs[0]**2)*np.sqrt(1 - vs[1])*(dv_dt_PN[0]*(1 - vs[1]) - 0.5*vs[0]*dv_dt_PN[1])
#same for dddelta_dt2
dddeltal_PN_approx = dv_dt_PN[3]*((5/vs[0])*dv_dt_PN[0] - 1.5*dv_dt_PN[1]/(1-vs[1])) #we are approximatinge D\delta\lambda \propto y^5 !!

#compute the amplitudes
ps = np.arange(pmin, pmax+1)
Ps, E2s = np.meshgrid(ps, vs[1],indexing='ij')
start_soltime = time.time()
N20 = N20_Newtonian(Ps, E2s)
N22 = N22_Newtonian(Ps, E2s)
print("\nTime to compute the amplitudes: %s seconds \n" % (time.time() - start_soltime))

#compute stuff for Euler precession angles
ys, DJ2s, bpsips = RR_sol_interp(ts, idxs=[0, 4, 5])
start_soltime = time.time()
prec_Euler_constants = constants_precesion_Euler_angles(ys, DJ2s, m1, m2, sz_1, sz_2, sp2_1, sp2_2)
print("\nconstants_precesion_Euler_angles runtime: %s seconds \n" % (time.time() - start_soltime))

start_soltime = time.time()
dphiz, dzeta, costhL = precesion_Euler_angles(bpsips, *prec_Euler_constants)
print("\nprecesion_Euler_angles runtime: %s seconds \n" % (time.time() - start_soltime))

#test new and old ways to compute basic precession quantities
prec_new = basic_prec_quantities(vs[0], vs[4], m1, m2, sz_1, sz_2, sp2_1, sp2_2, only_for_Dv=False)
#prec_old = basic_prec_quantities_v1(vs[0], vs[4], m1, m2, sz_1, sz_2, sp2_1 + sz_1*sz_1 , sp2_2 + sz_2*sz_2, only_for_Dv=False)
prec_old = basic_prec_quantities_v2(vs[0], vs[4], m1, m2, sz_1, sz_2, sp2_1, sp2_2, only_for_Dv=False)
prec_labels = ['m', 'dchi_av', 'dchi_diff', 'chi_eff', 'J', 'L', 'sqY3mYm', 'Pp', 'Pm', 'PI_fact_p', 'PI_fact_m', 'PI_arg_p', 'PI_arg_m']

print('Comparison between new and old ways to compute precession averaged quantities')
for iprec, label in enumerate(prec_labels):
	print(label, np.linalg.norm(prec_new[iprec]-prec_old[iprec])/np.linalg.norm(prec_new[iprec]+prec_old[iprec]))

#make a plot of the errors in the derivatives
plt.figure(figsize=(16,10))
plt.plot(ts_coal_M, np.transpose(np.abs(delta_dv_dt)), label=[r'$y$', r'$e^2$', r'$\lambda$', r'$\delta_\lambda$', r'$J$', r'$\overline{\psi}_p$', r'$\phi_{z,0}$', r'$\zeta_0$'])
plt.plot(ts_coal_M, np.abs((ddlambda_interp - ddlambda_PN)/ddlambda_PN), label=r'$d\lambda/dt$')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('$(t_c^\mathrm{LO} - t)/M$')
plt.ylabel('Interpolation error in different derivatives')
plt.xlim(ts_coal_M[-1], ts_coal_M[0])
plt.grid(True)
plt.legend(ncol=3)
plt.tight_layout()

#make a plot comparing PN and interpolated derivatives of l and dl
plt.figure(figsize=(16,10))
plt.plot(ts_coal_M, dv_dt_interp[2], label=r'$d\lambda/dt$ interp')
plt.plot(ts_coal_M, dv_dt_PN[2], '--', label=r'$d\lambda/dt$ PN')
plt.plot(ts_coal_M, dv_dt_interp[3], label=r'$d\delta\lambda/dt$ interp')
plt.plot(ts_coal_M, dv_dt_PN[3], '--', label=r'$d\delta\lambda/dt$ PN')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('$(t_c^\mathrm{LO} - t)/M$')
plt.xlim(ts_coal_M[-1], ts_coal_M[0])
plt.grid(True)
plt.legend()
plt.tight_layout()

#make a plot comparing PN and interpolated derivatives of l and dl
plt.figure(figsize=(16,10))
plt.plot(ts_coal_M, ddlambda_interp, label=r'$d^2\lambda/dt^2$ interp')
plt.plot(ts_coal_M, ddlambda_PN, '--', label=r'$d^2\lambda/dt^2$ PN')
plt.plot(ts_coal_M, dddeltal_interp, label=r'$d^2\delta\lambda/dt^2$ interp')
plt.plot(ts_coal_M, dddeltal_PN_approx, '--', label=r'$d^2\delta\lambda/dt^2$ PN approx')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('$(t_c^\mathrm{LO} - t)/M$')
plt.xlim(ts_coal_M[-1], ts_coal_M[0])
plt.grid(True)
plt.legend()
plt.tight_layout()


#make a plot of the frequencies
n = 1
m = -6
plt.figure(figsize=(16,10))
tcoal_s = t_sun*ts_coal_M
for i_m, m in enumerate(range(-6,1)):
	freq_nm_PN = (M/t_sun)*(n*dv_dt_PN[2] + m*dv_dt_PN[3])/(2*np.pi)
	plt.plot(tcoal_s, freq_nm_PN, 'C%.f'%(i_m), label=r'(n,m)=(%s,%s)'%(n,m))

plt.xscale('log')
plt.yscale('log')
plt.xlabel('$(t_c^\mathrm{LO} - t)(M_\odot/M)$ [s]')
plt.ylabel(r'$M f = (M/M_\odot)(n \dot{\lambda} + m\dot{\delta\lambda})/(2 \pi)$ [Hz]')
plt.grid(True)
plt.xlim(tcoal_s[-1],tcoal_s[0])
plt.ylim(bottom=freq_nm_PN[0])
plt.legend()
plt.tight_layout()
plt.savefig('Plots/freq_nm.png')

#make a plot of the amplitudes
from matplotlib.collections import LineCollection
fig, ax = plt.subplots(figsize=(16,10))
N22_segments = [np.column_stack([ts_coal_M, np.abs(N22[i,:])]) for i in range(len(ps))]
line_segments = LineCollection(N22_segments, array=ps, cmap='jet')
ax.add_collection(line_segments)
fig.colorbar(line_segments, label=r'$p$', ax=ax)
ax.plot(ts_coal_M, np.sqrt(vs[1]), 'k--', label=r'$e$')
ax.set_xlabel(r'$(t_c^\mathrm{LO} - t)/M$')
ax.set_ylabel(r'$|N^{2 2}_p|$')
ax.set_xlim(ts_coal_M[-1], ts_coal_M[0])
ax.set_ylim(1e-3,2.2)
ax.set_yscale('log')
ax.set_xscale('log')
ax.legend()
plt.tight_layout()

#make a plot of stuff for Euler precession angles
labels = ['m', 'PI_fact_p', 'PI_fact_m', 'PI_arg_p', 'PI_arg_m', 'Pp_d', 'Pm_d', 'dzeta_E_fact', 'E_m', 'K_m', 'costhetaL_av', 'costhetaL_diff']
marker = ['o', '.', 'x', '+', 'v', '^', '<', '>', 's', 'd','o', 'x']

#make a plot of Euler precession angles constants
plt.figure(figsize=(16,10))
for il, label in enumerate(labels):
	plt.plot(tcoal_s, prec_Euler_constants[il], marker[il], label=label)

plt.xlabel('$(t_c^\mathrm{LO} - t)(M_\odot/M)$ [s]')
plt.xscale('log')
plt.legend()
plt.tight_layout()

#make a plot of Euler precession angles phases
ym3 = vs[0]**-3
plt.figure(figsize=(16,10))
plt.plot(ym3, dphiz, label=r'$\delta\phi_z$')
plt.plot(ym3, dzeta, label=r'$\delta\zeta$')
plt.plot(ym3, costhL, label=r'$\cos{\theta_L}$')
plt.xlabel('$y^{-3}$')
plt.legend()
plt.tight_layout()

#make also a plot of phiz and zeta
phiz0, zeta0 = RR_sol_interp(ts, idxs=[6,7])
phiz = dphiz + phiz0
zeta = dzeta + zeta0
plt.figure(figsize=(16,10))
plt.plot(ym3, phiz, 'C0-', label=r'$\phi_z$')
plt.plot(ym3, phiz0, 'C0--', label=r'$\phi_{z,0}$')
plt.plot(ym3, zeta, 'C1-', label=r'$\zeta$')
plt.plot(ym3, zeta0, 'C1--', label=r'$\zeta_0$')
plt.plot(ym3, zeta+phiz, 'C2-', label=r'$\zeta + \phi_z$')
plt.plot(ym3, vs[3], 'C3-', label=r'$\delta\lambda$')
plt.plot(ym3, 2*bpsips, 'C4-', label=r'$2\overline{\psi}_p$')
plt.xlabel('$y^{-3}$')
plt.legend()
plt.tight_layout()

#compute d\phi_z/dt and dzeta/dt normalized by y**-9
ym6 = vs[0]**-6
ym6_mean = 0.5*(ym6[:-1] + ym6[1:])
cthL_mean = 0.5*(costhL[:-1] + costhL[1:])
ym3_mean = 0.5*(ym3[:-1] + ym3[1:])
dphiz_dt = ym6_mean*np.diff(phiz)/np.diff(ts)
dzeta_dt = ym6_mean*np.diff(zeta)/np.diff(ts)
ddphiz_dt = ym6_mean*np.diff(dphiz)/np.diff(ts)
ddzeta_dt = ym6_mean*np.diff(dzeta)/np.diff(ts)
plt.figure(figsize=(16,10))
plt.plot(ym3_mean, dphiz_dt, '-C0', label=r'$\frac{1}{y^6} \frac{d\phi_z}{dt}$')
plt.plot(ym3_mean, ddphiz_dt, '--C0', label=r'$\frac{1}{y^6} \frac{d\delta\phi_z}{dt}$')
plt.plot(ym3_mean, dzeta_dt, '-C1', label=r'$\frac{1}{y^6} \frac{d\zeta}{dt}$')
plt.plot(ym3_mean, ddzeta_dt, '--C1', label=r'$\frac{1}{y^6} \frac{d\delta\zeta}{dt}$')
plt.plot(ym3_mean, cthL_mean*dphiz_dt + dzeta_dt, '-C2', label=r'$\frac{1}{y^6} \left(\cos{\theta_L} \frac{d\phi_z}{dt} + \frac{d\zeta}{dt}\right)$')
plt.plot(ym3_mean, cthL_mean*ddphiz_dt + ddzeta_dt, '--C2', label=r'$\frac{1}{y^6} \left(\cos{\theta_L} \frac{d\delta\phi_z}{dt} + \frac{d\delta\zeta}{dt}\right)$')
plt.xlabel('$y^{-3}$')
plt.legend(loc='upper right')
plt.tight_layout()

plt.figure(figsize=(16,10))
plot_labels =['m', 'Pm', 'Pp'] #['m', 'dchi_av', 'dchi_diff', 'sqY3mYm', 'Pp', 'Pm']#, 'PI_fact_p', 'PI_fact_m', 'PI_arg_p', 'PI_arg_m']
color=0
for iprec, label in enumerate(prec_labels):
	if label in plot_labels:
		plt.plot(vs[0], prec_new[iprec], '-C%s'%(color), alpha=0.5, label=label)
		plt.plot(vs[0], prec_old[iprec], '--C%s'%(color), alpha=0.5)
		color+=1
		
plt.xlabel(r'$y$')
plt.xscale('log')
plt.legend(loc='upper right')
plt.tight_layout()

#Runtime
print("\nRuntime: %s seconds" % (time.time() - start_runtime))

plt.show()

