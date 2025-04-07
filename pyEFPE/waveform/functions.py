import numpy as np
import scipy.special
from scipy.integrate import solve_ivp

from pyEFPE.utils.utils import *
#import my polynomial class with cython
from pyEFPE.utils.cython_utils import my_cpoly

#################################################################

#When not specified, equations come from 2106.10291
#For units we will always assume that [t] = [1/f] = [M]

#################################################################

#function to compute initial conditions in v=[y, e2, l, dl, DJ2, bpsip, phiz0, zeta0]
def initial_conditions_for_RR_eqs(f0_orb, ff_orb, e0, m1, m2, spin0_1, spin0_2, inclination, phi0, phi_e0, DJ2_tol=1e-10):

	'''
	Function to compute initial conditions and initialize our waveform. We try to follow the conventions of LAL. 
	https://lscsoft.docs.ligo.org/lalsuite/lalsimulation/group__lalsimulation__inspiral.html
	The direction of observation N is on the z axis, and the orbital angular momentum is on the x-z plane. 
	The inclination is the angle between L and N. 
	The spins are in a frame such that sz_i = dot(s0_i, L).
	The spin components in the N-aligned-frame are the result of doing a rotation R_y(inclination) of the input spins.
	NOTE that in LAL the spin components transverse to the orbital angular momentum are in a triad in which the x-axis is parallel to the vector pointing from body 1 to body 2. This is a bit difficult for eccentric orbits -.-
	--------------------------------------------------
	
	f0_orb: float
	        initial orbital frequency
	ff_orb: float
	        final orbital frequency. If it is larger than ISCO frequency, it will be forced to be the ISCO frequency.
	e0: float
	    initial (time-)eccentricity e_t from Quasi-Keplerian polarization
	m1: float
	    primary mass, assumed units are [1/f]
	m2: float
	    secondary mass, assumed units are [1/f]. If m2>m1, all properties of the components will be flip
	spin0_1: numpy array, shape (3,)
	      dimensionless spin vector spin1=S1/mu1^2=s1/mu1 of the primary object in cartesian coordinates
	spin0_2: numpy array, shape (3,)
	      dimensionless spin vector spin2=S2/mu2^2=s2/mu2 of the secondary object in cartesian coordinates
	phi0: float
	      initial orbital phase
	phi_e0: float
	        initial mean anomaly \ell_0 of quasi keplerian parametrization
	inclination : float
		Angle between orbital angular momentum and vector from binary to observer (N).

	'''
	
	#compute mass related things
	M, mu1, mu2, nu, dmu = mass_params_from_m1_m2(m1, m2)

	#compute initial squared eccentricity
	e20 = e0*e0

	#compute initial and final y using Eq.(5)
	y0 = ((2*np.pi*M*f0_orb)**(1/3))/np.sqrt(1-e20)
	
	#if final frequency is None, integrate to the ISCO
	yf_ISCO = 6**-0.5
	if ff_orb is None:
		yf = yf_ISCO
	#Otherwise, get final y from final orbital frequency. Assume all eccentricity has been radiated. From Eq.(5): y = (M\omega)^{1/3}/(1-e2)^{1/2}
	else:
		yf = (2*np.pi*M*ff_orb)**(1/3)
		
		#check that we are not surpassing the ISCO
		if yf > yf_ISCO:
			print("WARNING: With input final orbital frequency, yf=%.3f and the ISCO is surpassed, setting yf=yISCO=%.3f"%(yf, yf_ISCO))
			yf = yf_ISCO
	
	#compute initial \lambda
	l0 = phi0
	
	#compute initial \delta\lambda from above Eq.(17a) of 1801.08542
	dl0 = phi0 - phi_e0

	#compute the reduced spins as defined in Eq.(7). Note that S = mu^2 spin
	s0_1, s0_2 = mu1*spin0_1, mu2*spin0_2
	
	#compute spin components parallel to L0 
	sz_1 = s0_1[2]
	sz_2 = s0_2[2]
	
	#compute norms of perpendicular part of spin vectors
	sp2_1 = np.sum(s0_1[:2]**2)
	sp2_2 = np.sum(s0_2[:2]**2)

	#compute also the full norms of the spin vectors
	s2_1 = sp2_1 + sz_1*sz_1
	s2_2 = sp2_2 + sz_2*sz_2

	#compute initial \delta\chi from Eq.(11)
	dchi0 = sz_1 - sz_2
	
	#compute rotation matrix that moves the input spins to the physical frame, this is just of R_y(inclination), which would move z to L
	cinc, sinc = np.cos(inclination), np.sin(inclination)
	R_spin = np.array([[ cinc,   0, sinc],
	                   [    0,   1,    0],
	                   [-sinc,   0, cinc]])
	
	#compute the rotated spins
	s0p_1 = np.dot(R_spin, s0_1)
	s0p_2 = np.dot(R_spin, s0_2)

	#compute the initial Newtonian Angular Momentum, Eq.(8)
	L0 = nu/y0
	L0_hat = np.array([sinc, 0, cinc])
	L0_vec = L0*L0_hat

	#compute total angular momentum vector in physical frame
	J_vec = L0_vec + mu1*s0p_1 + mu2*s0p_2

	#compute rotation matrix that moves J to z axis
	sthJ, cthJ, sphJ, cphJ = spherical_coords_from_vec(J_vec, return_trigonometric=True)
	R_J_to_z = np.array([[cthJ*cphJ, cthJ*sphJ, -sthJ],
	                     [    -sphJ,      cphJ,     0],
	                     [sthJ*cphJ, sthJ*sphJ,  cthJ]])
	
	#compute orbital angular momentum in this frame
	L0_hat_J_aligned = np.dot(R_J_to_z, L0_hat)
	
	#compute rotation matrix that moves this rotated L vector to the x-z plane, and thus to the J-frame
	sthLJ, cthLJ, sphLJ, cphLJ = spherical_coords_from_vec(L0_hat_J_aligned, return_trigonometric=True)
	R_L0p_to_xz = np.array([[ cphLJ, sphLJ, 0],
	                        [-sphLJ, cphLJ, 0],
	                        [     0,     0, 1]])

	#compute unit vector pointing from the observer to the binary (in binary frame)
	k_hat = -np.array([0,0,1])
	
	#compute the result of applying the two rotation matrices to this unit vector
	k_hat_Jframe = np.dot(R_L0p_to_xz, np.dot(R_J_to_z, k_hat))
	
	#compute the corresponding spherical coordinate angles
	cos_theta_JN, phi_JN = spherical_coords_from_vec(k_hat_Jframe, return_trigonometric=False)
	
	#compute initial guess of DJ20 = (J0**2 - J02_no_DJ2)/nu, where J02_no_DJ2 = L**2 + 2*L*(mu1*sz_1 + mu2*sz_2) + 2*nu*sz_1*sz_2 + (mu1**2)*s2_1 + (mu2**2)*s2_2
	DJ20 = 2*(s0_1[0]*s0_2[0] + s0_1[1]*s0_2[1])

	#compute V as introduced in Eq.(138)
	Vol = np.dot(np.cross(L0_hat, s0p_1), s0p_2)
	
	#update this initial guess taking into account \delta J of Eq.(91)
	DJ20_OG = DJ20
	for iJ in range(10):
	
		#compute basic precesion averaged quantities
		m0, dchi_av, dchi_diff, chi_eff, _, _, sqY3mYm, _, _, _, _ = basic_prec_quantities(y0, DJ20, m1, m2, sz_1, sz_2, sp2_1, sp2_2, only_for_Dv=True)

		#compute the cos(2*am0) using Eq.(26), i.e. dchi = dchi_av - dchi_diff*cos(2*am0)
		if dchi_diff != 0: cos2am0 = max(-1, min(1, (dchi_av - dchi0)/dchi_diff))
		else:              cos2am0 = 1
		
		#compute initial am0 = am(phip0, m0) inverting Eq.(26)
		am0 = 0.5*np.arccos(cos2am0)

		#now obtain phip0 by using that psip0 = K(am0, m0)
		psip0 = scipy.special.ellipkinc(am0, m0)
		
		#compute the correct sign of \psi_p from Eq.(138) and Eq.(26) (i.e. use that V<0 => Ddchi<0 => \psi_p<0)
		if Vol<0: psip0, am0 = -psip0, -am0

		#now compute \delta DJ2 \approx 2 J0 \delta J/nu from Eq.(91). Use 2*dmu*dchi_diff/(m*sqY3mYm) = y*sqY3mYm. There is a factor of 2 missing in Eq.(91)
		dDJ2 = (4/3)*nu*sqY3mYm*((32 + 28*e20)/5)*(y0*y0/(1 - y0*chi_eff))*(scipy.special.ellipeinc(am0, m0) - scipy.special.ellipe(m0)*psip0/scipy.special.ellipk(m0))

		#compute new DJ20 using this \delta DJ2
		DJ20_new = DJ20_OG - dDJ2

		#if tolerance is reached, break the loop
		if abs(DJ20_new - DJ20) < DJ2_tol: break

		#update DJ2
		DJ20 = DJ20_new
	
	#compute initial \overline{\psi}_p from Eq.(95)
	bpsip0 = 0.5*np.pi*psip0/scipy.special.ellipk(m0)

	#compute \delta\phi_z and \delta\zeta from Eq.(49) and (52) respectively
	dphiz0, dzeta0, _ = precesion_Euler_angles(bpsip0, y0, DJ20, m1, m2, sz_1, sz_2, sp2_1, sp2_2)

	#set phiz(t0)=zeta(t0)=0, i.e., initially, L is set to the x'-z' plane of the J-frame
	phiz00 = -dphiz0
	zeta00 = -dzeta0

	#compute initial conditions on v=[y, e2, l, dl, DJ2, bpsip, phiz0, zeta0]
	v_ini = np.array([y0,e20,l0,dl0,DJ20,bpsip0,phiz00,zeta00])

	#return relevant parameters and initial conditions
	return v_ini, yf, sz_1, sz_2, chi_eff, sp2_1, sp2_2, s2_1, s2_2, cos_theta_JN, phi_JN

#Function to compute constant part of h0 from Eq.(20) of 2402.06804, i.e. we want to compute  4\sqrt{\pi/5} M \nu/d_L
# to obtain h0 we have to multiply by (M\omega)**(2/3) = ((1-e2)*(y**2)) (Eq.(5))
def compute_h0_pref(m1, m2, dL):
	return 4*((np.pi/5)**0.5)*(m1*m2/(m1+m2))/dL

#function to solve time diferential equations for v(t)=[y, e2, l, dl, DJ2, bpsip, phiz0, zeta0] 
#here J**2 = L**2 + 2*L*(mu1*sz_1 + mu2*sz_2) + 2*nu*sz_1*sz_2 + (mu1**2)*s2_1 + (mu2**2)*s2_2 + nu*DJ2
#use scipy solve_ivp and dense output
#terminate the integration when y reaches y_f
def solve_ivp_RR_eqs_t(v_ini, PN_derivatives, yf, m1, m2, sz_1, sz_2, sp2_1, sp2_2, rtol=[1e-10, 1e-10, 1e-12, 1e-12, 1e-10, 1e-12, 1e-12, 1e-12], atol=[1e-12, 1e-12,  1e-8,  1e-8, 1e-12,  1e-8,  1e-6,  1e-6]):
	
	#for the initial time give the 0PN estimate
	t0 = tLO_func(v_ini[0], v_ini[1], m1, m2)

	#function to compute dv/dt
	def dv_dt(t, v):		
		#return dv/dy
		return derivatives_prec_avg(v[0], max(v[1],0), v[4], PN_derivatives, m1, m2, sz_1, sz_2, sp2_1, sp2_2)

	#function to see if y is larger than the termination y=yf
	def yf_surpassed(t, v):
		return yf - v[0]

	#assign attribute to this function so that termination occurs if y>yf
	yf_surpassed.terminal = True

	#solve system of differential equations
	ivp_sol = solve_ivp(dv_dt, [t0, np.inf], v_ini, dense_output=True, method='RK45', events=yf_surpassed, rtol=rtol, atol=atol)
	
	#estimate the value of the final time from coalescence using final y and e2
	tf = tLO_func(ivp_sol.y[0,-1], ivp_sol.y[1,-1], m1, m2)
	
	#return an interpolant of the solution using our ivp_sol_interp class setting t=0 to be the coalescence time
	return ivp_sol_interp(ivp_sol.sol, t_final=tf)

#function to compute Newtonian amplitudes N20_p(e**2), as defined in 2402.06804
def N20_Newtonian(p, e2):
	return np.where(p==0, 0, ((2/3)**0.5)*scipy.special.jv(p,p*np.sqrt(e2)))

#function to compute Newtonian amplitudes N22_p(e**2), as defined in 2402.06804
def N22_Newtonian(p, e2):

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

#function to compute number of coefficients of Fourier series of Newtonian amplitudes N^{2m}_p needed for a given eccentricity
def Newtonian_orders_needed(e2, pmax, tol):

	#for 20 mode, compute 1 <= p <=pmax, since we will use N20_{-p} = N20_p
	p20 = np.arange(1,pmax+1)

	#for 22 mode compute -pmax <= p <=pmax
	p22 = np.arange(-pmax,pmax+1)
	
	#array with both order indexes and their correspondig labels
	ps = np.append(np.column_stack(( np.zeros_like(p20), p20)),
	               np.column_stack((2*np.ones_like(p22), p22)), axis=0)
	
	#compute fourier coeficients and put them together in array
	Njs = np.append(N20_Newtonian(p20, e2), N22_Newtonian(p22, e2))

	#compute also the norms i.e. \sum_p |N_{p}|^2
	norm = (4/3)*(-1 + 4/np.sqrt(1-e2))

	#compute the normalized amplitudes
	normed_Njs = (Njs**2)/norm
	
	#sort the elements from larger to smaller
	idxs_sort = np.argsort(-normed_Njs)
	normed_Njs_sorted = normed_Njs[idxs_sort]

	#compute cumulative norm	
	cum_norm = np.cumsum(normed_Njs_sorted, axis=0)
	
	#find how many orders have to be taken into account
	N_orders_needed = 1 + np.searchsorted(cum_norm, 1-tol, side='right')
	
	#return the indexes
	return ps[idxs_sort[:N_orders_needed]]

#solve the system of Eq.(127-128) of arXiv:2106.10291 for the SUA constants a_{k, k_\max}
#we move all factorials to the l.h.s. to have values closer to 1 in the Matrix of linear system
def compute_ak_SUA(kmax):

	#if kmax>9, there are problems with numerical precission, throw a warning
	if kmax>9: print('Warning: for SUA_kmax>9, finite precission errors when solving system of equations lead to untrustworthy values of a_{k,kmax}.')

	#rewritte it as a linear system M*a = b
	M = np.zeros((kmax+1, kmax+1), dtype=np.complex128)
	#b is 0.5 instead of 1 to take into account the 1/2 of Eq.(39) of 1408.5158
	b = np.full(kmax+1, 0.5)
	
	#compute k
	k=np.arange(kmax)+1
	
	#compute the matrix M_{q k} = ((i k^2)^q/(2q-1)!!)
	M[1:,1:] = np.cumprod(((1j*k*k)[None,:])/((2*k-1)[:,None]), axis=0)
	M[ 0, 0] = 0.5
	M[ 0,1:] = 1

	#now solve the system
	return np.linalg.solve(M, b)

#function to compute the derivatives appearing in Eqs.(101-109) of 2106.10291
def derivatives_prec_avg(y, e2, DJ2, PN_derivatives, m1, m2, sz_1, sz_2, sp2_1, sp2_2):

	#compute mass related stuff
	M, mu1, mu2, nu, dmu = mass_params_from_m1_m2(m1, m2)

	#compute the needed precesion averaged quantities
	J, L, chi_eff, K_m, sqY3mYm, Pp, Pm, dchi_prec_avg, dmudchi_prec_avg_m_dchi0, dchi2_prec_avg, sperp2_prec_avg = prec_avg_quantities_for_Dv(y, DJ2, m1, m2, sz_1, sz_2, sp2_1, sp2_2)
	
	#compute Dy, D(e^2), D\lambda and D\delta\lambda from Eqs.(101-104), using our class
	Dy, De2, Dl, Ddl = PN_derivatives.Dy_De2_Dl_Ddl(y, e2, dchi_prec_avg, dchi2_prec_avg, sperp2_prec_avg)
	#compute D\deltaJ^2, substituting J**2 = L**2 + 2*L*(mu1*sz_1 + mu2*sz_2) + 2*nu*sz_1*sz_2 + (mu1**2)*s2_1 + (mu2**2)*s2_2 + nu*DJ2 in Eq.(105)
	DDJ2 = -Dy*dmudchi_prec_avg_m_dchi0/(y*y)
	#compute stuff that will be needed for derivatives of Euler angles
	y6 = y**6
	A = 1 - y*chi_eff
	#compute D\overline{\psi}_p from Eq.(106)
	Dbpsip = 0.375*np.pi*A*y6*sqY3mYm/K_m
	#compute the prefactor multiplying the elliptic functions in Eqs.(107-108)
	P_pref = 0.75*A/(nu*K_m)
	#compute D\phi_{z,0} from Eq.(107)
	Dphiz0 = y6*(0.5*J + P_pref*(Pp + Pm))
	#compute D\zeta_0 from Eq.(108)
	Dzeta0 = y6*(-0.25*(2*L + chi_eff + dmu*dchi_prec_avg) - (1.5/nu)*(L + nu*chi_eff)*A + P_pref*(Pp - Pm))

	#compute D \equiv M/(1-e^2)^{3/2} d/dt (Eq.(4))
	D_fact = ((1 - e2)**1.5)/M
	
	#if y is an array, give the correct shape to D_fact
	if np.asarray(y).ndim!=0: D_fact = D_fact[np.newaxis,:]

	#return dvars_dt
	return D_fact*np.array([Dy, De2, Dl, Ddl, DDJ2, Dbpsip, Dphiz0, Dzeta0])

#function to compute basic precession averaged quantities of 2106.10291
def basic_prec_quantities(y, DJ2, m1, m2, sz_1, sz_2, sp2_1, sp2_2, only_for_Dv=True, min_2J1pmcthL=1e-10):

	#compute mass related stuff
	M, mu1, mu2, nu, dmu = mass_params_from_m1_m2(m1, m2)
	
	#compute chi_eff and dchi0
	chi_eff = sz_1 + sz_2
	dchi0 = sz_1 - sz_2

	#compute (y^2 q) and (y^3 q) from Eqs.(29,30). We have substituted the coefficients B, C and D from Eqs.(B1-B3) already
	#where we have substituted J**2 = L**2 + 2*L*(mu1*sz_1 + mu2*sz_2) + 2*nu*sz_1*sz_2 + (mu1**2)*s2_1 + (mu2**2)*s2_2 + nu*DJ2
	#compute stuff that will be used to simplyfy equations
	dmu2 = dmu*dmu
	j2a = (dmu2/y) + (chi_eff*chi_eff)*y #related to modulus of aligned angular momentum
	dChi = dmu*dchi0
	sp2_tot = sp2_1 + sp2_2 + DJ2 #total perpendicular spin |sp_1 + sp_2|^2
	bp = y*sp2_tot
	dp = y*dmu2*(4*sp2_1*sp2_2 - DJ2*DJ2)
	
	#compute part of p (Eq.(29)) that does not vanish in the aligned spin case
	pal = (j2a - 2*dChi + bp)/3
	pal2 = pal*pal
	#compute perpendicular part of p (i.e. that vanishes in the aligned spin case)
	pperp = bp*dChi - dmu*(DJ2*dmu + (sp2_1 - sp2_2)*y*chi_eff)
	pal_pal2pperp = pal*(pal2 + pperp)
	
	#compute y^6 times the cubic discriminant (p/3)**3 - (0.5*q)**2 of Eq.(33). Where py2 = 3*pal2 + 2*pperp and qy3 = -2*pal_pal2pperp + dp. We expand in terms of pal, pperp and dp to avoid numerical errors
	discy6 = (pperp*pperp)*(9*pal2 + 8*pperp)/27 + dp*(pal_pal2pperp - 0.25*dp)
	
	#compute arg(G)/3 from Eq.(33). The discriminant has to be larger than 0 since there are three real roots in cubic equation.
	argG_3 = my_where(discy6>=0, np.arctan2(discy6**0.5, pal_pal2pperp-0.5*dp)/3, 0)
	
	#compute the sine and cosine of arg(G)/3
	sargG_3, cargG_3 = np.sin(argG_3), np.cos(argG_3)
	
	#compute cos(argG/3 - pi/6)
	cargG_3_m_pi_6 = 0.5*((3**0.5)*cargG_3 + sargG_3)
	
	#compute \sqrt(y^2 p) = \sqrt{3*pal2 + 2*pperp} that appears in the Y's of Eqs.(31,32)
	py2 = 3*pal2 + 2*pperp
	sqpy2 = abs(py2)**0.5
	
	#compute dmu*(dchi_av - dchi0) and dmu*dchi_diff, where dchi_diff=(chi_+ - chi_-)/2 and dchi_av=(chi_+ + chi_-)/2, we write them in such a way that avoids numerical error
	dmudchiav_m_dchi0 = my_where((py2>0) & (pal>0), (pal2*sargG_3*sargG_3 - (2/3)*pperp*cargG_3*cargG_3)/(pal + (3**-0.5)*sqpy2*cargG_3), pal - (3**-0.5)*sqpy2*cargG_3)
	dmudchi_diff = sqpy2*sargG_3

	#compute also dchi_av and dchi_diff, since we have to divide by dmu, consider the case in which it is very small separately and use their taylor expansion there
	if dmu>1e-12:
		dchi_av = dchi0 + (dmudchiav_m_dchi0/dmu)
		dchi_diff = dmudchi_diff/dmu
	else:
		s2_tot = sp2_tot + chi_eff*chi_eff
		dchi_av = chi_eff*(sp2_1 - sp2_2 + dchi0*chi_eff)/s2_tot
		dchi_diff = np.sqrt(sp2_tot*(4*(sp2_2*sz_1*sz_1 + sp2_1*(sz_2*sz_2 + sp2_2))-DJ2*(4*sz_1*sz_2 + DJ2)))/s2_tot

	#compute m from Eq.(38)
	m = sargG_3/cargG_3_m_pi_6
	
	#compute sqrt(Y3 - Y_-), which is what actually appears in equations
	sqY3mYm = (2*sqpy2*cargG_3_m_pi_6/y)**0.5
	
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
	J_p_J0Lh = my_where(small_x & (J0Lh<0),-dJ_small_x, J + J0Lh)
	J_m_J0Lh = my_where(small_x & (J0Lh>0), dJ_small_x, J - J0Lh)

	#compute Np=N_+ and Nm=N_- from Eqs.(41-42)
	muSz = 2*(mu1*mu1*sz_1 + mu2*mu2*sz_2)
	dmudSp2 = dmu*(Sperp2_1-Sperp2_2)
	Np = J_p_J0Lh*(J_p_J0Lh - muSz)-dmudSp2
	Nm = J_m_J0Lh*(J_m_J0Lh + muSz)-dmudSp2

	#compute (Bp-Cp) = 2J*min(1+cos(\theta_L)) from Eqs.(26,43-46)
	Bp_m_Cp = 2*J_p_J0Lh + dmudchiav_m_dchi0 - dmudchi_diff
	#to avoid singularities force it to be larger than min_2J1pmcthL
	Bp_m_Cp = my_where(Bp_m_Cp>min_2J1pmcthL, Bp_m_Cp, min_2J1pmcthL)
	
	#compute (Bm+Cm) = 2J*min(1-cos(\theta_L)) from Eqs.(26,43-46)
	Bm_p_Cm = 2*J_m_J0Lh - dmudchiav_m_dchi0 - dmudchi_diff
	#to avoid singularities, force it to be larger than min_2J1pmcthL
	Bm_p_Cm = my_where(Bm_p_Cm>min_2J1pmcthL, Bm_p_Cm, min_2J1pmcthL)
	#now compute (Bm-Cm)=(Bm+Cm)-2*Cm
	Bm_m_Cm = Bm_p_Cm + 2*dmudchi_diff

	#compute prefactor's to elliptic PI's appearing in Eq.(109) of 2106.10291
	PI_fact_p = Np/Bp_m_Cp
	PI_fact_m = Nm/Bm_m_Cm

	#compute also the arguments -2*C/(B-C) using that Cp = -Cm = dmu*dchi_diff
	PI_arg_p = -2*dmudchi_diff/Bp_m_Cp
	PI_arg_m =  2*dmudchi_diff/Bm_m_Cm

	#compute Pp and Pm appearing in Eq.(109) of 2106.10291
	Pp = PI_fact_p*my_ellipPI(PI_arg_p, m)
	Pm = PI_fact_m*my_ellipPI(PI_arg_m, m)

	#choose whether to return extra stuff not needed to compute Dv
	if only_for_Dv:
		return m, dchi_av, dchi_diff, chi_eff, J, L, sqY3mYm, Pp, Pm, dmudchiav_m_dchi0, dmudchi_diff
	else:
		return m, dchi_av, dchi_diff, chi_eff, J, L, sqY3mYm, Pp, Pm, PI_fact_p, PI_fact_m, PI_arg_p, PI_arg_m

#function to compute the precesion average factors of Eq.(65) and Eq.(72) of 2106.10291 that appear in beta and sigma
def precesion_average_factors_betasigma(m, mthreas=0.3):
	
	#consider first the case that m is not an array
	if np.asarray(m).ndim == 0:
		if m<mthreas:
			return precesion_average_factors_betasigma_small_m(m)
		else:
			return precesion_average_factors_betasigma_large_m(m)
	#if m is a numpy array, the process is a bit more involved
	else:
		#initialize numpy arrays to put the result in
		m_factor_dchi_prec_avg, m_factor_sigma = np.zeros_like(m), np.zeros_like(m)
		
		#find indexes corresponding to small m's and put the corresponding results
		idxs_small = (m<mthreas)
		if np.any(idxs_small):
			m_factor_dchi_prec_avg[idxs_small], m_factor_sigma[idxs_small] = precesion_average_factors_betasigma_small_m(m[idxs_small])
		
		#do the same for the large m's
		idxs_large = np.logical_not(idxs_small)
		if np.any(idxs_large):
			m_factor_dchi_prec_avg[idxs_large], m_factor_sigma[idxs_large] = precesion_average_factors_betasigma_large_m(m[idxs_large])

		return m_factor_dchi_prec_avg, m_factor_sigma

#make a function for the large m case that uses the exact expressions of Eq.(65) and Eq.(72)
def precesion_average_factors_betasigma_large_m(m):

	#compute E(m)/K(m)
	E_m = scipy.special.ellipe(m)
	K_m = scipy.special.ellipk(m)
	E_K_m = E_m/K_m

	#compute the m factors as defined in Eq.(65) and Eq.(72) respectively
	m_factor_dchi_prec_avg = (E_K_m - 1 + 0.5*m)/m
	m_factor_sigma = ((1/3) + m*((-1/3) + m/8) + E_K_m*((2/3)*(m-2) + E_K_m))/(m*m)

	return m_factor_dchi_prec_avg, m_factor_sigma

#make a function for the small m case that uses Pade approximants of Eq.(65) and Eq.(72).
def precesion_average_factors_betasigma_small_m(m):

	#For the factor in Eq.(65) use a {3,3} Pade around m=0. We expect the absolute/relative error on this factor to be smaller than 1.3e-8/5.9e-7 for m<0.3
	m_factor_dchi_prec_avg = -m*(1 + m*(-1 + m*(71/384)))/(16 + m*(-24 + m*((59/6) - m*(11/12))))
	#For the factor in Eq.(72) use a {2,5} Pade around m=0. We expect the absolute/relative error on this factor to be smaller than 6.1e-10/4.9e-6 for m<0.3
	m2 = m*m
	m_factor_sigma = m2/(1024 + m*(-1024 + m*(96 - m2*(133/32)*(1+m))))

	return m_factor_dchi_prec_avg, m_factor_sigma

#function to compute the precession averages <dchi>, <dchi^2> and <sperp^2>
def compute_dchi_dchi2_sperp2_prec_avg(y, m, dchi_av, dchi_diff, dmudchiav_m_dchi0, dmudchi_diff, DJ2, sp2_1, sp2_2):
	
	#compute the m factors as defined in Eq.(65) and Eq.(72) respectively
	m_factor_dchi_prec_avg, m_factor_sigma = precesion_average_factors_betasigma(m)
	
	#compute dchi_prec_avg = <dchi> from Eq.(65)
	dchi_prec_avg = dchi_av - 2*dchi_diff*m_factor_dchi_prec_avg
	
	#compute also dmu*(dchi_prec_avg - dchi0)
	dmudchi_prec_avg_m_dchi0 = dmudchiav_m_dchi0 - 2*dmudchi_diff*m_factor_dchi_prec_avg

	#compute dchi2_prec_avg = <dchi^2>, note that in Eqs.(69-70) of 2106.10291 there is a typo in the sign of the m_factor_sigma
	dchi2_prec_avg = dchi_prec_avg*dchi_prec_avg + dchi_diff*dchi_diff*(0.5 - 4*m_factor_sigma)

	#compute <s_\perp^2> = <\sigma_0^(1)> - chi_eff2 using Eq.(71)
	sperp2_prec_avg = sp2_1 + sp2_2 + DJ2 - dmudchi_prec_avg_m_dchi0/y

	return dchi_prec_avg, dmudchi_prec_avg_m_dchi0, dchi2_prec_avg, sperp2_prec_avg

#function to compute the relevant precession averaged quantities of 2106.10291 needed for differential equations of Eqs.(101-109) of 2106.10291
def prec_avg_quantities_for_Dv(y, DJ2, m1, m2, sz_1, sz_2, sp2_1, sp2_2):

	#compute basic quantities coming from solving precession equation in 2106.10291
	m, dchi_av, dchi_diff, chi_eff, J, L, sqY3mYm, Pp, Pm, dmudchiav_m_dchi0, dmudchi_diff = basic_prec_quantities(y, DJ2, m1, m2, sz_1, sz_2, sp2_1, sp2_2, only_for_Dv=True)
	
	#compute dchi_prec_avg, betas and sigmas from 2106.10291
	dchi_prec_avg, dmudchi_prec_avg_m_dchi0, dchi2_prec_avg, sperp2_prec_avg = compute_dchi_dchi2_sperp2_prec_avg(y, m, dchi_av, dchi_diff, dmudchiav_m_dchi0, dmudchi_diff, DJ2, sp2_1, sp2_2)
	
	#compute also K(m)
	K_m = scipy.special.ellipk(m)
	
	#return the stuff needed for the derivatives
	return J, L, chi_eff, K_m, sqY3mYm, Pp, Pm, dchi_prec_avg, dmudchi_prec_avg_m_dchi0, dchi2_prec_avg, sperp2_prec_avg

#function to compute the variation of the Euler angles on precession time-scales following arXiv:2106.10291
def precesion_Euler_angles(bpsip, y, DJ2, m1, m2, sz_1, sz_2, sp2_1, sp2_2):

	#compute mass related stuff
	M, mu1, mu2, nu, dmu = mass_params_from_m1_m2(m1, m2)

	#compute basic quantities coming from solving precession equation in 2106.10291
	m, dchi_av, dchi_diff, chi_eff, J, L, sqY3mYm, Pp, Pm, PI_fact_p, PI_fact_m, PI_arg_p, PI_arg_m = basic_prec_quantities(y, DJ2, m1, m2, sz_1, sz_2, sp2_1, sp2_2, only_for_Dv=False)

	#compute also the prefactor of the first term in \delta\zeta of Eq.(52), we simplify
	#2*dmu*dchi_diff/(m*sqY3mYm) = y*sqY3mYm in C=2*dmu*dchi_diff/(3*m*(1-y*chi_eff)*sqY3mYm)
	dzeta_E_fact = y*sqY3mYm/(3*(1-y*chi_eff))
	
	#compute E(m) and K(m)
	E_m = scipy.special.ellipe(m)
	K_m = scipy.special.ellipk(m)
	
	#compute hbpsip_pi_2 = (2/pi)\hat{\overline{\psi}}_p (after Eq.49)
	hbpsip_pi_2 = np.mod((2/np.pi)*bpsip + 1, 2) - 1
	
	#compute the value of \hat{\psi}_p from \overline{\psi}_p using Eq.(95)
	hpsip = K_m*hbpsip_pi_2

	#compute the Jacobic elliptic functions. These correspond to sn and am of Eq.(A7)
	sn, cn, dn, am = scipy.special.ellipj(hpsip, m)

	#compute the incomplete elliptic integral of the second kind (Eq.(52))
	E_m_inc = dzeta_E_fact*(scipy.special.ellipeinc(am, m) - hbpsip_pi_2*E_m)

	#compute the incomplete elliptic integral of the third kind (Eq.(52))
	nusqY3mYm = nu*sqY3mYm
	ellipPI_p_inc = (PI_fact_p*my_ellipPI(PI_arg_p, m, phi=am) - hbpsip_pi_2*Pp)/nusqY3mYm
	ellipPI_m_inc = (PI_fact_m*my_ellipPI(PI_arg_m, m, phi=am) - hbpsip_pi_2*Pm)/nusqY3mYm
	
	#compute\delta\phi_z from Eq.(49)
	dphiz = ellipPI_p_inc + ellipPI_m_inc
	
	#compute \delta\zeta from Eq.(52)
	dzeta = E_m_inc + ellipPI_p_inc - ellipPI_m_inc

	#compute cos(\theta_L) from Eqs.(15, 26)
	dchi = dchi_av - dchi_diff*(1 - 2*sn*sn)
	costhL = (L + 0.5*(chi_eff + dmu*dchi))/J

	#return variation of Euler angles on precession timescales, forcing costhL to be between -1 and 1
	return dphiz, dzeta, np.maximum(np.minimum(costhL,1),-1)


#class to compute the precession averaged value of Dy, D(e^2), D\lambda and D\delta\lambda to 3PN in non-spining and aligned-spin and 2PN in fully spinning using the PN formulas derived in our EFPE paper. We can select the n-th PN order of the spinning/non-spinnig part with pn_phase_order/pn_spin_order = 2*n. If pn_xxxx_order=-1, set it to the maximum.
class pyEFPE_PN_derivatives:

	#initialize computing the constants that depend on nu
	def __init__(self, m1, m2, chi_eff, s2_1, s2_2, q1, q2, pn_phase_order=6, pn_spin_order=6):
		
		#save pn orders, taking into account that an order of -1 means to take the maximum order
		if pn_phase_order==-1: self.pn_phase_order = 6
		else:                  self.pn_phase_order = pn_phase_order

		if pn_spin_order==-1: self.pn_spin_order = 6
		else:                 self.pn_spin_order = pn_spin_order
		
		self.pn_max_order = max(self.pn_phase_order, self.pn_spin_order)
		
		#if we are requesting a pn order higher than what is implemented, throw a Warning
		if self.pn_phase_order>6: print('Warning: pn_phase_order>6 not implemented. Input pn_phase_order: %s'%(self.pn_phase_order))
		if self.pn_spin_order>6: print('Warning: pn_spin_order>6 not implemented. Input pn_spin_order: %s'%(self.pn_spin_order))
		
		#compute mass related stuff
		M, mu1, mu2, nu, dmu = mass_params_from_m1_m2(m1, m2)
		nu2 = nu*nu
		nu3 = nu2*nu
		pi2 = np.pi*np.pi
		self.nu = nu
		
		#compute symetric and antisymetric combinations of quadrupole parameters
		dqS = q1 + q2 - 2
		dqA = q1 - q2
		dqAdmu = dqA*dmu
		
		#compute spin related stuff
		s2iS =  s2_1 + s2_2
		s2iA =  s2_1 - s2_2
		chi_eff2 = chi_eff*chi_eff
		
		#store the e^{2n} coefficients that enter the tail terms of Eq (C4) of 1801.08542
		c_phiy  = np.array([1., 97./32., 49./128., -49./18432., -109./147456., -2567./58982400.])
		c_phie  = np.array([1., 5969./3940., 24217./189120., 623./4538880., -96811./363110400., -5971./4357324800.])
		c_psiy  = np.array([1., -207671./8318., -8382869./266176., -8437609./4791168., 10075915./306634752., -38077159./15331737600.])
		c_zetay = np.array([1., 113002./11907., 6035543./762048., 253177./571536., -850489./877879296., -1888651./10973491200.])
		c_psie  = np.array([1., -9904271./891056., -101704075./10692672., -217413779./513248256., 35703577./6843310080., -3311197679./9854366515200.])
		c_zetae = np.array([1., 11228233./2440576., 37095275./14643456., 151238443./1405771776., -118111./611205120., -407523451./26990818099300.])
		c_kappay = 244*np.log(2.)*np.array([0., 1., -18881./1098., 6159821./39528., -16811095./19764., 446132351./123525.])-243*np.log(3.)*np.array([0., 1., -39./4., 2735./64., 25959./512., -638032239./409600.])-(48828125*np.log(5.)/5184.)*np.array([0., 0., 0., 1., -83./8., 12637./256.]) -(4747561509943.*np.log(7.)/33177600.)*np.array([0., 0., 0., 0., 0., 1.])
		c_kappae = 6536*np.log(2.)*np.array([1., -22314./817., 7170067./19608., -10943033./4128., 230370959./15480., -866124466133./8823600.])-6561*np.log(3.)*np.array([1., -49./4., 4369./64., 214449./512., -623830739./81920., 76513915569./1638400.])-(48828125.*np.log(5.)/64.)*np.array([0., 0., 1., -293./24., 159007./2304., -6631171./27648.])-(4747561509943.*np.log(7.)/245760.)*np.array([0.,0.,0.,0.,1.,-259./20.])

		#store also the e^{2n} coefficients in the Spin-Orbit tail-terms
		c_thyc = np.array([1., 21263./3008., 52387./12032., 253973./1732608., -82103./13860864.])
		c_thyd = np.array([1., 1897./592., -461./2368., -42581./340992., -3803./1363968.])
		c_thec = np.array([1., 377077./92444., 7978379./4437312., 5258749./106495488.])
		c_thed = np.array([1., 37477./19748., 95561./947904., -631523./22749696.])

		#now store the e^{2n} coefficients appearing in the non-spinning part of dy/dt
		self.p_a0NS = my_cpoly(np.array([32./5., 28./5.]))
		
		self.p_a2NS = my_cpoly(np.array([-1486./105. - (88./5.)*nu, 12296./105. - (5258./45.)*nu, 3007./84. - (244./9.)*nu]))
		
		self.p_a3NS = my_cpoly((128./5.)*np.pi*c_phiy)
		
		self.p_a4NS = my_cpoly(np.array([34103./2835. + (13661./315.)*nu + (944./45.)*nu2, -489191./1890. - (209729./630.)*nu + (147443./270.)*nu2, 2098919./7560. - (2928257./2520.)*nu + (34679./45.)*nu2, 53881./2520. - (7357./90.)*nu + (9392./135.)*nu2]))
		self.sqrt_a4NS = my_cpoly(np.array([16. - (32./5.)*nu, 266. - (532./5.)*nu, -859./2. + (859./5.)*nu, -65. + 26*nu]))
		
		self.p_a5NS = my_cpoly(np.pi*(-(4159./105.)*c_psiy-(756./5.)*nu*c_zetay))
		
		self.p_a6NS = my_cpoly(np.array([16447322263./21829500. - (54784./525.)*np.euler_gamma + (512./15.)*pi2 + (-(56198689./34020.) + (902./15.)*pi2)*nu + (541./140.)*nu2 - (1121./81.)*nu3, 33232226053./10914750. - (392048./525.)*np.euler_gamma + (3664./15.)*pi2 + (-(588778./1701.) + (2747./40.)*pi2)*nu - (846121./1260.)*nu2 - (392945./324.)*nu3, -227539553251./58212000. - (93304./175.)*np.euler_gamma + (872./5.)*pi2 + ((124929721./12960.) - (41287./960.)*pi2)*nu + (148514441./30240.)*nu2 - (2198212./405.)*nu3, -300856627./67375. - (4922./175.)*np.euler_gamma + (46./5.)*pi2 + ((1588607./432.) - (369./80.)*pi2)*nu + (12594313./3780.)*nu2 - (44338./15.)*nu3, -243511057./887040. + (4179523./15120.)*nu + (83701./3780.)*nu2 - (1876./15.)*nu3, 0.]) + (1284./175.)*c_kappay)
		self.sqrt_a6NS = my_cpoly(np.array([-616471./1575. + ((9874./315.)- (41./30.)*pi2)*nu + (632./15.)*nu2, 2385427./1050. + (-(274234./45.) + (4223./240.)*pi2)*nu + (70946./45.)*nu2, 8364697./4200. + ((1900517./630.) - (32267./960.)*pi2)*nu - (47443./90.)*nu2, -167385119./25200. + ((4272491./504.) - (123./160.)*pi2)*nu - (43607./18.)*nu2, -65279./168. + (510361./1260.)*nu - (5623./45.)*nu2]))
		self.log_a6NS = my_cpoly(np.array([54784./525., 392048./525., 93304./175., 4922./175.]))

		#now store the e^{2n} coefficients appearing in the non-spinning part of d(e^2)/dt
		self.p_b0NS = my_cpoly(np.array([608./15., 242./15.]))
		
		self.p_b2NS = my_cpoly(np.array([-1878./35. - (8168./45.)*nu, 59834./105. - (7753./15.)*nu, 13929./140. - (3328./45.)*nu]))
		
		self.p_b3NS = my_cpoly((788./3.)*np.pi*c_phie)
		
		self.p_b4NS = my_cpoly(np.array([-949877./945. + (18763./21.)*nu + (1504./5.)*nu2, -3082783./1260. - (988423./420.)*nu + (64433./20.)*nu2, 23289859./7560. - (13018711./2520.)*nu + (127411./45.)*nu2, 420727./1680. - (362071./1260.)*nu + (1642./9.)*nu2]))
		self.sqrt_b4NS = my_cpoly(np.array([2672./3. - (5344./15.)*nu, 2321. - (4642./5.)*nu, 565./3. - (226./3.)*nu]))
		
		self.p_b5NS = my_cpoly(np.pi*(-(55691./105.)*c_psie-(610144./315.)*nu*c_zetae))

		self.p_b6NS = my_cpoly(np.array([61669369961./4365900. - (2633056./1575.)*np.euler_gamma + (24608./45.)*pi2 + ((50099023./56700.) + (779./5.)*pi2)*nu - (4088921./1260.)*nu2 - (61001./243.)*nu3, 66319591307./21829500. - (9525568./1575.)*np.euler_gamma + (89024./45.)*pi2 + ((28141879./450.) - (139031./480.)*pi2)*nu - (21283907./1512.)*nu2 - (86910509./9720.)*nu3, -1149383987023./58212000. - (4588588./1575.)*np.euler_gamma + (42884./45.)*pi2 + ((11499615139./453600.) - (271871./960.)*pi2)*nu + (61093675./2016.)*nu2 - (2223241./90.)*nu3, 40262284807./4312000. - (20437./175.)*np.euler_gamma + (191./5.)*pi2 + (-(5028323./280.) - (6519./320.)*pi2)*nu + (24757667./1260.)*nu2 - (11792069./1215.)*nu3, 302322169./887040. - (1921387./5040.)*nu + (41179./108.)*nu2 - (386792./1215.)*nu3, 0.]) + (428./1575.)*c_kappae)
		self.sqrt_b6NS = my_cpoly(np.array([-22713049./7875. + (-(11053982./945.) + (8323./90.)*pi2)*nu + (108664./45.)*nu2, 178791374./7875. + (-(38295557./630.) + (94177./480.)*pi2)*nu + (681989./45.)*nu2, 5321445613./189000. + (-(26478311./756.) + (2501./1440.)*pi2)*nu + (450212./45.)*nu2, 186961./168. - (289691./252.)*nu + (3197./9.)*nu2]))
		self.one_m_sqrt_b6NS = 1460336./23625
		self.log_b6NS = my_cpoly(np.array([2633056./1575., 9525568./1575., 4588588./1575., 20437./175.]))
		
		#now store the e^{2n} coefficients appearing in the Spin-Orbit part of dy/dt
		self.chi_p_a3SO = my_cpoly(chi_eff*np.array([-752./15., -138., -611./30.]))
		self.dch_p_a3SO = my_cpoly(dmu*np.array([-152./15., -154./15., 17./30.]))
		
		self.chi_p_a5SO = my_cpoly(chi_eff*np.array([-5861./45. + (4004./15.)*nu, -968539./630. + (259643./135.)*nu, -4856917./2520. + (943721./540.)*nu, -64903./560. + (5081./45.)*nu]))
		self.chi_e2sqrt_a5SO = my_cpoly(chi_eff*np.array([-1416./5. + (1652./15.)*nu, 2469./5. - (5761./30.)*nu, 222./5. - (259./15.)*nu]))
		
		self.dch_p_a5SO = my_cpoly(dmu*np.array([-21611./315. + (632./15.)*nu, -55415./126. + (36239./135.)*nu, -72631./360. + (12151./108.)*nu, 909./560. - (143./45.)*nu]))
		self.dch_e2sqrt_a5SO = my_cpoly(dmu*np.array([-472./5. + (236./15.)*nu, 823./5. - (823./30.)*nu, 74./5. - (37./15.)*nu]))
		
		chi_p_a6SO_arr = -(3008./15.)*np.pi*chi_eff*c_thyc
		dch_p_a6SO_arr = -(592./15.)*np.pi*dmu*c_thyd

		#now store the e^{2n} coefficients appearing in the Spin-Orbit part of d(e^2)/dt
		self.chi_p_b3SO = my_cpoly(chi_eff*np.array([-3272./9., -26263./45., -812./15.]))
		self.dch_p_b3SO = my_cpoly(dmu*np.array([-3328./45., -1993./45., 23./15.]))
		
		self.chi_p_b5SO = my_cpoly(chi_eff*np.array([-13103./35. + (289208./135.)*nu, -548929./63. + (61355./6.)*nu, -6215453./840. + (1725437./270.)*nu, -87873./280. + (13177./45.)*nu]))
		self.chi_sqrt_b5SO = my_cpoly(chi_eff*np.array([-1184. + (4144./9.)*nu, -13854./5. + (16163./15.)*nu, -626./5. + (2191./45.)*nu]))
		
		self.dch_p_b5SO = my_cpoly(dmu*np.array([-32857./105. + (52916./135.)*nu, -1396159./630. + (126833./90.)*nu, -203999./280. + (56368./135.)*nu, 5681./1120. - (376./45.)*nu]))
		self.dch_sqrt_b5SO = my_cpoly(dmu*np.array([-1184./3. + (592./9.)*nu, -4618./5. + (2309./15.)*nu, -626./15. + (313./45.)*nu]))
		
		chi_p_b6SO_arr = -(92444./45.)*np.pi*chi_eff*c_thec
		dch_p_b6SO_arr = -(19748./45.)*np.pi*dmu*c_thed

		#now store the e^{2n} coefficients appearing in the precession averaged Spin-Spin part of dy/dt
		c_s2iS_a4SS = s2iS*np.array([8./5. - 8*dqS, 24./5. - (108./5.)*dqS, 3./5. - (63./20.)*dqS])
		c_s2iA_a4SS = dqA*s2iA*np.array([-8., -108./5., -63./20.])
		c_chi2_a4SS = chi_eff2*np.array([156./5. + 12*dqS, 84. + (162./5.)*dqS, 123./10. + (189./40.)*dqS])
		self.const_a4SS = my_cpoly(c_s2iS_a4SS + c_s2iA_a4SS + c_chi2_a4SS)
		self.sperp2_a4SS = my_cpoly(np.array([-84./5., -228./5., -33./5.]))
		self.chidch_a4SS = my_cpoly(chi_eff*dqA*np.array([24., 324./5., 189./20.]))
		self.dch2_a4SS = my_cpoly(np.array([-2./5. + 12*dqS, -6./5. + (162./5.)*dqS, -3./20. + (189./40.)*dqS]))
		
		chi2_p_a6SS_arr = chi_eff2*np.array([30596./105. + (2539./105.)*dqS + (443./30.)*dqAdmu +  (-(688./5.) - (172./5.)*dqS)*nu, 115078./45. + (21317./60.)*dqS + (3253./60.)*dqAdmu + (-(3962./3.) - (1981./6.)*dqS)*nu, 4476649./2520. + (133703./420.)*dqS + (481./48.)*dqAdmu + (-(53267./45.) - (53267./180.)*dqS)*nu, 17019./140. + (29831./1120.)*dqS + (29./160.)*dqAdmu + (-(1343./15.) - (1343./60.)*dqS)*nu, 0.])
		self.chi2_sqrt_a6SS = my_cpoly(chi_eff2*np.array([-(244./15.) - (52./15.)*dqS - (4./15.)*dqAdmu + (16./5. + (4./5.)*dqS)*nu, 6283./30. + (1339./30.)*dqS + (103./30.)*dqAdmu + (-(206./5.) - (103./10.)*dqS)*nu, -(48007./120.) - (10231./120.)*dqS - (787./120.)*dqAdmu + (787./10. + (787./40.)*dqS)*nu, -(183./20.) - (39./20.)*dqS - (3./20.)*dqAdmu + (9./5. + (9./20.)*dqS)*nu]))
		chidch_p_a6SS_arr = chi_eff*np.array([(3134./15. + (443./15.)*dqS)*dmu + (5078./105. - (344./5.)*nu)*dqA, (30421./45. + (3253./30.)*dqS)*dmu + (21317./30. - (1981./3.)*nu)*dqA, (-(111./5.) + (481./24.)*dqS)*dmu + (133703./210. - (53267./90.)*nu)*dqA, (-(149./40.) + (29./80.)*dqS)*dmu + (29831./560. - (1343./30.)*nu)*dqA, 0.])
		self.chidch_sqrt_a6SS = my_cpoly(chi_eff*np.array([(-(104./15.) - (8./15.)*dqS)*dmu + (-(104./15.) + (8./5.)*nu)*dqA, (1339./15. + (103./15.)*dqS)*dmu + (1339./15. - (103./5.)*nu)*dqA, (-(10231./60.) - (787./60.)*dqS)*dmu + (-(10231./60.) + (787./20.)*nu)*dqA, (-(39./10.) - (3./10.)*dqS)*dmu + (-(39./10.) + (9./10.)*nu)*dqA]))
		self.dch2_p_a6SS = my_cpoly(np.array([39./5. + (2539./105.)*dqS + (443./30.)*dqAdmu + (-(1163./15.) - (172./5.)*dqS)*nu, 659./15. + (21317./60.)*dqS + (3253./60.)*dqAdmu + (-(2399./15.) - (1981./6.)*dqS)*nu, 1769./90. + (133703./420.)*dqS + (481./48.)*dqAdmu + (2021./72. - (53267./180.)*dqS)*nu, 19./10. + (29831./1120.)*dqS + (29./160.)*dqAdmu + (-(3./10.) - (1343./60.)*dqS)*nu]))
		self.dch2_sqrt_a6SS = my_cpoly(np.array([-(4./15.) - (52./15.)*dqS - (4./15.)*dqAdmu + (32./15. + (4./5.)*dqS)*nu, 103./30. + (1339./30.)*dqS + (103./30.)*dqAdmu + (-(412./15.) - (103./10.)*dqS)*nu, -(787./120.) - (10231./120.)*dqS - (787./120.)*dqAdmu +  (787./15. + (787./40.)*dqS)*nu, -(3./20.) - (39./20.)*dqS - (3./20.)*dqAdmu + (6./5. + (9./20.)*dqS)*nu]))
		
		#now store the e^{2n} coefficients appearing in the precession averaged Spin-Spin part of d(e^2)/dt
		c_s2iS_b4SS = s2iS*np.array([-4./3., 34./3. - (938./15.)*dqS, 49./2. - (595./6.)*dqS, 9./4. - (37./4.)*dqS])
		c_s2iA_b4SS = dqA*s2iA*np.array([0., -938./15., -595./6., -37./4.])
		c_chi2_b4SS = chi_eff2*np.array([2./3., 3667./15. + (469./5.)*dqS, 4613./12. + (595./4.)*dqS, 287./8. + (111./8.)*dqS])
		self.const_b4SS = my_cpoly(c_s2iS_b4SS + c_s2iA_b4SS + c_chi2_b4SS)
		self.sperp2_b4SS = my_cpoly(np.array([2./3., -1961./15., -2527./12., -157./8.]))
		self.chidch_b4SS = my_cpoly(chi_eff*dqA*np.array([0., 938./5., 595./2., 111./4.]))
		self.dch2_b4SS = my_cpoly(np.array([2./3., 1./3. + (469./5.)*dqS, -13./4. + (595./4.)*dqS, -3./8. + (111./8.)*dqS]))
		
		chi2_p_b6SS_arr = chi_eff2*np.array([1468414./945. + (2852./105.)*dqS + (3461./30.)*dqAdmu + (-(57844./45.) - (14461./45.)*dqS)*nu, 47715853./3780. + (1464091./840.)*dqS + (11007./40.)*dqAdmu + (-(21865./3.) - (21865./12.)*dqS)*nu, 4255831./504. + (166844./105.)*dqS + (2941./48.)*dqAdmu + (-(222533./45.) - (222533./180.)*dqS)*nu, 414027./1120. + (365363./4480.)*dqS + (511./640.)*dqAdmu + (-(1287./5.) - (1287./20.)*dqS)*nu])
		self.chi2_sqrt_b6SS = my_cpoly(chi_eff2*np.array([49532./45. + (10556./45.)*dqS + (812./45.)*dqAdmu + (-(3248./15.) - (812./15.)*dqS)*nu, 140117./60. + (29861./60.)*dqS + (2297./60.)*dqAdmu + (-(2297./5.) - (2297./20.)*dqS)*nu, 3721./180. + (793./180.)*dqS + (61./180.)*dqAdmu + (-(61./15.) - (61./60.)*dqS)*nu]))
		chidch_p_b6SS_arr = chi_eff*np.array([(176426./135. + (3461./15.)*dqS)*dmu + (5704./105. - (28922./45.)*nu)*dqA, (387212./135. + (11007./20.)*dqS)*dmu + (1464091./420. - (21865./6.)*nu)*dqA, (2562./5. + (2941./24.)*dqS)*dmu + (333688./105. - (222533./90.)*nu)*dqA, (-(33./32.) + (511./320.)*dqS)*dmu + (365363./2240. - (1287./10.)*nu)*dqA])
		self.chidch_sqrt_b6SS = my_cpoly(chi_eff*np.array([(21112./45. + (1624./45.)*dqS)*dmu + (21112./45. - (1624./15.)*nu)*dqA, (29861./30. + (2297./30.)*dqS)*dmu + (29861./30. - (2297./10.)*nu)*dqA, (793./90. + (61./90.)*dqS)*dmu + (793./90. - (61./30.)*nu)*dqA]))
		self.dch2_p_b6SS = my_cpoly(np.array([8887./135. + (2852./105.)*dqS + (3461./30.)*dqAdmu + (-(13127./27.) - (14461./45.)*dqS)*nu, 161077./540. + (1464091./840.)*dqS + (11007./40.)*dqAdmu + (-(185723./270.) - (21865./12.)*dqS)*nu, 14827./90. + (166844./105.)*dqS + (2941./48.)*dqAdmu + (-(45373./360.) - (222533./180.)*dqS)*nu, 283./32. + (365363./4480.)*dqS + (511./640.)*dqAdmu + (-(117./20.) - (1287./20.)*dqS)*nu]))
		self.dch2_sqrt_b6SS = my_cpoly(np.array([812./45. + (10556./45.)*dqS + (812./45.)*dqAdmu + (-(6496./45.) - (812./15.)*dqS)*nu, 2297./60. + (29861./60.)*dqS + (2297./60.)*dqAdmu + (-(4594./15.) - (2297./20.)*dqS)*nu, 61./180. + (793./180.)*dqS + (61./180.)*dqAdmu + (-(122./45.) - (61./60.)*dqS)*nu]))
		
		#join the parts of a6 and b6 that do not depend on dchi
		self.const_p_a6SOSS = my_cpoly(chi_p_a6SO_arr + chi2_p_a6SS_arr)
		self.const_p_b6SOSS = my_cpoly(chi_p_b6SO_arr + chi2_p_b6SS_arr)
		
		#join the parts of a6 and b6 that are proportional to dchi
		self.dch_p_a6SOSS = my_cpoly(dch_p_a6SO_arr + chidch_p_a6SS_arr)
		self.dch_p_b6SOSS = my_cpoly(dch_p_b6SO_arr + chidch_p_b6SS_arr)
		
		#store the e^{2n} coefficients of the non-spinning part of the periastron precession k
		self.k0NS = 3
		
		self.p_k2NS = my_cpoly(np.array([27./2. - 7*nu, 51./4. - (13./2.)*nu]))
		
		self.p_k4NS = my_cpoly(np.array([105./2. + (-(625./4.) + (123./32.)*pi2)*nu + 7*nu2, 573./4. + (-(357./2.) + (123./128.)*pi2)*nu + 40*nu2, 39./2. - (55./4.)*nu + (65./8.)*nu2]))
		self.sqrt_k4NS = my_cpoly(np.array([15. - 6*nu, 30. - 12*nu]))
		
		#store the e^{2n} coefficients of the spin-orbit part of the periastron precession k
		self.chi_k1SO = -(7./2.)*chi_eff
		self.dch_k1SO = -(1./2.)*dmu
		
		self.chi_p_k3SO = my_cpoly(chi_eff*np.array([-26. + 8*nu, -(105./4.) + (49./4.)*nu]))
		self.dch_p_k3SO = my_cpoly(dmu*np.array([-8. + (1./2.)*nu, -(15./4.) + (7./4.)*nu]))
		
		#store the e^{2n} coefficients appearing in the precession averaged Spin-Spin part of the periastron precession k
		c_s2iS_k2SS = -(3./8.)*dqS*s2iS
		c_s2iA_k2SS = -(3./8.)*dqA*s2iA
		c_chi2_k2SS = (3./2. + (9./16.)*dqS)*chi_eff2
		self.const_k2SS = c_s2iS_k2SS + c_s2iA_k2SS + c_chi2_k2SS
		self.sperp2_k2SS = -(3./4.)
		self.chidch_k2SS = (9./8.)*dqA*chi_eff
		self.dch2_k2SS = (9./16.)*dqS
		
		self.chi2_p_k4SS = my_cpoly(chi_eff2*np.array([181./8. + (33./8.)*dqS + (3./4.)*dqAdmu + (-(5./2.) - (5./8.)*dqS)*nu, 369./16. + (75./16.)*dqS + (3./16.)*dqAdmu + (-(29./4.) - (29./16.)*dqS)*nu]))
		self.chidch_p_k4SS = my_cpoly(chi_eff*np.array([(43./4. + (3./2.)*dqS)*dmu + (33./4. - (5./4.)*nu)*dqA, (21./8. + (3./8.)*dqS)*dmu + (75./8. - (29./8.)*nu)*dqA]))
		self.dch2_p_k4SS = my_cpoly(np.array([1./8. + (33./8.)*dqS + (3./4.)*dqAdmu + (-(7./2.) - (5./8.)*dqS)*nu, -(3./16.) + (75./16.)*dqS + (3./16.)*dqAdmu - (29./16.)*dqS*nu]))

	#function to compute Dy and De^2
	def Dy_De2_Dl_Ddl(self, y, e2, dchi, dchi2, sperp2):

		#compute different things that will be needed
		sqrt1me2 = (1-e2)**0.5
		one_m_sqrt = 1 - sqrt1me2
		sqrt_a = (1 - sqrt1me2)/sqrt1me2
		e2sqrt = e2/sqrt1me2
		log_fact = np.log((1 + sqrt1me2)/(8*y*sqrt1me2*(1-e2)))
		y2 = y*y
		y8 = y**8

		#initialize the PN derivatives
		Dy  = 0
		De2 = 0
		k   = 0
		
		#add the required terms
		if self.pn_max_order>=6:
			if self.pn_phase_order>=6:
				Dy  += self.p_a6NS(e2) + sqrt_a*self.sqrt_a6NS(e2) + log_fact*self.log_a6NS(e2)
				De2 += e2*(self.p_b6NS(e2) + sqrt1me2*self.sqrt_b6NS(e2) + log_fact*self.log_b6NS(e2)) + one_m_sqrt*self.one_m_sqrt_b6NS
				k   += self.p_k4NS(e2) + sqrt1me2*self.sqrt_k4NS(e2)
			if self.pn_spin_order>=6:
				Dy  += self.const_p_a6SOSS(e2) + dchi*self.dch_p_a6SOSS(e2) + sqrt_a*self.chi2_sqrt_a6SS(e2) + dchi*sqrt_a*self.chidch_sqrt_a6SS(e2) + dchi2*(self.dch2_p_a6SS(e2) + sqrt_a*self.dch2_sqrt_a6SS(e2))
				De2 += e2*(self.const_p_b6SOSS(e2) + dchi*self.dch_p_b6SOSS(e2) + sqrt1me2*self.chi2_sqrt_b6SS(e2) + dchi*sqrt1me2*self.chidch_sqrt_b6SS(e2) + dchi2*(self.dch2_p_b6SS(e2) + sqrt1me2*self.dch2_sqrt_b6SS(e2)))
				k   += self.chi2_p_k4SS(e2) + dchi*self.chidch_p_k4SS(e2) + dchi2*self.dch2_p_k4SS(e2)
			Dy  *= y
			De2 *= y
			k   *= y
		if self.pn_max_order>=5:
			if self.pn_phase_order>=5:
				Dy  += self.p_a5NS(e2)
				De2 += e2*self.p_b5NS(e2)
			if self.pn_spin_order>=5:
				Dy  += self.chi_p_a5SO(e2) + e2sqrt*self.chi_e2sqrt_a5SO(e2) + dchi*(self.dch_p_a5SO(e2) + e2sqrt*self.dch_e2sqrt_a5SO(e2))
				De2 += e2*(self.chi_p_b5SO(e2) + sqrt1me2*self.chi_sqrt_b5SO(e2) + dchi*(self.dch_p_b5SO(e2) + sqrt1me2*self.dch_sqrt_b5SO(e2)))
				k   += self.chi_p_k3SO(e2) + dchi*self.dch_p_k3SO(e2)
			Dy  *= y
			De2 *= y
			k   *= y
		if self.pn_max_order>=4:
			if self.pn_phase_order>=4:
				Dy  += self.p_a4NS(e2) + sqrt_a*self.sqrt_a4NS(e2)
				De2 += e2*(self.p_b4NS(e2) + sqrt1me2*self.sqrt_b4NS(e2))
				k   += self.p_k2NS(e2)
			if self.pn_spin_order>=4:
				Dy  += self.const_a4SS(e2) + sperp2*self.sperp2_a4SS(e2) + dchi*self.chidch_a4SS(e2) + dchi2*self.dch2_a4SS(e2)
				De2 += self.const_b4SS(e2) + sperp2*self.sperp2_b4SS(e2) + dchi*self.chidch_b4SS(e2) + dchi2*self.dch2_b4SS(e2)
				k   += self.const_k2SS     + sperp2*self.sperp2_k2SS     + dchi*self.chidch_k2SS     + dchi2*self.dch2_k2SS
			Dy  *= y
			De2 *= y
			k   *= y
		if self.pn_max_order>=3:
			if self.pn_phase_order>=3:
				Dy  += self.p_a3NS(e2)
				De2 += e2*self.p_b3NS(e2)
			if self.pn_spin_order>=3:
				Dy  += self.chi_p_a3SO(e2) + dchi*self.dch_p_a3SO(e2)
				De2 += e2*(self.chi_p_b3SO(e2) + dchi*self.dch_p_b3SO(e2))
				k   += self.chi_k1SO + dchi*self.dch_k1SO
			Dy  *= y
			De2 *= y
			k   *= y
		if self.pn_phase_order>=2:
			Dy  = y2*(Dy  + self.p_a2NS(e2))
			De2 = y2*(De2 + e2*self.p_b2NS(e2))
			k   = y2*(k   + self.k0NS)
		
		#always take into account the 0PN terms of Dy and De2
		Dy = y*y8*self.nu*(self.p_a0NS(e2) + Dy)
		De2 = -y8*self.nu*(e2*self.p_b0NS(e2) + De2)

		#compute D\lambda from Eq.(103)
		Dl = y2*y

		#compute D\delta\lambda from Eq.(104)
		Ddl = k*Dl/(1+k)

		return Dy, De2, Dl, Ddl

#compute the series expansion of the tLO integral for x->0 (here x=e**2)
# F = (24/19)*x**(-24/19)*((1 + (121/304)*x)**(-3480/2299))*sqrt(1-x)*integral((x**(5/19))*((1 + (121/304)*x)**(1181/2299))*((1 - x)**(-3/2)))
def F_tLO_series_at_0(x):
	coefs = np.array([1.000000000000000, -0.1511627906976744, 0.2656836084021005, 0.007463780007501875, 0.08800790590714085, 0.03153077124184580, 0.04185392210761341, 0.02761371124737777, 0.02642686119635599, 0.02145414169943131, 0.01926563742403364, 0.01677217450181226, 0.01503898450331388, 0.01347003088516929, 0.01220348405026613, 0.01110346195121149, 0.01016749330223870, 0.009352447459389354, 0.008642114232972108, 0.008016709107501086, 0.007463457161231162,0.006970902964332086, 0.006530253812462707, 0.006134111124121483, 0.005776451398258840, 0.005452229768143720, 0.005157232928828976, 0.004887901117379935, 0.004641215172072290, 0.004414596725230578,0.004205833180162198, 0.004013015784562471, 0.003834490347048246,0.003668816871424952, 0.003514736646109113, 0.003371145103099870, 0.003237069368178693, 0.003111649567641214, 0.002994123194217794, 0.002883811964918853, 0.002780110725151045, 0.002682478039498533,0.002590428180410570, 0.002503524280447905, 0.002421372457429333,0.002343616756405161, 0.002269934780185545, 0.002200033902499887, 0.002133647975961232, 0.002070534461714599, 0.002010471919657125])
	return np.polyval(np.flip(coefs),x)

#compute the series expansion of the tLO integral for x->1 (here x=e**2)
def F_tLO_series_at_1(x):
	
	#we are going to approximate the integral of ((1 - 1/u**2)**(5/19))*(1 - (121/425)/u**2)**(1181/2299) - 1 as -f0 + sum_{n=1}^{nmax} cn*u**-(2*n-1)
	cns = np.array([0.40941176470588236, 0.02286320645905421, 0.008142925951557094, 0.004155878512401501, 0.0024847568765827364, 0.001633290511588348, 0.0011455395465032605, 0.000842577094447224, 0.0006427195446513564, 0.0005045715814217113, 0.00040544477831148924, 0.0003321116545345162, 0.0002764636962467965, 0.00023331895675436905, 0.0001992473575067662, 0.00017190919726595898, 0.00014966654751355843, 0.000131346469484909, 0.00011609207361015848, 0.00010326617525262309, 9.238740896587563e-05, 8.308691874613337e-05, 7.50784086790562e-05, 6.813705814151314e-05, 6.20844345948999e-05, 5.677753690123865e-05, 5.210072977646673e-05, 4.7959732146031274e-05, 4.427708468062032e-05, 4.0988696117937945e-05, 3.80411855904995e-05, 3.538981870077757e-05, 3.2996890965966614e-05, 3.083045152872919e-05, 2.886328796018351e-05, 2.707211306399751e-05, 2.543690918050699e-05, 2.3940396192787112e-05, 2.256759736012561e-05, 2.1305483020854193e-05, 2.014267666038337e-05, 1.906921121897629e-05, 1.8076326095558655e-05, 1.7156297290322297e-05, 1.6302294667351972e-05, 1.550826151746742e-05, 1.4768812541410176e-05, 1.407914711455732e-05])
	
	#evaluate polynomial
	u = 1-x
	return ((48/19)*((425/304)**(1181/2299)))*(1 - 1.4555165803216864*np.sqrt(u) + u*np.polyval(np.flip(cns),u))*(x**(-24/19))*((1 + (121/304)*x)**(-3480/2299)) 

#function to choose series at x=0 or at x=1 (here x=e**2)
def F_tLO_series(x, x_thr=0.4):
	
	#distinguish case in which x is an array or not
	if np.asarray(x).ndim==0:
		#if x small, use series expansion at 0, otherwise use series expansion at 1
		if x<x_thr: return F_tLO_series_at_0(x)
		else: return F_tLO_series_at_1(x)
		
	else:
		#if x small, use series expansion at 0, otherwise use series expansion at 1
		F = np.zeros_like(x)
		i_low = x<x_thr
		i_high = np.logical_not(i_low)
		if sum(i_low)>0:  F[i_low] = F_tLO_series_at_0(x[i_low])
		if sum(i_high)>0: F[i_high] = F_tLO_series_at_1(x[i_high])
	
		return F

#function to compute Newtonian time
def tLO_func(y, e2, m1, m2):

	#compute mass related stuff
	M, _, _, nu, _ = mass_params_from_m1_m2(m1, m2)

	#return the LO time
	return -(5/256)*(M/nu)*(y**-8)*((1-e2)**-0.5)*F_tLO_series(e2)

