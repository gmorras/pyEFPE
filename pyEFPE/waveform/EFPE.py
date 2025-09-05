import numpy as np
from pyEFPE.utils.utils import *
from pyEFPE.waveform.functions import *
from scipy.interpolate import CubicSpline

#universal constants
t_sun_s = 4.92549094831e-6  #GMsun/c**3 [s]
Mpc_s = 1.02927125054339e14 #Mpc/c in [s]

#Class to compute an eccentric and precessing waveform at 3PN order (2PN in spins). We mostly follow arxiv:2106.10291 and arXiv:1801.08542, except for the amplitudes, where we take into account the 0PN (Newtonian) modes non-perturbatively in eccentricity.
class pyEFPE:

	def __init__(self, parameters):
	
		"""
		Initialize the class with the given parameters.
		
		``parameters`` is a dictionary in which we expect to find the following
		
		Parameters

		-----------  Physical parameters for waveform generation -----------

		mass1 : float
			Mass of companion 1, in solar masses - Required
		mass2 : float
			Mass of companion 2, in solar masses - Required
		e_start : float
			Initial eccentricity (at f_orbital_start) - Default: 0
		spin1x : float
			x-component of dimensionless spin of companion 1 - Default: 0
		spin1y : float
			y-component of dimensionless spin of companion 1 - Default: 0
		spin1z : float
			z-component of dimensionless spin of companion 1 - Default: 0
		spin2x : float
			x-component of dimensionless spin of companion 2 - Default: 0
		spin2y : float
			y-component of dimensionless spin of companion 2 - Default: 0
		spin2z : float
			z-component of dimensionless spin of companion 2 - Default: 0
		q1 : float
	    		quadrupole parameter of companion 1 (see arXiv:1801.08542) - Default: 1
		q2 : float
	    		quadrupole parameter of companion 2 - Default: 1
		distance : float
			Distance to the source, in Mpc - Default: 100 Mpc
		inclination : float
			Angle between orbital angular momentum and vector from binary to observer (N).
			Measured in radians, between 0 and Pi. - Default: 0
		f22_start : float
			Starting waveform generation frequency of 22 mode, in Hz 
			We have f_orbital_start = 0.5*f22_start - Default: 20 Hz
		f22_end : float or None
			Maximum 22 mode frequency in Hz the orbital motion is computed up to.
			If it is None, we compute all the way to the ISCO. Otherwise, we have f_orbital_end = min(0.5*f22_end, f_ISCO) - Default: None
		phi_start : float
			Initial orbital phase, in radians - Default: 0
		mean_anomaly_start : float
			initial mean anomaly of quasi keplerian parametrization - Default: 0
			
		----------- Optional parameters for solving and computing stuff -----------
		pn_phase_order: int
			Twice the Post-Newtonian order to use in the non-spinning part of the phasing (i.e. pn_phase_order=n corresponds to the (n/2)PN order).
			pn_phase_order=-1 selects the maximum pn order available (=6) - Default: 6
		pn_spin_order: int
			Twice the Post-Newtonian order to use in the spinning part of the phasing (i.e. pn_spin_order=n corresponds to the (n/2)PN order).
			pn_spin_order=-1 selects the maximum pn order available (=6) - Default: 6
		Amplitude_tol: float
			Tolerance in time-domain amplitude. It will control how many modes are taken in the Newtonian amplitudes - Default: 1e-2
		Amplitude_pmax: float
			Maximum number of modes taken in the Newtonian amplitudes - Default: 100
		DJ2_tol : float
			Tolerance in J**2 estimate when setting initial conditions - Default: 1e-10
		RR_sol_rtol: float or array-like, shape (7,)
			Relative tolerance when solving differential equations.
			If array, relative tolerances on [y, e2, l, dl, DJ2, bpsip, phiz0, zeta0]
			- Default: [1e-10, 1e-10, 1e-12, 1e-12, 1e-10, 1e-12, 1e-12, 1e-12]
		RR_sol_atol: float or array-like, shape (7,)
			Relative absolute when solving differential equations.
			If array, absolute tolerances on [y, e2, l, dl, DJ2, bpsip, phiz0, zeta0]
			- Default: [1e-12, 1e-12,  1e-8,  1e-8, 1e-12,  1e-8,  1e-6,  1e-6]
		Series_Reversion_Order: int
			Order up to which we perform series reversion when computing interpolant of stationary times - Default: 5
		Interpolate_Amplitudes: bool
			Choose whether or not to interpolate the amplitudes to speed up waveform evaluation - Default: True
		Interp_points_per_prec_cycle: int
			Number of points per precession cycle used to interpolate non-secular part of Euler angles \delta\phi_z, \delta\zeta and \cos theta_L - Default: 40
		Extra_interp_points_amplitudes: int
			Number of extra points over the Runge-Kutta order used to interpolate a polynomial to each Newtonian amplitude at each interpolation segment - Default: 2
		SPA_Frequency_rtol: float
			Minimum accepted value of (f(t_SPA) - f)/f when solving for t_SPA - Default: 1e-12
		SUA_kmax: int
			Maximum order kmax to compute the constants a_{k,kmax} (Eq.(127-128) of arXiv:2106.10291) that appear in the Shifted Uniform Asymptotics (SUA) method. - Default: 3
		"""
		#dictionary with default params
		params = {'e_start': 0,
		          'spin1x': 0,
		          'spin1y': 0,
		          'spin1z': 0,
		          'spin2x': 0,
		          'spin2y': 0,
		          'spin2z': 0,
		          'q1': 1,
		          'q2': 1,
		          'distance': 100,
		          'inclination': 0,
		          'f22_start': 20,
		          'f22_end': None,
		          'phi_start': 0,
		          'mean_anomaly_start': 0,
		          'pn_phase_order': 6,
		          'pn_spin_order': 6,
		          'Amplitude_tol': 1e-3,
		          'Amplitude_pmax': 100,
		          'DJ2_tol': 1e-10,
		          'RR_sol_rtol': [1e-10, 1e-10, 1e-12, 1e-12, 1e-10, 1e-12, 1e-12, 1e-12],
		          'RR_sol_atol': [1e-12, 1e-12,  1e-8,  1e-8, 1e-12,  1e-8,  1e-6,  1e-6],
		          'Series_Reversion_Order': 5,
		          'Interpolate_Amplitudes': True,
		          'Interp_points_per_prec_cycle': 40,
		          'Extra_interp_points_amplitudes': 2,
		          'SPA_Frequency_rtol': 1e-12,
		          'SUA_kmax': 3,
		         }
		
		#update default parameters with input parameters
		params.update(parameters)
		
		#save parameters
		self.params = params
		
		#extract things from the parameter dictionary
		#we convert the masses from Msun to seconds
		self.m1 = t_sun_s*params['mass1']
		self.m2 = t_sun_s*params['mass2']
		self.e0 = params['e_start']
		#Extract the dimensionless component spins (S_i/mu_i^2 = s_i/mu_i)
		self.spin0_1 = np.array([params['spin1x'], params['spin1y'], params['spin1z']])
		self.spin0_2 = np.array([params['spin2x'], params['spin2y'], params['spin2z']])
		self.q1 = params['q1']
		self.q2 = params['q2']
		#convert the input luminosity distance from Mpc to s
		self.dL = Mpc_s*params['distance']
		self.inclination = params['inclination']
		#obtain initial and final orbital frequencies from initial and final 22 mode GW frequencies (f_orb = 0.5*f22)
		self.f0_orb = 0.5*params['f22_start']
		if params['f22_end'] is None: self.ff_orb = None
		else: self.ff_orb = 0.5*params['f22_end']
		self.phi0 = params['phi_start']
		self.phi_e0 = params['mean_anomaly_start']
		
		#if m2>m1, flip everything
		if self.m2>self.m1:
			self.m1, self.m2 = self.m2, self.m1
			self.q1, self.q2 = self.q2, self.q1
			self.spin0_1, self.spin0_2 = self.spin0_2, self.spin0_1
			self.phi0 = self.phi0 + np.pi #flip orbital phase
			self.phi_e0 = self.phi_e0 + np.pi #flip argument of periastron

		#compute initial conditions, in the following, v=[y, e2, l, dl, DJ2, bpsip, phiz0, zeta0]
		self.v_ini, self.yf, self.sz_1, self.sz_2, self.chi_eff, self.sp2_1, self.sp2_2, self.s2_1, self.s2_2, self.cos_theta_JN, self.phi_JN = initial_conditions_for_RR_eqs(self.f0_orb, self.ff_orb, self.e0, self.m1, self.m2, self.spin0_1, self.spin0_2, self.inclination, self.phi0, self.phi_e0, DJ2_tol=params['DJ2_tol'])
		
		#initialize class to compute PN derivatives
		self.PN_derivatives    = pyEFPE_PN_derivatives(self.m1, self.m2, self.chi_eff, self.s2_1, self.s2_2, self.q1, self.q2, pn_phase_order=params['pn_phase_order'], pn_spin_order=params['pn_spin_order'])
		
		#initialize spin -2 spherical harmonics _{-2}Y_{2,m'}
		self.m2_Y2mp           = compute_m2_Y2(self.cos_theta_JN, self.phi_JN)

		#compute also (-1)^m' conj(Y^{l -m'}), which will be used to compute polarizations
		self.m2_Y2mp_mod       = np.conj(self.m2_Y2mp[::-1])
		self.m2_Y2mp_mod[1::2] = -self.m2_Y2mp_mod[1::2]
		
		#compute 0.5*[1,-1j]*(_{-2}Y_{2,m'} \pm (-1)^m' conj(Y^{l -m'})) to proyect h^{[0,2] m'} GW modes into [hp, hc] polarizations
		Ap_proj       =  0.5*(self.m2_Y2mp  + self.m2_Y2mp_mod)
		Ac_proj       = -0.5j*(self.m2_Y2mp - self.m2_Y2mp_mod) #change the sign of hc to go to LAL convention
		self.Apc_proj = np.transpose([Ap_proj, Ac_proj])
		
		#Compute constant part of h0 from Eq.(20) of 2402.06804, i.e. we want to compute  4\sqrt{\pi/5} M \nu/d_L
		self.h0_pref  = compute_h0_pref(self.m1, self.m2, self.dL)
		
		#compute the solution of v(t)=[y, e2, l, dl, DJ2, bpsip, phiz0, zeta0] from Eqs.(101-109) of 2106.10291
		self.sol      = solve_ivp_RR_eqs_t(self.v_ini, self.PN_derivatives, self.yf, self.m1, self.m2, self.sz_1, self.sz_2, self.sp2_1, self.sp2_2,
		                              rtol=params['RR_sol_rtol'], atol=params['RR_sol_atol'])
		
		#compute the SUA constants a_{k,k_max} by solving a system similar to Eq.(127-128) of arXiv:2106.10291
		self.ak_SUA = compute_ak_SUA(params['SUA_kmax'])

		#################### Computation of the necessary modes ####################
		
		#compute the necessary modes at each segment of the interpolant
		p_necessary = []
		mraw_necessary = []
		pmax = params['Amplitude_pmax']
		self.mode_interp_idx  = []
		#loop over initial squared eccentricity at each segment
		for i_interp, e2 in enumerate(self.sol.ys[0][:,1]):
			
			#obtain necessary modes [|m|,p] of N^{l m}_p that have to be taken into account (arXiv:2402.06804)
			mraw, p = np.transpose(Newtonian_orders_needed(max(e2,0), pmax, params['Amplitude_tol']))
			
			#update the pmax estimate, since e2 will be (generally) decreasing
			pmax = 1+np.amax(np.abs(p))
			
			#save |m| and p on lists
			p_necessary    += p.tolist()
			mraw_necessary += mraw.tolist()

			#save the RK interpolant each mode is in
			self.mode_interp_idx += np.full(len(p), i_interp, dtype=int).tolist()

		#convert stuff to numpy arrays
		p = np.array(p_necessary)
		mraw = np.array(mraw_necessary)
		self.mode_interp_idx  = np.array(self.mode_interp_idx, dtype=int)

		####################### Interpolation of the phases #######################

		#from p and m indexes, compute n=p+m (Eqs.(30-31) of arXiv:2402.06804)
		n = p + mraw
		
		#make sure that the phase n\lambda + (m - n)\delta\lambda is positive by doing [n,m]->[-n,-m]
		m = np.where(n>=0, mraw, -mraw)
		n = np.abs(n)
		
		#save the [m,p] used to characterize the phase
		self.necessary_modes = np.transpose([m,p])
			
		#compute the interpolant of the phase of the n-th mode: n\lambda + (m - n)\delta\lambda
		m_n = m - n

		#compute the interpolants of phase and the first two derivatives
		self.mode_phases_y0 = []
		self.mode_phases_Qs = []
		for der in range(3):
			#compute constant part for each mode
			self.mode_phases_y0.append(n*self.sol.ys[der][self.mode_interp_idx,2] + m_n*self.sol.ys[der][self.mode_interp_idx,3])
			#compute polynomial part for each mode
			self.mode_phases_Qs.append(n[:,np.newaxis]*self.sol.Qs[der][self.mode_interp_idx,2] + m_n[:,np.newaxis]*self.sol.Qs[der][self.mode_interp_idx,3])

		####################### Interpolation of the SPA times #######################
		
		#put the initial and final frequencies (\omega=2*pi*f) of the time interpolant in arrays
		self.ts_interp_w0 = self.mode_phases_y0[1]
		self.ts_interp_wf = self.ts_interp_w0 + np.sum(self.mode_phases_Qs[1], axis=-1)

		#compute the matrix of the interpolant for stationary time t(f)
		self.ts_interp_Q = series_reversion(self.mode_phases_Qs[1], order=params['Series_Reversion_Order'])

		################################# Amplitudes #################################
		
		#if amplitudes are going to be interpolated, set it up
		if params['Interpolate_Amplitudes']:
			#set up the interpolation of precesion amplitudes
			self.interpolate_Apc_prec()
			self.compute_Apc_prec = self.compute_Apc_prec_interpolated
			#set up the interpolation of Newtonian Fourier modes
			self.interpolate_N2m()
			self.compute_N2m = self.compute_N2m_interpolated
		#otherwise, use exact amplitudes
		else:
			self.compute_Apc_prec = self.compute_Apc_prec_exact
			self.compute_N2m = self.compute_N2m_exact

	#function to compute stationary times given an input array of frequencies (see Eq.(46) of arXiv:1801.08542)
	def stationary_times(self, freqs, rtol=1e-12):
		
		#compute the omega associated with these frequencies
		ws = 2*np.pi*freqs

		#check in which interpolant of the different modes these ws are in
		f_idxs, interp_idxs = sorted_vals_in_intervals(ws, self.ts_interp_w0, self.ts_interp_wf)
		
		#we will need the ws at the f_idxs
		ws = ws[f_idxs]

		#compute to which time interpolant each case corresponds
		idxs_t_interp = self.mode_interp_idx[interp_idxs]
		
		#compute the x = (t - t0)/h of the t interpolant
		dws = ws - self.ts_interp_w0[interp_idxs]
		xs = np.sum(np.transpose(self.ts_interp_Q[interp_idxs,:])*power_range(dws, self.ts_interp_Q.shape[1]), axis=0)
		
		#perform iterations of Newton-Rhapson until desired tolerance is reached
		idxs_update = np.ones_like(xs, dtype=bool)
		while np.any(idxs_update):
			
			#compute the approximate \omega and d\omega/dt
			px = power_range(xs[idxs_update], self.mode_phases_Qs[1].shape[1])
			interp_idxs_u = interp_idxs[idxs_update]
			dws_x = np.sum(np.transpose(self.mode_phases_Qs[1][interp_idxs_u,:])*px, axis=0)
			dw_dts = self.mode_phases_y0[2][interp_idxs_u] + np.sum(np.transpose(self.mode_phases_Qs[2][interp_idxs_u,:])*px[:-1], axis=0)

			#compute difference between target \omega and \omega(x)
			delta_ws = dws[idxs_update] - dws_x
			
			#compute the updated value of x, taking into account that d/dx = h d/dt
			xs[idxs_update] += (delta_ws/dw_dts)/self.sol.hs[idxs_t_interp[idxs_update]]

			#find the indexes that are above tolerance and have to be updated for next iteration
			idxs_update[idxs_update] = (np.abs(delta_ws) > np.abs(rtol*ws[idxs_update]))
			
		#now compute the corresponding stationary times
		t_SPA = self.sol.ts[idxs_t_interp] + self.sol.hs[idxs_t_interp]*xs
		
		#compute the stationary phase psi_SPA and SPA time scale T_SPA
		px = power_range(xs, self.mode_phases_Qs[0].shape[1])
		psi_SPA =  self.mode_phases_y0[0][interp_idxs] + np.sum(np.transpose(self.mode_phases_Qs[0][interp_idxs,:])*px     , axis=0)
		T_SPA   = (self.mode_phases_y0[2][interp_idxs] + np.sum(np.transpose(self.mode_phases_Qs[2][interp_idxs,:])*px[:-2], axis=0))**-0.5

		#take into account that \psi_s = 2 \pi f t_s - \phi(t_s) - pi/4
		psi_SPA = ws*t_SPA - psi_SPA - 0.25*np.pi

		#return all the SPA related things that will be needed in the future
		return f_idxs, interp_idxs, t_SPA, psi_SPA, T_SPA

	#function to compute exact Wigner D matrices D^2_{m',2} and D^2_{m', 0}
	def compute_D2_mp_exact(self, times):

		#compute the Euler angles and y, e at the input times
		y, e2, DJ2, bpsip, phiz, zeta = self.sol(times, idxs=[0,1,4,5,6,7])

		#if the perpendicular spins are 0, do not allow precession (this could be done deeper in the code)
		if (self.sp2_1==0) and (self.sp2_2==0):
			phiz, zeta, costhL = np.zeros_like(y), np.zeros_like(y), np.ones_like(y)
		#otherwise compute the full Euler angles
		else:
			dphiz, dzeta, costhL = precesion_Euler_angles(bpsip, y, DJ2, self.m1, self.m2, self.sz_1, self.sz_2, self.sp2_1, self.sp2_2)
			phiz += dphiz
			zeta += dzeta
		
		#compute the Wigner Matrices at the input times
		D2_mp2, D2_mp0 = compute_necessary_Wigner_D2(phiz, costhL, zeta, return_D2_mp0=True)

		#include the factor (M\omega)^{2/3} = (1-e^2)*y^2 in the Wigner Matrices
		omega_factor = ((1 - np.maximum(e2,0))*np.square(y))[:,np.newaxis]
		D2_mp2 = omega_factor*np.transpose(D2_mp2)
		D2_mp0 = omega_factor*np.transpose(D2_mp0)

		#return the Wigner matrices
		return D2_mp2, D2_mp0

	#method to interpolate the precession amplitudes \mathsf{A}^{+,\times}_{l,m} = h_0 \sum_{m'=-l}^{l} \mathsf{P}^{+,\times}_{l,m,m'}(\Theta, \Phi)  D^l_{m'm}(\phi_z,\theta_L,\zeta), where
	#\mathsf{P}^{+}_{l,m,m'} = \frac{1}{2}\left[{}_{-2}Y^{l m'} + (-1)^{l + m + m'} ({}_{-2}Y^{l -m'})^{*}\right]
	#\mathsf{P}^{\times}_{l,m,m'} = \frac{\rmi}{2}\left[ {}_{-2}Y^{l m'} - (-1)^{l + m + m'} ({}_{-2}Y^{l -m'})^{*}\right]
	def interpolate_Apc_prec(self, ):

		#compute the step in (bpsip+phiz) for precession amplitude interpolation (a precesion cycle is a change of pi in bpsip or in phiz).
		self.dEuler_interp = np.pi/(1 + self.params['Interp_points_per_prec_cycle'])

		#invert the (bpsip + |phiz|)(t) polynomial series (phiz can be decreasing when \vec{J}\cdot\vec{L}<0)
		Euler_Qs = self.sol.Qs[0][:,5] + np.sign(self.sol.Qs[0][:,6,[0]])*self.sol.Qs[0][:,6]
		t_dEuler_Qs = series_reversion(Euler_Qs, order=self.params['Series_Reversion_Order'])
		
		#compute how much (bpsip + phiz) varies in each interpolation region
		Delta_Euler = np.sum(Euler_Qs, axis=-1)

		#loop over interpolants and find how many points we need to put on each one. We always take the first point, to be as safe as possible.
		dEuler, idxs_dEuler = [], []
		for i_interp in range(len(self.sol.ts)):

			#compute the values of (psi + phiz) in which we are going to interpolate the precession amplitudes
			dEuler += np.arange(0, Delta_Euler[i_interp], self.dEuler_interp).tolist()
			
			#put the interpolant index of these angles also in an array
			idxs_dEuler += np.full(len(dEuler) - len(idxs_dEuler), i_interp, dtype=int).tolist()
		
		#compute the interpolation times corresponding to the (dbpsip + dphiz)
		prec_interp_xs = np.einsum('ij,ji->i',t_dEuler_Qs[idxs_dEuler], power_range(dEuler, t_dEuler_Qs.shape[-1]))
		#if x>=1, it is outside it's interpolation region and we neglect it
		idxs_xs_l_1 = (prec_interp_xs<1)
		idxs_dEuler = np.array(idxs_dEuler)[idxs_xs_l_1]
		#reconstruct time from x
		self.prec_interp_ts = self.sol.ts[idxs_dEuler] + self.sol.hs[idxs_dEuler]*prec_interp_xs[idxs_xs_l_1]

		#Add the final time to the interpolation times for precession
		self.prec_interp_ts = np.append(self.prec_interp_ts, self.sol.all_ts[-1])

		#take only the strictly increasing times
		self.prec_interp_ts = np.unique(self.prec_interp_ts)

		#compute the necessary Wigner matrices at the interpolation times for precession
		D2_mp2_interp, D2_mp0_interp = self.compute_D2_mp_exact(self.prec_interp_ts)

		#compute the precession amplitudes with m=0 and m=2, the ones with m=-2 can be obtained from \mathsf{A}^{+,\times}_{2,-m} = conj(\mathsf{A}^{+,\times}_{2, m})
		self.Apc_prec_2_interp = np.matmul(D2_mp2_interp, self.Apc_proj)
		self.Apc_prec_0_interp = np.matmul(D2_mp0_interp, self.Apc_proj)
		
		#compute corresponding cubic spline
		self.Apc_prec_2_cspline = CubicSpline(self.prec_interp_ts, self.Apc_prec_2_interp, axis=0)
		self.Apc_prec_0_cspline = CubicSpline(self.prec_interp_ts, self.Apc_prec_0_interp, axis=0)

	#function to compute the interpolated precession amplitudes
	def compute_Apc_prec_interpolated(self, times, m):
		
		#Initialize the precession amplitudes with zeros
		Apc_prec = np.zeros((len(times),2), dtype=np.complex128)
		
		#consider the m=0 case
		m0_idxs = (m==0)
		if np.any(m0_idxs): Apc_prec[m0_idxs] = self.Apc_prec_0_cspline(times[m0_idxs])
		
		#consider the |m|=2 case
		m2_idxs = (np.abs(m)==2)
		if np.any(m2_idxs): Apc_prec[m2_idxs] = self.Apc_prec_2_cspline(times[m2_idxs])
		
		#consider the m=-2 case
		mm2_idxs = (m==-2)
		#Use that A_{-m} = conj(A_{m})
		if np.any(mm2_idxs): Apc_prec[mm2_idxs] = np.conj(Apc_prec[mm2_idxs])

		return Apc_prec

	#function to compute the exact precession amplitudes
	def compute_Apc_prec_exact(self, times, m):
		
		#compute the exact Wigner D-matrices
		D2_mp2, D2_mp0 = self.compute_D2_mp_exact(times)
		
		#initialize the precession amplitudes
		Apc_prec = np.zeros((len(times), 2), dtype=np.complex128)
		
		#consider the m=0 case
		m0_idxs = (m==0)
		if np.any(m0_idxs): Apc_prec[m0_idxs] = np.matmul(D2_mp0[m0_idxs], self.Apc_proj)
		
		#consider the |m|=2 case
		m2_idxs = (np.abs(m)==2)
		if np.any(m2_idxs): Apc_prec[m2_idxs] = np.matmul(D2_mp2[m2_idxs], self.Apc_proj)
		
		#consider the m=-2 case
		mm2_idxs = (m==-2)
		#Use that A_{-m} = conj(A_{m})
		if np.any(mm2_idxs): Apc_prec[mm2_idxs] = np.conj(Apc_prec[mm2_idxs])

		return Apc_prec

	#function to compute the exact Newtonian Fourier Mode Amplitudes
	def compute_N2m_exact(self, times, interp_idxs):
		
		#compute e2 at the input times
		e2s = np.maximum(self.sol(times, idxs=1),0)
		
		#initialize the Fourier Mode Amplitudes
		N2ms = np.zeros_like(e2s)

		#compute the m's and p's of the amplitudes requested
		ms, ps = np.transpose(self.necessary_modes[interp_idxs])
		
		#consider the m=0 case
		m0_idxs = (ms==0)
		if np.any(m0_idxs): N2ms[m0_idxs] = N20_Newtonian(ps[m0_idxs], e2s[m0_idxs])
		
		#consider the |m|=2 case
		m2_idxs = (np.abs(ms)==2)
		if np.any(m2_idxs): N2ms[m2_idxs] = N22_Newtonian(ps[m2_idxs], e2s[m2_idxs])
		
		return N2ms
		
	#method to interpolate Newtonian Fourier Mode Amplitudes
	def interpolate_N2m(self,):

		#compute the initial and final times of the interpolants of the N2m's. We take into account that due to SUA timeshifts it will be called outside [t[i], t[i+1]]
		#we use fmin and fmax to avoid propagating NANs
		self.N2m_interp_ts = np.fmax(self.sol.all_ts[self.mode_interp_idx] - self.params['SUA_kmax']/np.sqrt(self.mode_phases_y0[2]), self.sol.all_ts[0])
		N2m_interp_tf = np.fmin(self.sol.all_ts[self.mode_interp_idx+1] + self.params['SUA_kmax']/np.sqrt(self.mode_phases_y0[2] + np.sum(self.mode_phases_Qs[2], axis=-1)), self.sol.all_ts[-1])
		
		#save also timestep of the interpolant
		self.N2m_interp_hs = N2m_interp_tf - self.N2m_interp_ts
		
		#compute the times at wich we will evaluate the interpolants for the Newtonian modes, we use Chebyshev nodes of the second kind
		N2m_interp_n_nodes = self.sol.Qs[0].shape[2] + 1 + max(self.params['Extra_interp_points_amplitudes'], 0)
		N2m_interp_xs = np.sin(0.5*np.pi*np.arange(N2m_interp_n_nodes)/(N2m_interp_n_nodes - 1))**2 #we want them in [0,1]
		N2m_interp_ts_eval = self.N2m_interp_ts[:,np.newaxis] + self.N2m_interp_hs[:,np.newaxis]*N2m_interp_xs[np.newaxis,:]

		#compute the exact Newtonian Fourier Mode Amplitudes
		N2m_interp_ts_eval_flat = N2m_interp_ts_eval.flatten()
		N2m_interp_idxs_flat = np.transpose(np.tile(np.arange(N2m_interp_ts_eval.shape[0]), (N2m_interp_ts_eval.shape[1], 1))).flatten()
		N2ms = self.compute_N2m_exact(N2m_interp_ts_eval_flat, N2m_interp_idxs_flat).reshape(N2m_interp_ts_eval.shape)
		
		#obtain the coefficients of the polynomials, choose the same order as the RK, and since modes are smooth functions of e2, we expect the error to be similar to RK error of e2
		self.N2m_interp_coefs = np.transpose(np.polyfit(N2m_interp_xs, np.transpose(N2ms), self.sol.Qs[0].shape[2]))
		
	#function to compute the interpolated newtonian amplitudes
	def compute_N2m_interpolated(self, times, interp_idxs):
		
		#compute the xs for polinomial interpolation
		x = (times - self.N2m_interp_ts[interp_idxs])/self.N2m_interp_hs[interp_idxs]
		
		#compute the polynomial coeficients
		coefs = np.transpose(self.N2m_interp_coefs[interp_idxs])
		
		#now compute the polynomial using Horners method
		N2m = coefs[0]
		for c in coefs[1:]:
			N2m = c + N2m*x
		
		return N2m

	#function to compute the amplitudes given an input array of times
	def compute_Amplitudes(self, times, m, interp_idxs):

		#compute the newtonian amplitudes
		N2m = self.compute_N2m(times, interp_idxs)

		#compute precession amplitudes
		Apc_prec = self.compute_Apc_prec(times, m)
		
		#return the mode amplitudes h2m
		return Apc_prec*N2m[:,np.newaxis]
	
	#function to compute the amplitudes using the SUA
	def SUA_Amplitudes(self, times, m, interp_idxs, T_SPA):
		
		#compute the times for which the stencil points are in range and therefore we can do SUA
		iSUA = ((times - self.params['SUA_kmax']*T_SPA) >=  self.sol.all_ts[0]) & ((times + self.params['SUA_kmax']*T_SPA) <=  self.sol.all_ts[-1])
		
		#initialize amplitudes
		Amps = np.zeros((len(times), 2), dtype=np.complex128)
		
		#compute the amplitudes for which no SUA is done
		if np.any(~iSUA):
			Amps[~iSUA] = self.compute_Amplitudes(times[~iSUA], m[~iSUA], interp_idxs[~iSUA])
		
		#loop over the SUA time shifts
		for k_SUA in np.arange(-self.params['SUA_kmax'], self.params['SUA_kmax']+1):
			#Compute the amplitudes at the stencil points A(tSPA + k*TSPA)
			Amps[iSUA] += self.ak_SUA[abs(k_SUA)]*self.compute_Amplitudes(times[iSUA] + k_SUA*T_SPA[iSUA], m[iSUA], interp_idxs[iSUA])
			
		#return the SUA'd amplitudes
		return Amps
		
	#function to compute the waveform h_{+,\times} polarizations given an input array of frequencies (see Eq.(46) of arXiv:1801.08542)
	def generate_waveform(self, freqs):
		"""
		Generate the array [hp, hc] with waveform polarizations for a given array of frequencies.
		
		Parameters:
		-----------
		freqs : array-like
			Array of frequencies at which to generate the waveform.

		Returns:
		--------
		waveform : ndarray
			The generated waveform, with shape (2, len(freqs)). The first row contains the plus polarization, while the second row contains the cross polarization of the waveform.
		
		Notes:
		------
		This function computes the waveform using the SPA method as described in Eq.(46c) and Eq.(46f) of arXiv:1801.08542.
		The waveform is computed by:
		1. Calculating the stationary times and related indices.
		2. Computing the SPA factor.
		3. Looking up the necessary modes for each stationary time.
		4. Calculating the SUA'd amplitudes.
		5. Multiplying the SPA factor and amplitudes to obtain the waveform at each stationary time.
		6. Summing the contributions for each frequency.
		7. Transposing and conjugating the result to match the standard Fourier transform definition.
		8. Multiplying by the h0 prefactor and the sqrt(2*pi) factor from the SPA.
		"""

		#first compute the SPA related things
		f_idxs, interp_idxs, t_SPA, psi_SPA, T_SPA = self.stationary_times(freqs, rtol=self.params['SPA_Frequency_rtol'])
		
		#compute SPA factor = T*exp(1j*\psi) (Eq.(46c) of 1801.08542)
		SPA_fact    = T_SPA * (np.cos(psi_SPA) + 1j*np.sin(psi_SPA))
		
		#compute the m modes each ot these SPA times corresponds to
		m           = self.necessary_modes[interp_idxs,0]
		
		#compute the SPA'd Amplitudes (Eq.(46f) of 1801.08542)
		Amps        = self.SUA_Amplitudes(t_SPA, m, interp_idxs, T_SPA)

		#now multiply both of them to obtain waveform at each stationary time
		waveform_ts = SPA_fact[:,np.newaxis]*Amps
		
		#initialize array to store result of adding terms corresponding to same frequency
		waveform    = np.zeros((len(freqs), 2), dtype=waveform_ts.dtype)
		
		#add amplitudes in place so that repeated indices will be accumulated
		np.add.at(waveform, f_idxs, waveform_ts)

		#transpose to have correct shape (2, frequencies) and conjugate to use standard fourier transform definition
		waveform = np.transpose(np.conj(waveform))

		#now return the result, multiplying by the h0 prefactor and the \sqrt{2\pi} coming from the SPA
		return (((2*np.pi)**0.5)*self.h0_pref)*waveform

	#Function to compute the time domain waveform polarizations h_{+,\times}(t) given an input array of times (or a time spacing)
	#When there are many Fourier modes, the computation could be done much more efficiently, since many things, such as v(t) or Apc_prec(t) are the same for all times
	#Similarly, we could compute phi(t) = n\lambda(t) + (m - n)\delta\lambda(t) without going throught the modes
	#However, for simplicity, we choose to do things as similarly as possible to the Fourier Domain computation
	def generate_tdomain_waveform(self, times=None, delta_t=None):
		
		#if no time-array is given, create it
		if times is None:
			if delta_t is None: raise ValueError("To compute time-domain waveform, please give either an array of times or a time spacing delta_t")
			#make an equaly spaced array for all available times
			times = np.arange(self.sol.all_ts[0], self.sol.all_ts[-1], delta_t)
			#by construction, all times are valid
			i_valid = np.ones_like(times, dtype=bool)
			valid_times = times.copy()
		else:
			#make sure it is a numpy array
			times = np.asarray(times)
			#find valid times (i.e. where the system has been simulated)
			i_valid = (times>=self.sol.all_ts[0]) & (times<=self.sol.all_ts[-1])
			valid_times = times[i_valid]
		
		#find the values of interp_idxs for each time
		t_idxs, interp_idxs = sorted_vals_in_intervals(valid_times, self.sol.all_ts[self.mode_interp_idx], self.sol.all_ts[self.mode_interp_idx+1])
		ts_compute = valid_times[t_idxs]
		
		#compute the m of each time and mode
		ms = self.necessary_modes[interp_idxs,0]

		#compute the Amplitudes
		Amps = self.compute_Amplitudes(valid_times[t_idxs], ms, interp_idxs)
		
		#find segment of Runge-Kutta each interp_idx is in
		idxs_t_interp = self.mode_interp_idx[interp_idxs]
		
		#compute the phases
		xs = (ts_compute - self.sol.ts[idxs_t_interp])/self.sol.hs[idxs_t_interp]
		phi_t = xs*self.mode_phases_Qs[0][interp_idxs,-1]
		for iQ in reversed(range(self.mode_phases_Qs[0].shape[1]-1)):
			phi_t = xs*(self.mode_phases_Qs[0][interp_idxs,iQ] + phi_t)
		phi_t+= self.mode_phases_y0[0][interp_idxs]

		#compute the waveform h(t) = 2*Re(A(t)*e^{-i\phi(t)})
		waveform_ts = 2*np.real(Amps*((np.cos(phi_t) - 1j*np.sin(phi_t))[:,None]))

		#initialize array to store result of adding terms corresponding to same time
		waveform = np.zeros((2, len(times)), dtype=waveform_ts.dtype)

		#add values of the waveform corresponding to the same time together
		#Since np.bincount only takes 1D weights, take opportunity to transpose to have correct shape (2, frequencies)
		for iw in range(len(waveform)):
			waveform[iw,i_valid] = np.bincount(t_idxs, weights=waveform_ts[:,iw], minlength=len(valid_times))

		#return polarizations, multiplying by h0 prefactor
		return self.h0_pref*waveform
