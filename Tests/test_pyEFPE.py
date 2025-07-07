import pyEFPE
import numpy as np
import time
start_runtime = time.time()

#array with parameters
params = {'mass1': 1.824,
          'mass2': 0.739,
          'e_start': 0.7,
          'spin1x': -0.44,
          'spin1y': -0.26,
          'spin1z': 0.48,
          'spin2x': -0.31,
          'spin2y': 0.01,
          'spin2z': -0.84,
          'inclination': 1.57,
          'f22_start': 10,
          'Amplitude_tol': 1e-4,
          }

#stuff for frequency array generation
seglen = 256
flow = 20
fhigh = 4096

#initialize waveform class
start_soltime = time.time()
wf = pyEFPE.pyEFPE(params)
print("\nTime to initialize waveform class: %s seconds \n" % (time.time() - start_soltime))

#print waveform time-domain amplitude
print('Waveform duration: %.5gs'%(-wf.sol.ts[0]))
#print info on the necessary modes
interp_idx, mode_counts = np.unique(wf.mode_interp_idx, return_counts=True)
print('Necessary Modes: Total: %s modes in %s segments'%(len(wf.mode_interp_idx),len(interp_idx)))
print(mode_counts[np.argsort(interp_idx)])
print('Number of points to interpolate Wigner D-matrices:', len(wf.prec_interp_ts))

#compute the amplitudes at the interpolation times exactly
N_N2m_test = wf.sol.Qs[0].shape[2]+20
#compute the times at wich we will evaluate the interpolants for the Newtonian modes
N2m_interp_xs = np.linspace(0, 1, N_N2m_test)
N2m_interp_ts_eval = wf.N2m_interp_ts[:,np.newaxis] + wf.N2m_interp_hs[:,np.newaxis]*N2m_interp_xs[np.newaxis,:]

#compute also the interpolated ones
N2m_exact = np.zeros_like(N2m_interp_ts_eval)
N2m_interp = np.zeros_like(N2m_interp_ts_eval)
for i, ts_interp in enumerate(N2m_interp_ts_eval):
	N2m_exact[i] = wf.compute_N2m_exact(ts_interp, np.full(len(ts_interp), i))
	N2m_interp[i] = wf.compute_N2m_interpolated(ts_interp, np.full(len(ts_interp), i))

#analize error
N2m_abs_err = np.abs(N2m_exact - N2m_interp)
N2m_rel_err = np.abs(0.5*(N2m_exact - N2m_interp)/(N2m_exact + N2m_interp))
print('Absolute error in Newtonian mode interpolation: MSE: %.3g    max: %.3g'%(np.linalg.norm(N2m_abs_err/N2m_exact.size), np.amax(N2m_abs_err)))
print('Relative error in Newtonian mode interpolation: MSE: %.3g    max: %.3g'%(np.linalg.norm(N2m_rel_err/N2m_exact.size), np.amax(N2m_rel_err)))

#test series reversion
from pyEFPE.utils import *
Nx_test = 1001
orders = [1,2,3,4,5]

#compute test input (it goes between 0 and 1)
x = np.linspace(0, 1, Nx_test)

#extract matrices of all interpolants
Q = wf.mode_phases_Qs[1]

#compute value of interpolant
px = np.cumprod(np.tile(x, (Q.shape[1],1)), axis=0)
y = np.dot(Q, px)

#loop over orders
errors = np.zeros(len(orders))
for iorder, order in enumerate(orders):
	
	#compute inverse of interpolant
	invQ = series_reversion(Q, order=order)
		
	#compute its appoximate inverse (it should be equal to x)
	py = np.cumprod(np.tile(y, (invQ.shape[1],1,1)), axis=0)
	x_approx = np.einsum('ij,jil->il',invQ, py, optimize='greedy')
		
	#compute error in approximate index
	errors[iorder] = np.amax(np.abs(x[...,:] - x_approx)) #np.linalg.norm(x[...,:] - x_approx)/np.sqrt(np.prod(x_approx.shape))

print('\nSeries reversing errors for orders =', orders, '->', errors)

#test subroutine to compute stationary times
freqs = np.arange(flow, fhigh, 1/seglen)

start_soltime = time.time()
f_idxs, interp_idxs, t_stationary, phase_stationary, Tnm_stationary = wf.stationary_times(freqs)
print("\nTime to compute stationary times: %s seconds \n" % (time.time() - start_soltime))

#compute frequencies deduced from these stationary times
idxs_t_interp = wf.mode_interp_idx[interp_idxs]
x_stationary = (t_stationary - wf.sol.ts[idxs_t_interp])/wf.sol.hs[idxs_t_interp]
ws_approx = wf.ts_interp_w0[interp_idxs] + np.sum(np.transpose(wf.mode_phases_Qs[1][interp_idxs,:])*power_range(x_stationary, wf.mode_phases_Qs[1].shape[1]), axis=0)

print('Mean squared error in f(ts(f)):     %.3g Hz'%(np.linalg.norm((ws_approx/(2*np.pi) - freqs[f_idxs])/freqs[f_idxs])/np.sqrt(len(f_idxs))))
print('Maximum relative error in f(ts(f)): %.3g'%(np.amax(np.abs(ws_approx/(2*np.pi) - freqs[f_idxs])/freqs[f_idxs])))

#test Amplitude generation
m, p = np.transpose(wf.necessary_modes[interp_idxs])
start_soltime = time.time()
Apc_prec = wf.compute_Apc_prec(t_stationary, m)
print("\nTime to compute Apc_prec: %s seconds \n" % (time.time() - start_soltime))

#test amplitude generation
start_soltime = time.time()
Amps = wf.compute_Amplitudes(t_stationary, m, interp_idxs)
print("\nTime to compute Amplitudes: %s seconds \n" % (time.time() - start_soltime))

#test SUA amplitude generation
start_soltime = time.time()
SUA_Amps = wf.SUA_Amplitudes(t_stationary, m, interp_idxs, Tnm_stationary)
print("\nTime to compute SUA Amplitudes: %s seconds \n" % (time.time() - start_soltime))

#test polarization generation
start_soltime = time.time()
hp, hc = wf.generate_waveform(freqs)
print("\nTime to compute hp, hc: %s seconds \n" % (time.time() - start_soltime))

#compute the waveforms in the time domain
from scipy.fft import irfft, rfft
#compute the time array
delta_t = 1/(2*fhigh)
times = delta_t*np.arange(int(2*seglen*fhigh))
fft_len = int(2*seglen*fhigh)
#loop over polarizations
h_td = list()
for h in [hp, hc]:
	#put the low frequencies of h (they are 0)
	h_padded = np.zeros(int(seglen*fhigh), dtype=h.dtype)
	h_padded[int(seglen*flow):int(seglen*fhigh)] = h
	#perform the inverse FFT, taking into account that h is the FFT of a real signal
	h_td.append((fft_len/seglen)*irfft(h_padded, n=fft_len))

#compute time-domain waveform
times = times-times[-1]
start_soltime = time.time()
hp_td_direct, hc_td_direct = wf.generate_tdomain_waveform(times)
print("\nTime to compute time-domain waveform: %s seconds \n" % (time.time() - start_soltime))

#compute frequency-domain and filtered waveforms
h_fd = list()
h_filtered = list()
for h in [hp_td_direct, hc_td_direct]:
	#compute fourier domain waveform
	h_rfft = delta_t*rfft(h)
	#create frequency mask
	frequency_mask = np.zeros(len(h_rfft), dtype=bool)
	frequency_mask[int(seglen*flow):int(seglen*fhigh)] = True
	#save it without low and high frequencies
	h_fd.append(h_rfft[frequency_mask])
	#make zero the values of rfft outside frequency range
	h_rfft[~frequency_mask] = 0
	#do the inverse fft
	h_filtered.append((fft_len/seglen)*irfft(h_rfft, n=fft_len))


from matplotlib import pyplot as plt
plt.rcParams.update({'font.size': 24})
plt.rcParams.update({'lines.linewidth': 2})

#approximated
Nt_plot = 100000
times_plot = -np.geomspace(-wf.sol.all_ts[0], -wf.sol.all_ts[-1], Nt_plot)
bpsip_plot, phiz_plot = wf.sol(times_plot, idxs=[5,6])
x_plot = bpsip_plot + phiz_plot
bpsip_interp, phiz_interp = wf.sol(wf.prec_interp_ts, idxs=[5,6])
m_plot = np.full(Nt_plot, 2)

#compute exact dinamical variables
y, e2, DJ2, bpsip, phiz, zeta = wf.sol(times_plot, idxs=[0,1,4,5,6,7])
dphiz, dzeta, costhL = pyEFPE.precesion_Euler_angles(bpsip, y, DJ2, wf.m1, wf.m2, wf.sz_1, wf.sz_2, wf.sp2_1, wf.sp2_2)
phiz += dphiz
zeta += dzeta
D2_mp2_exact = np.transpose(pyEFPE.compute_necessary_Wigner_D2(phiz, costhL, zeta, return_D2_mp0=False))

#extract the factor (1 - e^2) y^2 from D2_mp2
omega_factor = ((1 - np.maximum(e2,0))*np.square(y))[:,np.newaxis]

#compute the exact and interpolated amplitudes of the plus and cross polarization for m=2
Apc_prec_exact =  wf.compute_Apc_prec_exact(times_plot, np.full(len(times_plot),2))/omega_factor
Apc_prec_interp = wf.compute_Apc_prec_interpolated(times_plot, np.full(len(times_plot),2))/omega_factor

#compute the derivatives of the Euler angles
Dphiz_dt = np.diff(phiz)/np.diff(times_plot)
Dzeta_dt = np.diff(zeta)/np.diff(times_plot)
costhL_mid = 0.5*(costhL[1:] + costhL[:-1])
bpsip_plot_mid = 0.5*(bpsip_plot[1:] + bpsip_plot[:-1])
times_plot_mid = 0.5*(times_plot[1:] + times_plot[:-1])
from scipy.integrate import cumulative_trapezoid
min_rot_viol = cumulative_trapezoid(Dzeta_dt + costhL_mid*Dphiz_dt , x=times_plot_mid, initial=0)


plt.figure(figsize=(13,8))
for iP, label_P in enumerate([r'+', r'\times']):
	plt.plot(x_plot, np.abs(Apc_prec_exact[:,iP]), 'C%s-'%(iP), label=r'$|A_{%s}|$'%(label_P))
	plt.plot(x_plot, np.real(Apc_prec_exact[:,iP]), 'C%s--'%(iP), label=r'$\mathrm{Re}(A_{%s})$'%(label_P))
	plt.plot(x_plot, np.abs(Apc_prec_interp[:,iP]), 'k:')
	plt.plot(x_plot, np.real(Apc_prec_interp[:,iP]), 'k:')
plt.xlabel(r'$\overline{\psi}_p + \phi_{z,0} $ [s]')
plt.legend()
plt.tight_layout()

plt.figure(figsize=(13,8))
for iW in range(5):
	plt.plot(x_plot, np.real(D2_mp2_exact[:,-1-iW]), label=r'$\mathrm{Re}(D_{m,2}^\mathrm{interp}) \; m=%s $'%(2 - iW))

#plt.scatter(bpsip_interp + phiz_interp, np.zeros_like(wf.prec_interp_ts), c='k', label='Interpolation points')
plt.xlabel(r'$\overline{\psi}_p + \phi_{z,0} $ [s]')
plt.legend()
plt.tight_layout()

#make a plot of Euler angles
plt.figure(figsize=(13,8))
plt.plot((2/np.pi)*bpsip_plot, dphiz, label=r'$\delta\phi_z$')
plt.plot((2/np.pi)*bpsip_plot, dzeta, label=r'$\delta\zeta$')
plt.plot((2/np.pi)*bpsip_plot, 1-costhL, label=r'$1-\cos{\theta_L}$')
plt.plot((2/np.pi)*bpsip_plot_mid, min_rot_viol , label=r'$\int \mathrm{d}t \left(\dot{\zeta} + \cos(\theta_L) \dot{\phi}_z \right)$')
plt.xlabel(r'$2\overline{\psi}_p/\pi$ [s]')
plt.legend()
plt.tight_layout()

plt.figure(figsize=(13,8))
plt.plot(freqs,  np.abs(hp), 'C0-', alpha=0.5, label=r'$|h_+|$')
plt.plot(freqs,  np.abs(h_fd[0]), 'C0--', alpha=0.5, label=r'$|h_+^\mathrm{FFT}|$')
plt.plot(freqs,  np.abs(hc), 'C1-', alpha=0.5, label=r'$|h_\times|$')
plt.plot(freqs,  np.abs(h_fd[1]), 'C1--', alpha=0.5, label=r'$|h_\times^\mathrm{FFT}|$')
plt.xlabel(r'$f$ [Hz]')
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.tight_layout()

#plot the waveform in the time domain
plt.figure(figsize=(13,8))
plt.plot(times, h_td[0], 'C0-', alpha=0.5, label=r'$h_+^\mathrm{iFFT}(t)$')
plt.plot(times, h_filtered[0], 'C0--', alpha=0.5, label=r'$h_+(t)$')
plt.plot(times, h_td[1], 'C1-', alpha=0.5, label=r'$h_\times^\mathrm{iFFT}(t)$')
plt.plot(times, h_filtered[1], 'C1--', alpha=0.5, label=r'$h_\times(t)$')
plt.xlabel(r'$t$ [s]')
plt.legend()
plt.tight_layout()


#Runtime
print("\nRuntime: %s seconds" % (time.time() - start_runtime))

plt.show()
