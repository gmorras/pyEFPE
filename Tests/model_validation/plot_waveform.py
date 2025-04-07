import pyEFPE
import numpy as np
import time
import matplotlib.pyplot as plt

from utils_compute_mismatches import *

start_runtime = time.time()


#array with parameters
params = {'mass1': 2.4,
          'mass2': 1.2,
          'e_start': 0.7,
          'spin1x': -0.44,
          'spin1y': -0.26,
          'spin1z': 0.48,
          'spin2x': -0.31,
          'spin2y': 0.01,
          'spin2z': -0.84,
          'inclination': 0.5*np.pi,
          'f22_start': 10,
          'distance': 100,
          'Interpolate_Amplitudes': True
          }

print('chi1 = %.4g   chi2 = %.4g'%(np.linalg.norm([params['spin1x'], params['spin1y'], params['spin1z']]),
                                   np.linalg.norm([params['spin2x'], params['spin2y'], params['spin2z']])))

#stuff for frequency array generation
seglen = 128
flow   = 20
fhigh  = 4096

#zoom times
t0_zoom_1, tf_zoom_1  = -19, -13.7
t0_zoom_2, tf_zoom_2  = -16.85, -16.14

#polarization colors
hp_color = 'C1'
hc_color = 'C0'

#psds to compare and plot
test_psd_names     = ['AplusDesign', 'avirgo_O5high_NEW']
plot_psd_name      = 'AplusDesign'
plot_psd_plot_name = r'LIGO \, A+'

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

#test subroutine to compute stationary times
freqs = np.arange(flow, fhigh, 1/seglen)

#test polarization generation
start_soltime = time.time()
hp, hc = wf.generate_waveform(freqs)
print("\nTime to compute hp, hc: %s seconds \n" % (time.time() - start_soltime))

#compute the waveforms in the time domain
from scipy.fft import irfft
#compute the time array
times = (1/(2*fhigh))*np.arange(int(2*seglen*fhigh))
#loop over polarizations
h_td = list()
for h in [hp, hc]:
	#put the low frequencies of h (they are 0)
	h_padded = np.zeros(int(seglen*fhigh), dtype=h.dtype)
	h_padded[int(seglen*flow):int(seglen*fhigh)] = h
	#perform the inverse FFT, taking into account that h is the FFT of a real signal
	fft_len = int(2*seglen*fhigh)
	h_td.append((fft_len/seglen)*irfft(h_padded, n=fft_len))

#approximated
Nt_plot = 10000
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

#compute the psds that will be used
test_asds = {}
for test_psd_name in test_psd_names:
	#compute this psd
	test_asds[test_psd_name] = compute_asd(1/seglen, flow, fhigh, psd_name=test_psd_name)

plot_asd = compute_asd(1/seglen, flow, fhigh, psd_name=plot_psd_name)

#compute the SNR of the test waveform
hp_snr = np.linalg.norm(hp/plot_asd)*2/np.sqrt(seglen)
hc_snr = np.linalg.norm(hc/plot_asd)*2/np.sqrt(seglen)

print('SNRs-> hp: %.3g hc: %.3g'%(hp_snr, hc_snr))

##################### make plots ###########################
plt.rcParams.update({'font.size': 24})
plt.rcParams.update({'lines.linewidth': 2})
from matplotlib import patches

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

plt.figure(figsize=(11,7),dpi=100)
#plt.plot(freqs, np.real(hp), 'C0--', label=r'$\mathrm{Re}(h_+)$')
#plt.plot(freqs, np.imag(hp), 'C0:' , label=r'$\mathrm{Im}(h_+)')
plt.plot(freqs,  np.abs(hp), label=r'$|\tilde{h}_+|$ $[\mathrm{Hz}^{-1}]$', color=hp_color, linewidth=2.5)
#plt.plot(freqs, np.real(hc), 'C1--', label=r'$\mathrm{Re}(h_\times)$')
#plt.plot(freqs, np.imag(hc), 'C1:' , label=r'$\mathrm{Im}(h_\times)')
plt.plot(freqs,  np.abs(hc), label=r'$|\tilde{h}_\times|$ $[\mathrm{Hz}^{-1}]$', color=hc_color , linewidth=2.0)
plt.plot(freqs,  plot_asd, label=r'$\sqrt{S_n^\mathrm{%s}}$ $[\mathrm{Hz}^{-1/2}]$'%(plot_psd_plot_name), color='k', linewidth=2.5)
plt.xlabel(r'$f$ [Hz]')
plt.xscale('log');
plt.yscale('log');
#plt.xlim(freqs[0], freqs[-1])
plt.xlim(20,2048);
plt.ylim(1e-26,5.4e-23);
plt.legend(loc='upper right')
plt.tight_layout()
plt.savefig('outdir/waveform_example_fd.pdf')

#plot the waveform in the time domain
times = times-times[-1]
plt.figure(figsize=(13,8))
plt.plot(times, h_td[0], label=r'$h_+$', color=hp_color)
plt.plot(times, h_td[1], label=r'$h_\times$', color=hc_color)
plt.xlabel(r'$t$ [s]')
plt.legend()
plt.tight_layout()

#put it with multiple zooms to show the different scales
fig, axs = plt.subplots(3, 1, figsize=(20, 12))
#Unzoomed plot
axs[0].plot(times, h_td[0], label=r'$h_+$', color=hp_color)
axs[0].plot(times, h_td[1], label=r'$h_\times$', color=hc_color)
axs[0].legend(loc='upper left')
axs[0].set_xlim(times[0], times[-1])
axs[0].set_ylabel(r'Strain $h$')
#make the first zoom
i_zoom_1 = (times>t0_zoom_1) & (times<tf_zoom_1)
times_zoom_1, hp_zoom_1, hc_zoom_1 = times[i_zoom_1], h_td[0][i_zoom_1], h_td[1][i_zoom_1]
tlow_1, thigh_1, hlow_1, hhigh_1 = min(times_zoom_1), max(times_zoom_1), 1.025*min(min(hp_zoom_1), min(hp_zoom_1)), 1.025*max(max(hp_zoom_1), max(hp_zoom_1))
axs[1].plot(times_zoom_1, hp_zoom_1, label=r'$h_+$', color=hp_color)
axs[1].plot(times_zoom_1, hc_zoom_1, label=r'$h_\times$', color=hc_color)
axs[1].set_xlim(tlow_1, thigh_1)
axs[1].set_ylim(hlow_1, hhigh_1)
axs[1].set_ylabel(r'Strain $h$')
#inidcate the first zoom region
patch_zoom_1 = patches.Rectangle((tlow_1, hlow_1), thigh_1-tlow_1, hhigh_1 - hlow_1, edgecolor='black', facecolor='none', zorder=10)
con_low_1 = patches.ConnectionPatch(xyA=(tlow_1, hlow_1), xyB=(tlow_1, hhigh_1), coordsA="data", coordsB="data", axesA=axs[0], axesB=axs[1], edgecolor='black')
con_high_1 = patches.ConnectionPatch(xyA=(thigh_1, hlow_1), xyB=(thigh_1, hhigh_1), coordsA="data", coordsB="data", axesA=axs[0], axesB=axs[1], edgecolor='black')
axs[0].add_patch(patch_zoom_1)
fig.add_artist(con_low_1)
fig.add_artist(con_high_1)
#make the second zoom
i_zoom_2 = (times>t0_zoom_2) & (times<tf_zoom_2)
times_zoom_2, hp_zoom_2, hc_zoom_2 = times[i_zoom_2], h_td[0][i_zoom_2], h_td[1][i_zoom_2]
tlow_2, thigh_2, hlow_2, hhigh_2 = min(times_zoom_2), max(times_zoom_2), 1.025*min(min(hp_zoom_2), min(hp_zoom_2)), 1.025*max(max(hp_zoom_2), max(hp_zoom_2))
axs[2].plot(times_zoom_2, hp_zoom_2, label=r'$h_+$', color=hp_color, linewidth=2.5)
axs[2].plot(times_zoom_2, hc_zoom_2, label=r'$h_\times$', color=hc_color, linewidth=2.5)
axs[2].set_xlim(tlow_2, thigh_2)
axs[2].set_ylim(hlow_2, hhigh_2)
axs[2].set_ylabel(r'Strain $h$')
#indicate the second zoom region
patch_zoom_2 = patches.Rectangle((tlow_2, hlow_2), thigh_2-tlow_2, hhigh_2 - hlow_2, edgecolor='black', facecolor='none', zorder=10)
con_low_2    = patches.ConnectionPatch(xyA=(tlow_2, hlow_2), xyB=(tlow_2, hhigh_2), coordsA="data", coordsB="data", axesA=axs[1], axesB=axs[2], edgecolor    = 'black')
con_high_2   = patches.ConnectionPatch(xyA=(thigh_2, hlow_2), xyB=(thigh_2, hhigh_2), coordsA="data", coordsB="data", axesA=axs[1], axesB=axs[2], edgecolor    = 'black')
axs[1].add_patch(patch_zoom_2)
fig.add_artist(con_low_2)
fig.add_artist(con_high_2)
#common stuff
axs[2].set_xlabel(r'$t - t_\mathrm{c}$ [s]')
plt.tight_layout()
plt.savefig('outdir/waveform_example_td.pdf')

#make a plot of the psds being consideres
plt.figure(figsize=(13,8))
for test_psd_name in test_psd_names:
	#plot it
	plt.loglog(freqs, test_asds[test_psd_name], label=test_psd_name)

plt.xlabel(r'$f$ $[\mathrm{Hz}]$')
plt.ylabel(r'ASD $[\mathrm{Hz}^{-1/2}]$')
plt.xlim(freqs[0], freqs[-1])
plt.legend()
plt.savefig('outdir/PSDs.png')

#Runtime
print("\nRuntime: %s seconds" % (time.time() - start_runtime))

plt.show()
