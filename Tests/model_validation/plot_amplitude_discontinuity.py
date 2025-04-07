import pyEFPE
import numpy as np

from matplotlib import pyplot as plt

from matplotlib import patches

import time
start_runtime = time.time()

#array with parameters
params = {'distance': 10, 'f22_start': 20, 'mass1': 1.6562726877323808, 'mass2': 1.3308923544954927, 'e_start': 0.5957802134964009, 'spin1x': -0.09239644519075679, 'spin1y': -0.7411760949352235, 'spin1z': 0.5008164040471363, 'spin2x': 0.3581030373389089, 'spin2y': -0.11778721591082868, 'spin2z': 0.2603190513633763, 'inclination': 1.4784677838701057, 'phi_start': 1.198524474535042, 'mean_anomaly_start': 0.7162449541571039}

#number of times to plot
Nt_plot = 1000000

#zoom region
x0_zoom, xf_zoom  = 83.4, 84.2

#initialize waveform class
start_soltime = time.time()
wf = pyEFPE.pyEFPE(params)
print("\nTime to initialize waveform class: %s seconds \n" % (time.time() - start_soltime))

#times to plot
times_plot = -np.geomspace(-wf.sol.all_ts[0], -wf.sol.all_ts[-1], Nt_plot)
#compute Euler angles
bpsip_plot, phiz_plot = wf.sol(times_plot, idxs=[5,6])
x_plot = bpsip_plot + phiz_plot
bpsip_interp, phiz_interp = wf.sol(wf.prec_interp_ts, idxs=[5,6])
m_plot = np.full(Nt_plot, 2)

#compute the exact and interpolated amplitudes of the plus and cross polarization for m=2
Apc_prec_exact =  wf.h0_pref*wf.compute_Apc_prec_exact(times_plot, np.full(len(times_plot),2))
Apc_prec_interp = wf.h0_pref*wf.compute_Apc_prec_interpolated(times_plot, np.full(len(times_plot),2))

#make the figure of exact and interpolated amplitudes
fig, axs = plt.subplots(2, 1, figsize=(13, 8))

#make full plot
handles = []
for iP, label_P in enumerate([r'+', r'\times']):
	handles.append(*axs[0].plot(x_plot, np.real(Apc_prec_exact[:,iP]), 'C%s'%(iP), linewidth=2))
	interp_handle = axs[0].plot(x_plot, np.real(Apc_prec_interp[:,iP]), 'k:', linewidth=2)

handles.append(*interp_handle)
axs[0].set_xlim(x_plot[0], x_plot[-1])

#make a zoom to discontinuity region
i_zoom = (x_plot>x0_zoom) & (x_plot < xf_zoom)
x_zoom, Apc_prec_exact_zoom, Apc_prec_interp_zoom = x_plot[i_zoom], Apc_prec_exact[i_zoom], Apc_prec_interp[i_zoom]
xlow_zoom, xhigh_zoom, Alow_zoom, Ahigh_zoom = min(x_zoom), max(x_zoom), min(np.amin(np.real(Apc_prec_exact_zoom)), np.amin(np.real(Apc_prec_interp_zoom))), max(np.amax(np.real(Apc_prec_exact_zoom)), np.amax(np.real(Apc_prec_interp_zoom)))
dA = 0.05*max(np.abs(Alow_zoom), np.abs(Ahigh_zoom))
Alow_zoom, Ahigh_zoom = Alow_zoom - dA, Ahigh_zoom + dA
#plot evaluation points
t0_zoom, tf_zoom = times_plot[i_zoom][[0,-1]]
ts_interp_zoom = wf.prec_interp_ts[(wf.prec_interp_ts > t0_zoom) & (wf.prec_interp_ts < tf_zoom)]
xs_interp_zoom = np.sum(wf.sol(ts_interp_zoom, idxs=[5,6]), axis=0)
Apc_exact_interp_zoom = wf.h0_pref*wf.compute_Apc_prec_exact(ts_interp_zoom, np.full(len(ts_interp_zoom),2))
for iP, label_P in enumerate([r'+', r'\times']):
	axs[1].plot(x_zoom, np.real(Apc_prec_exact_zoom[:,iP]),  'C%s'%(iP), linewidth=2)
	axs[1].plot(x_zoom, np.real(Apc_prec_interp_zoom[:,iP]), 'k:', linewidth=2)
	handles.append(axs[1].scatter(xs_interp_zoom, np.real(Apc_exact_interp_zoom[:, iP]), s=50, c='C%s'%(iP)))

#labels for all handles
labels = [r'Exact $\mathrm{Re}(\mathsf{A}^{+}_{2,2})$', r'Exact $\mathrm{Re}(\mathsf{A}^{\times}_{2,2})$', r'Interpolated', r'$\mathrm{Re}(\mathsf{A}^{+}_{2,2})$ Interpolation Points', r'$\mathrm{Re}(\mathsf{A}^{\times}_{2,2})$ Interpolation Points']
axs[0].legend(handles=handles[:3], labels =labels[:3], ncols=3, loc=(0,1), fontsize=21, framealpha=0)
axs[1].legend(handles=handles[3:], labels =labels[3:], ncols=2, loc=(0,1), fontsize=21, framealpha=0)
axs[1].set_xlim(xlow_zoom, xhigh_zoom)
axs[1].set_ylim(Alow_zoom, Ahigh_zoom)
axs[1].set_xlabel(r'$\overline{\psi}_\mathrm{p} + \phi_{z,0} $')
import matplotlib.ticker as ticker
axs[0].yaxis.set_major_formatter(ticker.FormatStrFormatter('%.3g'))
axs[1].yaxis.set_major_formatter(ticker.FormatStrFormatter('%.3g'))
#indicate the zoom region
patch_zoom = patches.Rectangle((xlow_zoom, Alow_zoom), xhigh_zoom - xlow_zoom, Ahigh_zoom - Alow_zoom, edgecolor='black', facecolor='none', zorder=10)
con_low = patches.ConnectionPatch(xyA=(xlow_zoom, Alow_zoom), xyB=(xlow_zoom, Ahigh_zoom), coordsA="data", coordsB="data", axesA=axs[0], axesB=axs[1], edgecolor='black')
con_high = patches.ConnectionPatch(xyA=(xhigh_zoom, Alow_zoom), xyB=(xhigh_zoom, Ahigh_zoom), coordsA="data", coordsB="data", axesA=axs[0], axesB=axs[1], edgecolor='black')
axs[0].add_patch(patch_zoom)
fig.add_artist(con_low)
fig.add_artist(con_high)

plt.tight_layout()
plt.savefig('outdir/amplitude_discontinuity_example.pdf')

#Runtime
print("\nRuntime: %s seconds" % (time.time() - start_runtime))

plt.show()
