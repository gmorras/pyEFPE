import os
os.environ.update(
    OMP_NUM_THREADS = '1',
    OPENBLAS_NUM_THREADS = '1',
    NUMEXPR_NUM_THREADS = '1',
    MKL_NUM_THREADS = '1',
)

from utils_compute_mismatches import *
import numpy as np
from matplotlib import pyplot as plt

import time
start_runtime = time.time()

############################################################################

#location of files to plot
files_dir = './outdir/waveform_comparisons/'

#sample files to plot
file_names = ['params_ecc_0.15_TaylorF2Ecc_N_10000_mc_%.3g_%.3g_q_0.05_1_s1_0.3_s2_0.3_fmax_0.8ISCO_PN_spin_6_phase_6_SUA_3_AplusDesign.pickle',
              'params_ecc_0.15_TaylorF2Ecc_N_10000_mc_%.3g_%.3g_q_0.05_1_s1_0.3_s2_0.3_fmax_0.8ISCO_PN_spin_6_phase_6_SUA_3_Atol_1_AplusDesign.pickle']

#labels of each realization
labels = ['$\\epsilon_N = 0.001$',
          '$\\epsilon_N = 1$']
          
#chirp mass ranges
mc_lows  = [12,  8, 5, 3.3, 2.2, 1.4, 0.95]
mc_highs = [20, 12, 8, 5, 3.3, 2.2, 1.40]

#parameter ranges to consider
q_min = 0.5
siz_max = 0.15
ecc_max = 0.15

############################################################################

#loop over files
mismatches, eccs, mcs = [], [], []
for file_name in file_names:
	mismatches.append(np.array([]))
	eccs.append(np.array([]))
	mcs.append(np.array([]))	
	#loop over chirp mass ranges	
	for imc in range(len(mc_lows)):
		
		#load samples		
		with open(files_dir+'/'+file_name%(mc_lows[imc], mc_highs[imc]), 'rb') as handle: samples = pickle.load(handle)

		#samples to select
		idxs_sel = (samples['q']>q_min) & (np.abs(samples['s1z'])<siz_max) & (np.abs(samples['s2z'])<siz_max) & (samples['ecc']<ecc_max)

		#print the selected number of samples
		print('Selected %s/%s samples'%(sum(idxs_sel), len(idxs_sel)))

		#append the samples to lists
		mismatches[-1] = np.append(mismatches[-1], samples['mismatches'][idxs_sel])
		eccs[-1] = np.append(eccs[-1], samples['ecc'][idxs_sel])
		mcs[-1] = np.append(mcs[-1], samples['mc'][idxs_sel])

#make dot marker fot legend
from matplotlib.lines import Line2D
import matplotlib.colors as colors
black_point = Line2D([], [], color='k', marker='o', linestyle='None')

#make plot
fig, axs = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True, figsize=(12,5.5))

axs[0].set_ylabel(r'$\overline{\mathcal{MM}}$')
axs[0].set_yscale('log')
for i, ax in enumerate(axs):
	scat = ax.scatter(eccs[i], mismatches[i], c=mcs[i], s=8, label=labels[i], norm=colors.BoundaryNorm(boundaries=np.flip(mc_highs+[mc_lows[-1]]), ncolors=plt.cm.viridis.N))
	ax.set_xlabel(r'$e_0$')
	ax.set_xlim(0, ecc_max - 1e-10)
	
for i, ax in enumerate(axs.flat): ax.legend(handles=[black_point], labels=[labels[i]])

plt.tight_layout()
plt.subplots_adjust(wspace=0, hspace=0)

fig.colorbar(scat, ax=axs.ravel().tolist(), label=r'$\mathcal{M}_c$ [$M_\odot$]')

plt.savefig(files_dir+'/ecc_scatter_siz_max_%s_q_min_%s.pdf'%(siz_max, q_min))

plt.show()	
	
	
              
