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

#output directory
outdir='./outdir/waveform_comparisons'

#Setting up boundary conditions for random parameter generation
seglen = 256
mc_low = 0.95
mc_high = 1.4
ecc_high = 0
q_low = 0.05
q_high = 1
s1_high = 0.9
s2_high = 0.9
f_min = 20
distance_Mpc = 10  # 10 Mpc is default 
approximant = 'TaylorF2'
flim_type = 'ISCO' #one of ['ISCO', 'MECO']
fmax_flim = 0.8 #fraction of flim that fmax represents
precessing = False

#PN orders for LAL waveform
pn_spin_order = 6
pn_phase_order = 6
pn_amplitude_order = 0

#default parameters for my class
params_pyEFPE = {'SUA_kmax': 3,
                 #'Amplitude_tol': 1,
                 }

#PSD to use, put psd_name=None to turn off the psd
psd_name = 'AplusDesign' #'AplusDesign' 'aLIGO175MpcT1800545' 'aLIGOAPlusDesignSensitivityT1800042'

#Number of waveforms to test
Nsamples = 10000

#stuff for optimization of precession mismatches
Nph_tries = 40
rotate_individual_spins=False
additional_minimize_threshold=1e-2

#number of workers for parallelization
nworkers = 4 #set it to -1 to set nworkers=max(mp.cpu_count()-1, 1)

#Number of top mismatches to print
N_top_MM_print = 50

##############################################################################

#generate string id
string_id  = generate_string_ID(Nsamples, precessing, approximant, psd_name, mc_low, mc_high, q_low, q_high, s1_high, s2_high, flim_type, pn_spin_order, pn_phase_order, params_pyEFPE, rotate_individual_spins=rotate_individual_spins, fmax_flim=fmax_flim, ecc_high=ecc_high)

#Try to load a dictionary with the samples
try:
	with open(outdir+'/params'+string_id+'.pickle', 'rb') as handle: samples = pickle.load(handle)
except:
	#compute the mismatches using our function
	samples = random_mismatches_with_EFPE(Nsamples, approximant, seglen=seglen, f_min=f_min, flim_type=flim_type, fmax_flim=fmax_flim, psd_name=psd_name, distance_Mpc=distance_Mpc, mc_low=mc_low, mc_high=mc_high, q_low=q_low, q_high=q_high, s1_high=s1_high, s2_high=s2_high, ecc_high=ecc_high, precessing=precessing, params_pyEFPE=params_pyEFPE, pn_spin_order=pn_spin_order, pn_phase_order=pn_phase_order, pn_amplitude_order=pn_amplitude_order, outdir=outdir, Nph_tries=Nph_tries, rotate_individual_spins=rotate_individual_spins, additional_minimize_threshold=additional_minimize_threshold, nworkers=nworkers)

print('\nTop %s realizations with largest mismatch\n'%(N_top_MM_print))
#print the parameters associated with the top ones
for iMM in np.argsort(-samples['mismatches'])[:N_top_MM_print]:
	print('\nMismatch: ', samples['mismatches'][iMM])
	print('params =', samples['all_params_pyEFPE'][iMM])

#if the directory for plots does not exist, create it
plots_dir = outdir+'/Plots/'
if not os.path.exists(plots_dir): os.makedirs(plots_dir)

#make a histogram of the mismatches
bins = np.geomspace(np.nanmin(samples['mismatches']), np.nanmax(samples['mismatches']), 25)
plt.figure(figsize=(12,8))
plt.hist(samples['mismatches'], bins=bins)
plt.xlabel(r'Mismatch')
plt.ylabel(r'Number of waveforms')
plt.xscale('log')
plt.xlim(bins[0], bins[-1])
plt.savefig(plots_dir+'histo_mismatches'+string_id+'.pdf')
plt.tight_layout()

#make a scatter plot of chi_eff-q
plt.figure(figsize=(12,8))
plt.scatter(samples['chi_eff'], samples['mismatches'], c=samples['q'], vmin=q_low, vmax=q_high)
plt.xlabel(r'$\chi_\mathrm{eff}$')
if s1_high==s2_high: plt.xlim(-s1_high, s1_high)
plt.ylabel(r'$\overline{\mathcal{MM}}$')
plt.yscale('log')
plt.colorbar(label=r'$q$')
plt.tight_layout()
plt.savefig(plots_dir+'mismatches_chi_eff_q'+string_id+'.pdf')

#make a scatter plot of chi_p-chi_eff
if precessing:
	plt.figure(figsize=(12,8))
	plt.scatter(samples['chi_p'], samples['mismatches'], c=np.abs(np.cos(samples['iota'])))
	plt.xlabel(r'$\chi_\mathrm{p}$')
	plt.ylabel(r'$\overline{\mathcal{MM}}$')
	plt.yscale('log')
	plt.colorbar(label=r'$|\cos{\iota}|$')
	plt.tight_layout()
	plt.savefig(plots_dir+'mismatches_chip_cinc'+string_id+'.pdf')

#if the system is eccentric, make a scatter plot of ecc-q
if ecc_high>0:
	plt.figure(figsize=(12,8))
	plt.scatter(samples['ecc'], samples['mismatches'], c=samples['q'], vmin=q_low, vmax=q_high)
	plt.xlabel(r'$e$')
	plt.xlim(0, ecc_high)
	plt.ylabel(r'$\overline{\mathcal{MM}}$')
	plt.yscale('log')
	plt.colorbar(label=r'$q$')
	plt.tight_layout()
	plt.savefig(plots_dir+'mismatches_ecc_q'+string_id+'.pdf')
	

#Runtime
print("\nRuntime: %s seconds" % (time.time() - start_runtime))

plt.show()

#make plots comparing waveforms explicitly
compare_waveforms_from_params(samples, plot_order='maximum mismatch')

plt.show()
