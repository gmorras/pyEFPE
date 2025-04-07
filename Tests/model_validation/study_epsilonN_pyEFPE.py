import os
os.environ.update(
    OMP_NUM_THREADS = '1',
    OPENBLAS_NUM_THREADS = '1',
    NUMEXPR_NUM_THREADS = '1',
    MKL_NUM_THREADS = '1',
)

#import the pyEFPE versions that are going to be used
import sys
sys.path.append('./legacy/no_MB/')

import pyEFPE
from utils_compute_mismatches import *

import numpy as np
from matplotlib import pyplot as plt
plt.rcParams.update({'font.size': 24})
plt.rcParams.update({'lines.linewidth': 2})

import time
start_runtime = time.time()

#universal constants
t_sun_s = 4.92549094831e-6  #GMsun/c**3 [s]

#function to compute the relative logL err (logL(h1,h1)-logL(h1,h2))/logL(h1,h1) = <h1-h2|h1-h2>/<h1|h1> without any minimizations
def compute_logL_rel_err(h1, h2, asd=None):

	#normalize the waveforms
	h1n, h1_norm = normalize_h(h1, asd=asd)
	h2n, h2_norm = normalize_h(h2, asd=asd)
	
	#compute the normalized waveform difference
	dhn = h1n - h2n
	
	#return the relative logL err
	return np.sum(np.square(np.abs(dhn)))/np.sum(np.square(np.abs(h1n)))


############################################################################

#duration to use to compute delta_f = 1/seglen
seglen = 256

#Setting up boundary conditions for random parameter generation
mc_low = 0.95
mc_high = 20
q_low = 0.05
q_high = 1
ecc_low = 0.0
ecc_high = 0.7
s1_low, cths1_low, phs1_low = 0, -1, 0
s1_high, cths1_high, phs1_high = 0.9, 1, 2.0*np.pi
s2_low, cths2_low, phs2_low = 0, -1, 0
s2_high, cths2_high, phs2_high = 0.9, 1, 2.0*np.pi
ciota_low = -1
ciota_high = 1
phiref_low = 0
phiref_high = 2*np.pi
mean_anomaly_low = 0
mean_anomaly_high = 2*np.pi
pol_low = 0
pol_high = 2*np.pi

#setting frequency array to call the waveform
f_min = 20
f_max = 4096
distance_Mpc = 10  # 10 Mpc is default 

#psd to use
psd_name = 'AplusDesign' #None 'AplusDesign'

#Number of waveforms to test
N_tests = 1000

#values of Amplitude tolerances to test
Atol_ref = 1e-9
Atols = np.geomspace(1e-5, 1, 100)

#default parameters for my class
params_pyEFPE = {'distance': distance_Mpc,
                 'f22_start': f_min,
                 'Amplitude_pmax': 2000,
                 }

#output directory
outdir = './outdir/pyEFPE_version_comparisons/'
plotdir = outdir+'/Plots'

#confidence level to plot
CL = 90

############################################################################

#compute params low and high
params_low = np.array([mc_low, q_low, ecc_low, s1_low, cths1_low, phs1_low, s2_low, cths2_low, phs2_low, ciota_low, phiref_low, mean_anomaly_low, pol_low])
params_high = np.array([mc_high, q_high, ecc_high, s1_high, cths1_high, phs1_high, s2_high, cths2_high, phs2_high, ciota_high, phiref_high, mean_anomaly_high, pol_high])

#compute delta_f from seglen
delta_f = 1/seglen

#initialize frequencies
freqs = np.arange(f_min, f_max, delta_f)

#name of the result
string_id = '_study_Atol_%s_%s_N_%s_Mc_%.4g_%.4g_seglen_%s_asd_%s'%(Atol_ref, Atols[0], N_tests, mc_low, mc_high, seglen, psd_name)

#if required, compute the ASD
if type(psd_name)==str:
	asd = compute_asd(delta_f, f_min, f_max, psd_name=psd_name)
	string_id += '_psd_'+psd_name
else:
	asd = None

#try to load the result
try:
	#load the result dictionary
	with open(outdir+'/result'+string_id+'.pickle', 'rb') as handle: result = pickle.load(handle)
	
	#make sure it is the same
	assert np.all(result['freqs']==freqs)
	assert np.all(result['params_low']==params_low)
	assert np.all(result['params_high']==params_high)
	assert len(result['logL_rel_err'])==N_tests
	assert np.all(result['Atols']==Atols)
#otherwise, create the result
except:
	#make sure directories to save stuff exist
	if not os.path.exists(outdir): os.makedirs(outdir)
	if not os.path.exists(plotdir): os.makedirs(plotdir)

	#make random parameters
	rand_params = np.random.uniform(params_low, params_high, size=(N_tests,len(params_low)))

	#generate random waveforms and compute their logL errs
	print('Seglen = %ss,  f_min = %sHz'%(seglen, f_min))
	logL_rel_err = []
	all_params_pyEFPE = []
	for iparam, [mc, q, e, s1, cths1, phs1, s2, cths2, phs2, ciota, phiref, mean_anomaly, pol] in enumerate(rand_params):
		
		#compute component masses in solar masses
		m1 = mc*(q**-0.6)*((1 + q)**0.2)
		m2 = m1*q
		
		#compute inclination
		iota = np.arccos(ciota)
		
		#compute cartesian spin components
		s1x, s1y, s1z = spherical_to_cartesian(np.array([s1, np.arccos(cths1), phs1]))
		s2x, s2y, s2z = spherical_to_cartesian(np.array([s2, np.arccos(cths2), phs2]))
		
		#print mass information
		print('\n%s/%s -> mc=%.3g, q=%.2f, e=%.2g, s1z=%.2g, s1p=%.2g, s2z=%.2g, s2p=%.2g'%(iparam+1, N_tests, mc, q, e, s1z, (s1x**2 + s1y**2)**0.5, s2z, (s2x**2 + s2y**2)**0.5))
		
		#put the params to pass to pyEFPE in dictionary
		params_pyEFPE['mass1'] = m1
		params_pyEFPE['mass2'] = m2
		params_pyEFPE['e_start'] = e
		params_pyEFPE['spin1x'] = s1x
		params_pyEFPE['spin1y'] = s1y
		params_pyEFPE['spin1z'] = s1z
		params_pyEFPE['spin2x'] = s2x
		params_pyEFPE['spin2y'] = s2y
		params_pyEFPE['spin2z'] = s2z
		params_pyEFPE['inclination'] = iota
		params_pyEFPE['phi_start'] = phiref
		params_pyEFPE['mean_anomaly_start'] = mean_anomaly
		all_params_pyEFPE.append(params_pyEFPE.copy())
		
		#Initialize reference pyEFPE waveform
		params_pyEFPE['Amplitude_tol'] = Atol_ref
		wf_ref = pyEFPE.pyEFPE(params_pyEFPE)
		hp_ref, hc_ref = wf_ref.generate_waveform(freqs)

		#Initialize second pyEFPE waveform
		logL_rel_err.append([])
		for Atol in Atols:
			params_pyEFPE['Amplitude_tol'] = Atol
			wf = pyEFPE.pyEFPE(params_pyEFPE)
			hp, hc = wf.generate_waveform(freqs)
			
			#compute logL rel err
			logL_rel_err[-1].append(compute_logL_rel_err(pol_response(hp_ref, hc_ref, pol), pol_response(hp, hc, pol), asd=asd))
			
			#print it
			print('Atol: %.3g, logLerr: %.3g'%(Atol, logL_rel_err[-1][-1]))
	
	#make result dictionary
	result = {'logL_rel_err': np.array(logL_rel_err), 'Atols': Atols,
	          'params': rand_params, 'params_pyEFPE': all_params_pyEFPE,
	          'freqs': freqs, 'params_low': params_low, 'params_high': params_high,
	          }
	
	#save the result file
	with open(outdir+'/result'+string_id+'.pickle', 'wb') as handle: pickle.dump(result, handle, protocol=pickle.HIGHEST_PROTOCOL)

#####################################################################
from matplotlib import pyplot as plt
import matplotlib.collections as mc

#compute the lowe bound-median-upper bound to a given conficence level
CL_levels = np.percentile(result['logL_rel_err'], 50 + CL*np.array([-0.5,0,0.5]) , axis=0)

#make a figure of logL rel err as a function of Atol for all samples
# Create segments for LineCollection
segments = [np.column_stack([Atols, DlogL]) for DlogL in result['logL_rel_err']]
#lc = mc.LineCollection(segments, array=result['params'][2], clim=(ecc_low, ecc_high), cmap='viridis', linewidth=1, alpha=0.5, label=r'$\frac{\langle \delta h | \delta h \rangle}{\langle h | h \rangle}$') #color lines according to eccentricity
lc = mc.LineCollection(segments, cmap='viridis', linewidth=0.25, alpha=0.05, color='k', label=r'Samples', zorder=-np.inf)
fig, ax = plt.subplots(figsize=(10, 8))
ax.add_collection(lc)
ax.fill_between(Atols, CL_levels[2], CL_levels[0], color='tab:orange', alpha=0.3, label=r'$%s\%%$ C.L.'%(CL))
ax.plot(Atols, CL_levels[1], 'limegreen', linewidth=4, label='Median')
#ax.plot(Atols, CL_levels[0], 'tab:orange', linewidth=3)
#ax.plot(Atols, CL_levels[2], 'tab:orange', linewidth=3)
ax.plot(Atols, Atols, 'k', linewidth=4, label=r'$\epsilon_N$')
ax.set_xlim(Atols.min(), Atols.max())
ax.set_xlabel(r'$\epsilon_N$')
ax.set_ylabel(r'$\frac{2 \Delta\log\mathcal{L}_\mathrm{sel}}{\langle h | h \rangle} = \frac{\langle \delta h | \delta h \rangle}{\langle h | h \rangle}$')
ax.set_yscale('log')
ax.set_xscale('log')
ax.legend()
#fig.colorbar(lc, ax=ax, label=r'Eccentricity $e_0$')
fig.tight_layout()
plt.savefig(outdir+'/logLrelerr_N_%s_Mc_%.4g_%.4g_%s.pdf'%(N_tests, mc_low, mc_high, psd_name))

#Runtime
print("\nRuntime: %s seconds" % (time.time() - start_runtime))

plt.show()
