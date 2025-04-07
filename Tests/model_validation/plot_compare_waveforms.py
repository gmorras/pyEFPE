from matplotlib import pyplot as plt

import numpy as np
import pickle

#store the different comparisons
comparison_info = {}

#compare non-precessing TaylorF2 and SpinTaylorT4
comparison_info['noprec_SpinTaylorT4_TaylorF2'] = {
'filenames': ['params_SpinTaylorT4_N_10000_mc_%.3g_%.3g_q_0.05_1_s1_0.9_s2_0.9_fmax_0.8ISCO_PN_spin_6_phase_6_SUA_3_AplusDesign.pickle',
              'params_TaylorF2_N_10000_mc_%.3g_%.3g_q_0.05_1_s1_0.9_s2_0.9_fmax_0.8ISCO_PN_spin_6_phase_6_SUA_3_AplusDesign.pickle'],
'labels': ['SpinTaylorT4', 'TaylorF2']
}

#compare precessing SpinTaylorT4 with pn_spin_order=4 and pn_spin_order=4
comparison_info['prec_SpinTaylorT4_PN_spin_4_6'] = {
'filenames': ['params_prec_SpinTaylorT4_N_2000_mc_%.3g_%.3g_q_0.05_1_s1_0.9_s2_0.9_fmax_0.8ISCO_PN_spin_4_phase_6_SUA_3_ROTs1s2_False_AplusDesign.pickle',
              'params_prec_SpinTaylorT4_N_2000_mc_%.3g_%.3g_q_0.05_1_s1_0.9_s2_0.9_fmax_0.8ISCO_PN_spin_6_phase_6_SUA_3_ROTs1s2_False_AplusDesign.pickle'],
'labels': ['2PN Spin', '3PN Spin']
}

#compare IMRPhenomXP
comparison_info['IMRPhenomXP_prec_noprec'] = {
'filenames': ['params_prec_IMRPhenomXP_N_2000_mc_%.3g_%.3g_q_0.05_1_s1_0.9_s2_0.9_fmax_0.8MECO_PN_spin_6_phase_6_SUA_3_ROTs1s2_False_AplusDesign.pickle',
              'params_IMRPhenomXP_N_10000_mc_%.3g_%.3g_q_0.05_1_s1_0.9_s2_0.9_fmax_0.8MECO_PN_spin_6_phase_6_SUA_3_AplusDesign.pickle'],
'labels': ['Generic Spins', 'Aligned Spins']
}

#compare IMRPhenomXP
comparison_info['IMRPhenomXP_fmax_0.8_0.2MECO'] = {
'filenames': ['params_prec_IMRPhenomXP_N_2000_mc_%.3g_%.3g_q_0.05_1_s1_0.9_s2_0.9_fmax_0.8MECO_PN_spin_6_phase_6_SUA_3_ROTs1s2_False_AplusDesign.pickle',
              'params_prec_IMRPhenomXP_N_2000_mc_%.3g_%.3g_q_0.05_1_s1_0.9_s2_0.9_fmax_0.2MECO_PN_spin_6_phase_6_SUA_3_ROTs1s2_False_AplusDesign.pickle'],
'labels': [r'$f_\mathrm{max} = 0.8f_\mathrm{MECO} $', r'$f_\mathrm{max} = 0.2 f_\mathrm{MECO}$'],
'legend_kwargs': dict(columnspacing=0.7, handletextpad=0.7),
}

#compare IMRPhenomXP
comparison_info['TaylorF2Ecc_Atol_1e-3_1'] = {
'filenames': ['params_ecc_0.15_TaylorF2Ecc_N_10000_mc_%.3g_%.3g_q_0.05_1_s1_0.3_s2_0.3_fmax_0.8ISCO_PN_spin_6_phase_6_SUA_3_AplusDesign.pickle',
              'params_ecc_0.15_TaylorF2Ecc_N_10000_mc_%.3g_%.3g_q_0.05_1_s1_0.3_s2_0.3_fmax_0.8ISCO_PN_spin_6_phase_6_SUA_3_Atol_1_AplusDesign.pickle'],
'labels': [r'$\epsilon_N = 0.001$', r'$\epsilon_N =1$']
}


##########################################################################################

#output directory
outdir = './outdir/waveform_comparisons/'

#comparison to plot
plot_label = 'TaylorF2Ecc_Atol_1e-3_1'

#chirp masses
mc_lows  = [12,  8, 5, 3.3, 2.2, 1.4, 0.95]
mc_highs = [20, 12, 8, 5, 3.3, 2.2, 1.40]
seglens =  [4, 8, 16, 32, 64, 128, 256]

#quantiles to show in plots
quantiles = [.05, .5, .95]

#colors
colors = ['C0', 'C1']

#remove the samples with f_max <= f_min + ndf_min*delta_f
remove_below_fmax = True
f_min = 20
ndf_min = 10

#choose to show also chirp mass range
show_chirp_mass_range = False

##########################################################################################

# load the data
mismatches, all_quantiles, xlabels = [], [], []
for i in range(len(mc_lows)):
	
	#load the result dictionary
	try:
		results = list()
		for filename in comparison_info[plot_label]['filenames']:
			with open(outdir+'/'+filename%(mc_lows[i], mc_highs[i]), 'rb') as handle: results.append(pickle.load(handle))
			#make sure the seglen is correct
			assert results[-1]['seglen'] == seglens[i]
	except:
		continue
	
	#append the mismatches and quantiles to use
	mismatches.append([np.log10(np.maximum(np.abs(result['mismatches']), 1e-17)) for result in results])

	#check if there are samples below the specified fmin
	if remove_below_fmax:
		for ir, result in enumerate(results):
			#samples below f_min
			i_remove = (result['f_max'] < (f_min + (1+ndf_min)/seglens[i]))
			#if there are any, do not display this case in plots
			if np.any(i_remove): mismatches[-1][ir] = [np.nan, np.nan]
		
	all_quantiles.append(quantiles)
	
	#append a label
	if show_chirp_mass_range: xlabels.append(r'%s\,s\n$[%s , %s]$'%(seglens[i], mc_lows[i], mc_highs[i]))
	else:                     xlabels.append(r'%s\,s'%(seglens[i]))
	
#transpose mismatches
mismatches = list(map(list, zip(*mismatches)))

# Create the plot
fig, ax = plt.subplots(figsize=(12, 8))

# Make violin plot
violins = list()
for color, mismatch, side in zip(colors, mismatches, ['low', 'high']):

	#plot this mismatch on its side of the violin
	violins.append(ax.violinplot([MM for MM in mismatch], widths=0.8, quantiles=all_quantiles, showextrema=False, side=side))

	#set colors
	for body in violins[-1]['bodies']:
		body.set_facecolor(color)
		body.set_edgecolor(color)
		body.set_alpha(1)
	violins[-1]['cquantiles'].set_color('k')

# Add labels etc
ax.set_xticks(1 + np.arange(len(xlabels)), xlabels)

if show_chirp_mass_range: ax.set_xlabel(r'Duration $T$ / Chirp mass range $\\left[\\mathcal{M}_{c,\\mathrm{min}}, \\mathcal{M}_{c,\\mathrm{max}} \\right]$ $[M_\\odot]$')
else:                     ax.set_xlabel(r'Duration $T$')

ax.tick_params(axis='x')
ax.set_xlim(0.5, len(all_quantiles)+0.5)
ax.set_ylim(top=0)
ax.set_ylabel(r'$\log_{10}(\overline{\mathcal{MM}})$')
if 'legend_kwargs' not in comparison_info[plot_label].keys(): comparison_info[plot_label]['legend_kwargs'] = {}
plt.legend([violins[0]['bodies'][0], violins[1]['bodies'][0], violins[0]['cquantiles']], [*comparison_info[plot_label]['labels'], '%s\nquantiles'%(quantiles)], ncols=3, loc=(0,1), framealpha=0, **comparison_info[plot_label]['legend_kwargs'])

plt.tight_layout()
plt.savefig(outdir+'/MM_violins_'+plot_label+'.pdf')


plt.show()

