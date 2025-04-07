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

#test waveform generation
freqs = np.arange(flow, fhigh, 1/seglen)
ws = 2*np.pi*freqs

start_soltime = time.time()
hp, hc = wf.generate_waveform(freqs)
print("\nTime to compute hp, hc: %s seconds \n" % (time.time() - start_soltime))

#check in which interpolant of the different modes the ws are in
start_soltime = time.time()
f_idxs, interp_idxs = pyEFPE.functions.sorted_vals_in_intervals(ws, wf.ts_interp_w0, wf.ts_interp_wf)
print("\nTime to compute sorted_vals_in_intervals with current implementation: %s seconds \n" % (time.time() - start_soltime))

#check in which interpolant of the different modes the ws are in
start_soltime = time.time()
interp_idxs_where, f_idxs_where = np.where((ws[np.newaxis,:]>=wf.ts_interp_w0[:,np.newaxis]) & (ws[np.newaxis,:]<=wf.ts_interp_wf[:,np.newaxis]))
print("\nTime to compute sorted_vals_in_intervals with np.where: %s seconds \n" % (time.time() - start_soltime))

print('\nAre all f_idxs the same?',np.all(f_idxs==f_idxs_where))
print('Are all interp_idxs the same?',np.all(interp_idxs==interp_idxs_where))

#Runtime
print("\nRuntime: %s seconds" % (time.time() - start_runtime))

