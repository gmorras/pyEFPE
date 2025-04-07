import numpy as np
import time

#function to compute exp(i*x) directly
def exp_ix_direct(x):
	return np.exp(1j*x)

#function to compute exp(i*x) from sine and cosine
def exp_ix_sincos(x):
	return np.cos(x)+1j*np.sin(x)

#function to compute exp(i*x) from sine and cosine faster
def exp_ix_npcomplex128(x):
	z = np.zeros_like(x, dtype=np.complex128)
	z.real, z.imag = np.cos(x), np.sin(x)
	return z


#function to compute runtime
def compute_runtime(func, x, Ntries):
	
	#start a timer
	start = time.time()
	
	#run the function Ntries times
	for _ in range(Ntries):
		func(x)
	
	#return the time taken
	total_time = time.time() - start
	return total_time/Ntries

#number of angles to test
Ns_test = 2**np.arange(27)

#number of tries of each
Ntries = 10

#functions to test
funcs = [exp_ix_direct, exp_ix_sincos, exp_ix_npcomplex128]

#their names
func_labels = [r'$\exp(\mathrm{i}x)$', r'$\cos(x) + \mathrm{i}\,\sin(x)$', r'Alt $\cos(x) + \mathrm{i}\,\sin(x)$']

#loop over number o points to test
run_times = np.zeros((len(funcs),len(Ns_test)))
for iN, N in enumerate(Ns_test):

	#generate random angles
	phi = np.random.uniform(low=-1000*np.pi, high=1000*np.pi, size=N)
	
	for ifunc, func in enumerate(funcs):
		run_times[ifunc, iN] = compute_runtime(func, phi, Ntries)
	
	print('N = %s -> run_times = %s s -> ratio = %.3f'%(N, run_times[:,iN], run_times[0,iN]/run_times[1,iN]))

#make a plot
from matplotlib import pyplot as plt
plt.rcParams.update({'font.size': 24})
plt.rcParams.update({'lines.linewidth': 2})

plt.figure(figsize=(16,10))
for ifunc, label in enumerate(func_labels):
	plt.loglog(Ns_test, run_times[ifunc], label=label)

plt.xlabel('Number of evaluation points')
plt.ylabel('Runtime [s]')
plt.xlim(Ns_test[0], Ns_test[-1])
plt.legend()
plt.show()

