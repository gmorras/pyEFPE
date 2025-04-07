import numpy as np
from correct_PN_derivatives import *

#define the samples we want to test
test_samples = []
test_samples.append({'m1': 1.58, 'm2': 0.975, 'chi_eff': 0.76, 's2_1': 0.916, 's2_2': 0.541, 'q1': 1.78, 'q2': 0.567, 'y': 0.89, 'e2': 0.78, 'dchi2': 0.81, 'dchi': 0.9, 'sperp2': 0.38})
test_samples.append({'m1': 1.58, 'm2': 0.975, 'chi_eff': 0, 's2_1': 0, 's2_2': 0, 'q1': 1.78, 'q2': 0.567, 'y': 0.89, 'e2': 0.78, 'dchi2': 0, 'dchi': 0, 'sperp2': 0})
test_samples.append({'m1': 1.58, 'm2': 0.457, 'chi_eff': 0.86, 's2_1': 0.916, 's2_2': 0.541, 'q1': 1.78, 'q2': 0.567, 'y': 0.99, 'e2': 0.97, 'dchi2': 0.95, 'dchi': 0.9, 'sperp2': 0.86})
test_samples.append({'m1': 1.58, 'm2': 0.457, 'chi_eff': 0.86, 's2_1': 0.916, 's2_2': 0.541, 'q1': 1.78, 'q2': 0.567, 'y': 0.2, 'e2': 0.97, 'dchi2': 0.95, 'dchi': 0.9, 'sperp2': 0.86})

#compute the PN derivatives for the different samples
for p in test_samples:
	
	#print the parameters
	print(p)
	
	#initialize class
	PN_derivatives = pyEFPE_PN_derivatives(p['m1'], p['m2'], p['chi_eff'], p['s2_1'], p['s2_2'], p['q1'], p['q2'])
	
	print(PN_derivatives.Dy_De2_Dl_Ddl(p['y'], p['e2'], p['dchi'], p['dchi2'], p['sperp2']))
	print()
