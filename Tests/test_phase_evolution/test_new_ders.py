import numpy as np
from tDomain_integrate_RR_funcs import pyEFPE_PN_derivatives_python as pyEFPE_PN_derivatives_old
from tDomain_integrate_RR_funcs import pyEFPE_PN_derivatives as pyEFPE_PN_derivatives_new
from matplotlib import pyplot as plt
plt.rcParams.update({'font.size': 24})
plt.rcParams.update({'lines.linewidth': 2})

#universal constants
t_sun = 4.92549094831e-6 #GMsun/c**3 [s]

import time
start_runtime = time.time()

#compute to compute and print relative errors
def print_rels_errs(x1, x2, name):
	dx = np.abs(0.5*(x1 - x2)/(x1 + x2))
	print('Difference in %s: MSE: %.3g   Max: %.3g '%(name, np.linalg.norm(dx)/np.sqrt(len(dx)), np.amax(dx)))
	return True


#####################################################

#Primary mass
m1 = 1*t_sun

#mass ratio
q = np.random.uniform(low=0, high=1)

#conserved part of the spins
s2_1 = np.random.uniform(low=0, high=1)
s2_2 = np.random.uniform(low=0, high=1)
chi_eff = np.random.uniform(low=-1, high=1)

#quadrupole parameters
q1 = np.random.uniform(low=0, high=1)
q2 = np.random.uniform(low=0, high=1)

#number of tests
Ntest = 100000

#####################################################

print('q=%.3g, chi_eff=%.3g, s2_1=%.3g, s2_2=%.3g, q1=%.3g, q2=%.3g'%(q, chi_eff, s2_1, s2_2, q1, q2))

#compute mass related stuff
m2 = q*m1

#compute random inputs
y = np.random.uniform(low=0, high=1, size=Ntest)
e2 = np.random.uniform(low=0, high=1, size=Ntest)
dchi = np.random.uniform(low=-1, high=1, size=Ntest)
dchi2 = np.random.uniform(low=0, high=1, size=Ntest)
sperp2 = np.random.uniform(low=0, high=1, size=Ntest)
try:
	#compute Dy, De2, D\lambda and D\delta\lambda with the old code
	start_soltime = time.time()
	PN_derivatives_old = pyEFPE_PN_derivatives_old(m1, m2, chi_eff, s2_1, s2_2, q1, q2)
	Dy_old, De2_old, Dl_old, Ddl_old = PN_derivatives_old.Dy_De2_Dl_Ddl(y, e2, dchi, dchi2, sperp2)
	print("\nTime to evaluate derivatives with old class (vectorized): %.3g seconds/call \n"%((time.time() - start_soltime)/Ntest))
except:
	print("\nCould not evaluate vectorized derivatives with old class")

try:
	#compute Dy, De2, D\lambda and D\delta\lambda with the new code
	start_soltime = time.time()
	PN_derivatives_new = pyEFPE_PN_derivatives_new(m1, m2, chi_eff, s2_1, s2_2, q1, q2)
	Dy_new, De2_new, Dl_new, Ddl_new = PN_derivatives_new.Dy_De2_Dl_Ddl(y, e2, dchi, dchi2, sperp2)
	print("\nTime to evaluate derivatives with new class (vectorized): %.3g seconds/call \n"%((time.time() - start_soltime)/Ntest))
except:
	print("\nCould not evaluate vectorized derivatives with new class")


Dy_old, De2_old, Dl_old, Ddl_old = np.zeros_like(y), np.zeros_like(y), np.zeros_like(y), np.zeros_like(y)
Dy_new, De2_new, Dl_new, Ddl_new = np.zeros_like(y), np.zeros_like(y), np.zeros_like(y), np.zeros_like(y)

#test timing on a loop
start_soltime = time.time()
for itest in range(Ntest):
	Dy_old[itest], De2_old[itest], Dl_old[itest], Ddl_old[itest] = PN_derivatives_old.Dy_De2_Dl_Ddl(y[itest], e2[itest], dchi[itest], dchi2[itest], sperp2[itest])
print("\nTime to evaluate derivatives with old class (in loop): %.3g seconds/call \n"%((time.time() - start_soltime)/Ntest))

#test timing on a loop
start_soltime = time.time()
for itest in range(Ntest):
	Dy_new[itest], De2_new[itest], Dl_new[itest], Ddl_new[itest] = PN_derivatives_new.Dy_De2_Dl_Ddl(y[itest], e2[itest], dchi[itest], dchi2[itest], sperp2[itest])
print("\nTime to evaluate derivatives with new class (in loop): %.3g seconds/call \n"%((time.time() - start_soltime)/Ntest))

#compute the difference
print_rels_errs(Dy_old, Dy_new, 'Dy')
print_rels_errs(De2_old, De2_new, 'De2')
print_rels_errs(Dl_old, Dl_new, 'Dl')
print_rels_errs(Ddl_old, Ddl_new, 'Ddl')



#Runtime
print("\nRuntime: %s seconds" % (time.time() - start_runtime))

plt.show()

