import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import cumulative_trapezoid
from scipy.special import gamma, hyp2f1, betainc, beta
from scipy.interpolate import interp1d

import time
start_runtime = time.time()

#function to compute the coefficients of the taylor series of (1 + a*x)^N
def taylor_power_series_coef(a, N, nmax):
	
	#compute the binomial coefficient iteratively
	coefs = np.ones(nmax)
	coefs[1] = a*N 
	for n in range(2, nmax):
		coefs[n] = coefs[n-1]*a*(N-n+1)/n

	return coefs

#compute the integral using hypergeometric functions
# F = (24/19)*x**(-24/19)*((1 + (121/304)*x)**(-3480/2299))*sqrt(1-x)*integral((x**(5/19))*((1 + (121/304)*x)**(1181/2299))*((1 - x)**(-3/2)))
def F_hypergeom_low_x(x, nmax=10):
	
	#compute coeficients of taylor series of (1 + (121/304)*x)**(1181/2299)
	cs = taylor_power_series_coef(121/304, 1181/2299, nmax)
	
	#compute the value of (1-x)**(1/2)x**(-24/19)*integral( c*(24/19)*x**(n+5/19)*(1-x)**(-3/2))
	integral = 0
	for n in range(nmax):
		integral = integral + cs[n]*(x**n)*hyp2f1(1, (29/38) + n, (43/19)+n, x)/(1 +  (19/24)*n)

	#multiply by extra factors of x and return 
	return integral*((1 + (121/304)*x)**(-3480/2299))
	
def F_hypergeom_high_x(x, nmax=10):
	
	#compute coeficients of taylor series of (1 - (121/425)*x)**(1181/2299)
	cs = taylor_power_series_coef(-121/425, 1181/2299, nmax)
	
	#compute u=1-x
	u = 1-x
	
	#compute the value of u**(1/2)*integral( c*(1-u)**(5/19)*u**(-3/2 + n))
	integral = 0
	for n in range(nmax):
		integral = integral + cs[n]*(u**n)*hyp2f1(-(5/19), n-0.5, n+0.5, u)/(n -  0.5)

	dintegral = -2.911033160643373 #=np.sum(cs*hyp2f1(-(5/19), np.arange(nmax)-0.5, np.arange(nmax)+0.5, 1)/(np.arange(nmax) -  0.5)) | nmax = 30

	#compute the integral taking into account that we want to start from x=0 (u=1)
	integral = dintegral*np.sqrt(1 - x) - integral

	#multiply by extra factors of x and return 
	return integral*(((425/304)**(1181/2299))*(24/19))*(x**(-24/19))*((1 + (121/304)*x)**(-3480/2299))

#put the two toguether
def F_hypergeom(x, nmax=10):
	
	#choose large x or small x at x*121/304 = (1-x)*121/425 -> x=304/729, each order will be suppresed by at most ~(121/729)**n
	#if each order contributes a!/(n!(a-n)!) b^n/(n+1), with a=1181/2299 and b=121/729, the precission can be well approximated as 10**-nmax, for nmax<15
	F = np.zeros_like(x)
	i_low = x<304/729
	i_high = np.logical_not(i_low)
	if sum(i_low)>0: F[i_low] = F_hypergeom_low_x(x[i_low], nmax=nmax)
	if sum(i_high)>0: F[i_high] = F_hypergeom_high_x(x[i_high], nmax=nmax)
	
	return F

#compute the tLO integral using beta functions
# F = (24/19)*x**(-24/19)*((1 + (121/304)*x)**(-3480/2299))*sqrt(1-x)*integral((x**(5/19))*((1 + (121/304)*x)**(1181/2299))*((1 - x)**(-3/2)))
def F_betas_high_x(x, nmax=10):
	
	#compute coeficients of taylor series of (1 - (121/425)*x)**(1181/2299)
	cs = taylor_power_series_coef(-121/425, 1181/2299, nmax)
	
	#compute the value of u**(1/2)*integral( c*(1-u)**(5/19)*u**(-3/2 + n)), for n>0
	integral=0
	for n in range(1, nmax):
		integral = integral + cs[n]*betainc(24/19,n-0.5,x)*beta(24/19,n-0.5)


	#compute the integral taking into account that we want to start from x=0 (u=1)
	dintegral = 2.6515364547943174 #= 2*hyp2f1(-(5/19), -0.5, 0.5, 1)
	integral = (integral - dintegral)*np.sqrt(1 - x) + 2*hyp2f1(-(5/19), -0.5, 0.5, 1 - x) 

	#multiply by extra factors of x and return 
	return integral*(((425/304)**(1181/2299))*(24/19))*(x**(-24/19))*((1 + (121/304)*x)**(-3480/2299))
	
#compute the series expansion of the tLO integral for x->0
# F = (24/19)*x**(-24/19)*((1 + (121/304)*x)**(-3480/2299))*sqrt(1-x)*integral((x**(5/19))*((1 + (121/304)*x)**(1181/2299))*((1 - x)**(-3/2)))
def F_series_at_0(x):
	coefs = np.array([1.000000000000000, -0.1511627906976744, 0.2656836084021005, 0.007463780007501875, 0.08800790590714085, 0.03153077124184580, 0.04185392210761341, 0.02761371124737777, 0.02642686119635599, 0.02145414169943131, 0.01926563742403364, 0.01677217450181226, 0.01503898450331388, 0.01347003088516929, 0.01220348405026613, 0.01110346195121149, 0.01016749330223870, 0.009352447459389354, 0.008642114232972108, 0.008016709107501086, 0.007463457161231162,0.006970902964332086, 0.006530253812462707, 0.006134111124121483, 0.005776451398258840, 0.005452229768143720, 0.005157232928828976, 0.004887901117379935, 0.004641215172072290, 0.004414596725230578,0.004205833180162198, 0.004013015784562471, 0.003834490347048246,0.003668816871424952, 0.003514736646109113, 0.003371145103099870, 0.003237069368178693, 0.003111649567641214, 0.002994123194217794, 0.002883811964918853, 0.002780110725151045, 0.002682478039498533,0.002590428180410570, 0.002503524280447905, 0.002421372457429333,0.002343616756405161, 0.002269934780185545, 0.002200033902499887, 0.002133647975961232, 0.002070534461714599, 0.002010471919657125])
	
	return np.polyval(np.flip(coefs),x)

#compute the series expansion of the tLO integral for x->1
def F_series_at_1(x):
	
	#we are going to approximate the integral of ((1 - 1/u**2)**(5/19))*(1 - (121/425)/u**2)**(1181/2299) - 1 as -f0 + sum_{n=1}^{nmax} cn*u**-(2*n-1)
	cns = np.array([0.40941176470588236, 0.02286320645905421, 0.008142925951557094, 0.004155878512401501, 0.0024847568765827364, 0.001633290511588348, 0.0011455395465032605, 0.000842577094447224, 0.0006427195446513564, 0.0005045715814217113, 0.00040544477831148924, 0.0003321116545345162, 0.0002764636962467965, 0.00023331895675436905, 0.0001992473575067662, 0.00017190919726595898, 0.00014966654751355843, 0.000131346469484909, 0.00011609207361015848, 0.00010326617525262309, 9.238740896587563e-05, 8.308691874613337e-05, 7.50784086790562e-05, 6.813705814151314e-05, 6.20844345948999e-05, 5.677753690123865e-05, 5.210072977646673e-05, 4.7959732146031274e-05, 4.427708468062032e-05, 4.0988696117937945e-05, 3.80411855904995e-05, 3.538981870077757e-05, 3.2996890965966614e-05, 3.083045152872919e-05, 2.886328796018351e-05, 2.707211306399751e-05, 2.543690918050699e-05, 2.3940396192787112e-05, 2.256759736012561e-05, 2.1305483020854193e-05, 2.014267666038337e-05, 1.906921121897629e-05, 1.8076326095558655e-05, 1.7156297290322297e-05, 1.6302294667351972e-05, 1.550826151746742e-05, 1.4768812541410176e-05, 1.407914711455732e-05])
	
	#evaluate polynomial
	u = 1-x
	return ((48/19)*((425/304)**(1181/2299)))*(1 - 1.4555165803216864*np.sqrt(u) + u*np.polyval(np.flip(cns),u))*(x**(-24/19))*((1 + (121/304)*x)**(-3480/2299)) 

#function to choose series at x=0 or at x=1
def F_series(x):
	
	#if x small, use series expansion at 0, otherwise use series expansion at 1
	F = np.zeros_like(x)
	i_low = x<0.4
	i_high = np.logical_not(i_low)
	if sum(i_low)>0:  F[i_low] = F_series_at_0(x[i_low])
	if sum(i_high)>0: F[i_high] = F_series_at_1(x[i_high])
	
	#return F
	return F

def F_betas(x, nmax=20):
	
	#if x is very small, do taylor expansion, otherwise use beta functions
	F = np.zeros_like(x)
	i_low = x<0.4
	i_high = np.logical_not(i_low)
	if sum(i_low)>0:  F[i_low] = F_series_at_0(x[i_low])
	if sum(i_high)>0: F[i_high] = F_betas_high_x(x[i_high], nmax=nmax)
	
	#return F
	return F


#approximate the integral ((1 - 1/u**2)**(5/19))*(1 - (121/425)/u**2)**(1181/2299) - 1 at large u
def fast_tNLO_int_large_u(u):
	
	#we are going to approximate the integral as -f0 + sum_{n=1}^{nmax} cn*u**-(2*n-1)
	cns = np.array([0.40941176470588236, 0.02286320645905421, 0.008142925951557094, 0.004155878512401501, 0.0024847568765827364, 0.001633290511588348, 0.0011455395465032605, 0.000842577094447224, 0.0006427195446513564, 0.0005045715814217113, 0.00040544477831148924, 0.0003321116545345162, 0.0002764636962467965, 0.00023331895675436905, 0.0001992473575067662, 0.00017190919726595898, 0.00014966654751355843, 0.000131346469484909, 0.00011609207361015848, 0.00010326617525262309, 9.238740896587563e-05, 8.308691874613337e-05, 7.50784086790562e-05, 6.813705814151314e-05, 6.20844345948999e-05, 5.677753690123865e-05, 5.210072977646673e-05, 4.7959732146031274e-05, 4.427708468062032e-05, 4.0988696117937945e-05, 3.80411855904995e-05, 3.538981870077757e-05, 3.2996890965966614e-05, 3.083045152872919e-05, 2.886328796018351e-05, 2.707211306399751e-05, 2.543690918050699e-05, 2.3940396192787112e-05, 2.256759736012561e-05, 2.1305483020854193e-05, 2.014267666038337e-05, 1.906921121897629e-05, 1.8076326095558655e-05, 1.7156297290322297e-05, 1.6302294667351972e-05, 1.550826151746742e-05, 1.4768812541410176e-05, 1.407914711455732e-05])
	
	#evaluate polynomial
	return np.polyval(np.flip(cns),u**-2)/u -0.4555165803216864

#compute the exact integral we want to approximate
x = np.linspace(0, 1, 100001)

#time F_hypergeom integral
start_soltime = time.time()
F_exact = F_hypergeom(x, nmax=25)
print("Analytical hypergeometrical runtime: %s seconds" % (time.time() - start_soltime))

#time F_hypergeom integral
start_soltime = time.time()
F_exact_betas = F_betas(x, nmax=14)
print("Analytical betas runtime: %s seconds" % (time.time() - start_soltime), "Max error:", np.amax(np.abs(F_exact - F_exact_betas)), "at x=", x[np.argmax(np.abs(F_exact - F_exact_betas))])

#time F_hypergeom integral
start_soltime = time.time()
F_exact_series = F_series(x)
print("Analytical series runtime: %s seconds" % (time.time() - start_soltime), "Max error:", np.amax(np.abs(F_exact - F_exact_series)), "at x=", x[np.argmax(np.abs(F_exact - F_exact_series))])


start_soltime = time.time()
F_exact_low = F_hypergeom_low_x(x, nmax=16)
print("Analytical low x integration runtime: %s seconds" % (time.time() - start_soltime))

start_soltime = time.time()
F_exact_high = F_hypergeom_high_x(x, nmax=16)
print("Analytical high x integration runtime: %s seconds" % (time.time() - start_soltime), "\n")


#compute integrand
integrand = (24/19)*(x**(5/19))*((1 + (121/304)*x)**(1181/2299))*((1 - x)**(-3/2))
tNLO = cumulative_trapezoid(integrand, x, initial=0)

#compute y
y = (x**(-3/19))*((1 + (121/304)*x)**(-435/2299))

taylor_series = 1 - (13/86)*x + (11333/42656)*(x**2)+ (2547/341248)*(x**3) + (9610407/109199360)*(x**4) + (1638934659/51978895360)*(x**5)

#compute u
u = 1/np.sqrt(1 - x)
print(fast_tNLO_int_large_u(u))
normed_tNLO_high_x = np.where(x>0.2075,(y**8)*(48/19)*((425/304)**(1181/2299))*(1 - np.sqrt(1 - x)*(1 - fast_tNLO_int_large_u(u))), taylor_series)

plt.figure(figsize=(12,8))
plt.plot(np.sqrt(x), F_exact)
plt.xlim(np.sqrt(x[0]), np.sqrt(x[-1]))
plt.xlabel('$e_0$')
plt.ylabel('$F(e_0)$')
plt.tight_layout()
plt.savefig('Fe_plot.pdf')


plt.figure(figsize=(12,8))
plt.plot(x, np.where(x>0.1, tNLO*(y**8)*(1-x)**(1/2), taylor_series), label='exact')
plt.plot(x, normed_tNLO_high_x,'--', label='high x approx')
plt.plot(x, F_series_at_0(x), label='Taylor')
plt.plot(x, F_hypergeom_low_x(x), label='Hypergeometric, low x')
plt.plot(x, F_hypergeom_high_x(x), label='Hypergeometric, high x')
plt.plot(x, F_exact, label='Hypergeometric')
plt.plot(x, F_betas_high_x(x), label='Betas' )
#plt.plot(x, np.abs(np.where(x>0.1, tNLO*(y**8)*(1-x)**(1/2), taylor_series)-normed_tNLO_high_x), label='exact')
#plt.yscale('log')
plt.xlim(x[0], x[-1])
plt.xlabel('$x$')
plt.grid(True)
plt.tight_layout()
plt.legend()

plt.figure(figsize=(12,8))
plt.plot(x, F_exact*(1 + 1.4555165803216864*np.sqrt(1-x)), label='Opt 1')
plt.plot(x, F_exact*(1 + 1.4555165803216864*(np.sqrt(1-x) - 0.4455172586553138*(1-x))), label='Opt 2')
plt.plot(x, F_exact*(1 + 1.4555165803216864*(np.sqrt(1-x) - 0.6558741097838263*(1-x))), label='Opt 3')
plt.plot(x, F_exact , label='exact')
#plt.plot(x, (768/425)/(1 + 1.4555165803216864*np.sqrt(1-x)), '--' , label='Opt 1')
#plt.plot(x, (768/425)/(1 + 1.4555165803216864*(np.sqrt(1-x) - 0.5*(1-x))), ':' , label='Opt 2')
plt.xlim(x[0], x[-1])
plt.xlabel('$x$')
plt.grid(True)
plt.tight_layout()
plt.legend()

#fit F using option 2
Nleg = 10
legfit = np.polynomial.legendre.legfit(x, F_exact*(1 + 1.4555165803216864*(np.sqrt(1-x) - 0.4455172586553138*(1-x))), Nleg)
poly = np.polynomial.legendre.leg2poly(legfit)
print(legfit)
print(poly.tolist(), '\n')
start_soltime = time.time()
F_poly = np.polyval(np.flip(poly),x)/(1 + 1.4555165803216864*(np.sqrt(1-x) - 0.4455172586553138*(1-x)))
print('Error of legendre polynomial:', np.amax(np.abs(F_poly - F_exact)), '  Runtime:',time.time() - start_soltime,'s')

#make an interpolation of F
n_nodes = 100
x_nodes = np.linspace(0, 1, n_nodes+1)
F_mod_nodes = F_hypergeom(x_nodes, nmax=16)*(1 + 1.4555165803216864*(np.sqrt(1-x_nodes) - 0.4455172586553138*(1-x_nodes)))

#linear interpolation
start_soltime = time.time()
F_interp = np.interp(x, x_nodes, F_mod_nodes)/(1 + 1.4555165803216864*(np.sqrt(1-x) - 0.4455172586553138*(1-x)))
print('Error of linear interpolation:', np.amax(np.abs(F_exact - F_interp)), '  Runtime:',time.time() - start_soltime,'s')
#cubic interpolation
start_soltime = time.time()
F_mod_interp_cubic_func = interp1d(x_nodes, F_mod_nodes, kind='cubic')
F_interp_cubic = F_mod_interp_cubic_func(x)/(1 + 1.4555165803216864*(np.sqrt(1-x) - 0.4455172586553138*(1-x)))
print('Error of cubic interpolation:', np.amax(np.abs(F_exact - F_interp_cubic)), '  Runtime:',time.time() - start_soltime,'s')


plt.show()


