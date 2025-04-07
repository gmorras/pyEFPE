import numpy as np
import pyximport; pyximport.install(setup_args={"script_args": ["--force"]}, language_level=3)
from pyEFPE_cython_functions import my_cpoly

#function to compute all mass parameters from m1, m2
def mass_params_from_m1_m2(m1, m2):

	#check that m1>m2
	if m1<m2: print('Warning: m1<m2')

	#now compute mass related stuff
	M = m1 + m2           #total mass
	mu1, mu2 = m1/M, m2/M #reduced individual masses
	nu = mu1*mu2          #symmetric mass ratio 
	dmu = mu1 - mu2       #dimensionless mass diference

	return M, mu1, mu2, nu, dmu	

#class to compute the precession averaged value of Dy, D(e^2), D\lambda and D\delta\lambda to 3PN in non-spining and aligned-spin and 2PN in fully spinning using the PN formulas derived in our EFPE paper. We can select the n-th PN order of the spinning/non-spinnig part with pn_phase_order/pn_spin_order = 2*n. If pn_xxxx_order=-1, set it to the maximum.
class pyEFPE_PN_derivatives:

	#initialize computing the constants that depend on nu
	def __init__(self, m1, m2, chi_eff, s2_1, s2_2, q1, q2, pn_phase_order=6, pn_spin_order=6):
		
		#save pn orders, taking into account that an order of -1 means to take the maximum order
		if pn_phase_order==-1: self.pn_phase_order = 6
		else:                  self.pn_phase_order = pn_phase_order

		if pn_spin_order==-1: self.pn_spin_order = 6
		else:                 self.pn_spin_order = pn_spin_order
		
		self.pn_max_order = max(self.pn_phase_order, self.pn_spin_order)
		
		#if we are requesting a pn order higher than what is implemented, throw a Warning
		if self.pn_phase_order>6: print('Warning: pn_phase_order>6 not implemented. Input pn_phase_order: %s'%(self.pn_phase_order))
		if self.pn_spin_order>6: print('Warning: pn_spin_order>6 not implemented. Input pn_spin_order: %s'%(self.pn_spin_order))
		
		#compute mass related stuff
		M, mu1, mu2, nu, dmu = mass_params_from_m1_m2(m1, m2)
		nu2 = nu*nu
		nu3 = nu2*nu
		pi2 = np.pi*np.pi
		self.nu = nu
		
		#compute symetric and antisymetric combinations of quadrupole parameters
		dqS = q1 + q2 - 2
		dqA = q1 - q2
		dqAdmu = dqA*dmu
		
		#compute spin related stuff
		s2iS =  s2_1 + s2_2
		s2iA =  s2_1 - s2_2
		chi_eff2 = chi_eff*chi_eff
		
		#store the e^{2n} coefficients that enter the tail terms of Eq (C4) of 1801.08542
		c_phiy  = np.array([1., 97./32., 49./128., -49./18432., -109./147456., -2567./58982400.])
		c_phie  = np.array([1., 5969./3940., 24217./189120., 623./4538880., -96811./363110400., -5971./4357324800.])
		c_psiy  = np.array([1., -207671./8318., -8382869./266176., -8437609./4791168., 10075915./306634752., -38077159./15331737600.])
		c_zetay = np.array([1., 113002./11907., 6035543./762048., 253177./571536., -850489./877879296., -1888651./10973491200.])
		c_psie  = np.array([1., -9904271./891056., -101704075./10692672., -217413779./513248256., 35703577./6843310080., -3311197679./9854366515200.])
		c_zetae = np.array([1., 11228233./2440576., 37095275./14643456., 151238443./1405771776., -118111./611205120., -407523451./26990818099300.])
		c_kappay = 244*np.log(2.)*np.array([0., 1., -18881./1098., 6159821./39528., -16811095./19764., 446132351./123525.])-243*np.log(3.)*np.array([0., 1., -39./4., 2735./64., 25959./512., -638032239./409600.])-(48828125*np.log(5.)/5184.)*np.array([0., 0., 0., 1., -83./8., 12637./256.]) -(4747561509943.*np.log(7.)/33177600.)*np.array([0., 0., 0., 0., 0., 1.])
		c_kappae = 6536*np.log(2.)*np.array([1., -22314./817., 7170067./19608., -10943033./4128., 230370959./15480., -866124466133./8823600.])-6561*np.log(3.)*np.array([1., -49./4., 4369./64., 214449./512., -623830739./81920., 76513915569./1638400.])-(48828125.*np.log(5.)/64.)*np.array([0., 0., 1., -293./24., 159007./2304., -6631171./27648.])-(4747561509943.*np.log(7.)/245760.)*np.array([0.,0.,0.,0.,1.,-259./20.])

		#store also the e^{2n} coefficients in the Spin-Orbit tail-terms 
		c_thyc = np.array([1., 21263./3008., 52387./12032., 253973./1732608., -82103./13860864.])
		c_thyd = np.array([1., 1897./592., -461./2368., -42581./340992., -3803./1363968.])
		c_thec = np.array([1., 377077./92444., 7978379./4437312., 5258749./106495488.])
		c_thed = np.array([1., 37477./19748., 95561./947904., -631523./22749696.])

		#now store the e^{2n} coefficients appearing in the non-spinning part of dy/dt
		self.p_a0NS = my_cpoly(np.array([32./5., 28./5.]))
		
		self.p_a2NS = my_cpoly(np.array([-1486./105. - (88./5.)*nu, 12296./105. - (5258./45.)*nu, 3007./84. - (244./9.)*nu]))
		
		self.p_a3NS = my_cpoly((128./5.)*np.pi*c_phiy)
		
		self.p_a4NS = my_cpoly(np.array([34103./2835. + (13661./315.)*nu + (944./45.)*nu2, -489191./1890. - (209729./630.)*nu + (147443./270.)*nu2, 2098919./7560. - (2928257./2520.)*nu + (34679./45.)*nu2, 53881./2520. - (7357./90.)*nu + (9392./135.)*nu2]))
		self.sqrt_a4NS = my_cpoly(np.array([16. - (32./5.)*nu, 266. - (532./5.)*nu, -859./2. + (859./5.)*nu, -65. + 26*nu]))
		
		self.p_a5NS = my_cpoly(np.pi*(-(4159./105.)*c_psiy-(756./5.)*nu*c_zetay))
		
		self.p_a6NS = my_cpoly(np.array([16447322263./21829500. - (54784./525.)*np.euler_gamma + (512./15.)*pi2 + (-(56198689./34020.) + (902./15.)*pi2)*nu + (541./140.)*nu2 - (1121./81.)*nu3, 33232226053./10914750. - (392048./525.)*np.euler_gamma + (3664./15.)*pi2 + (-(588778./1701.) + (2747./40.)*pi2)*nu - (846121./1260.)*nu2 - (392945./324.)*nu3, -227539553251./58212000. - (93304./175.)*np.euler_gamma + (872./5.)*pi2 + ((124929721./12960.) - (41287./960.)*pi2)*nu + (148514441./30240.)*nu2 - (2198212./405.)*nu3, -300856627./67375. - (4922./175.)*np.euler_gamma + (46./5.)*pi2 + ((1588607./432.) - (369./80.)*pi2)*nu + (12594313./3780.)*nu2 - (44338./15.)*nu3, -243511057./887040. + (4179523./15120.)*nu + (83701./3780.)*nu2 - (1876./15.)*nu3, 0.]) + (1284./175.)*c_kappay)
		self.sqrt_a6NS = my_cpoly(np.array([-616471./1575. + ((9874./315.)- (41./30.)*pi2)*nu + (632./15.)*nu2, 2385427./1050. + (-(274234./45.) + (4223./240.)*pi2)*nu + (70946./45.)*nu2, 8364697./4200. + ((1900517./630.) - (32267./960.)*pi2)*nu - (47443./90.)*nu2, -167385119./25200. + ((4272491./504.) - (123./160.)*pi2)*nu - (43607./18.)*nu2, -65279./168. + (510361./1260.)*nu - (5623./45.)*nu2]))
		self.log_a6NS = my_cpoly(np.array([54784./525., 392048./525., 93304./175., 4922./175.]))

		#now store the e^{2n} coefficients appearing in the non-spinning part of d(e^2)/dt
		self.p_b0NS = my_cpoly(np.array([608./15., 242./15.]))
		
		self.p_b2NS = my_cpoly(np.array([-1878./35. - (8168./45.)*nu, 59834./105. - (7753./15.)*nu, 13929./140. - (3328./45.)*nu]))
		
		self.p_b3NS = my_cpoly((788./3.)*np.pi*c_phie)
		
		self.p_b4NS = my_cpoly(np.array([-949877./945. + (18763./21.)*nu + (1504./5.)*nu2, -3082783./1260. - (988423./420.)*nu + (64433./20.)*nu2, 23289859./7560. - (13018711./2520.)*nu + (127411./45.)*nu2, 420727./1680. - (362071./1260.)*nu + (1642./9.)*nu2]))
		self.sqrt_b4NS = my_cpoly(np.array([2672./3. - (5344./15.)*nu, 2321. - (4642./5.)*nu, 565./3. - (226./3.)*nu]))
		
		self.p_b5NS = my_cpoly(np.pi*(-(55691./105.)*c_psie-(610144./315.)*nu*c_zetae))

		self.p_b6NS = my_cpoly(np.array([61669369961./4365900. - (2633056./1575.)*np.euler_gamma + (24608./45.)*pi2 + ((50099023./56700.) + (779./5.)*pi2)*nu - (4088921./1260.)*nu2 - (61001./243.)*nu3, 66319591307./21829500. - (9525568./1575.)*np.euler_gamma + (89024./45.)*pi2 + ((28141879./450.) - (139031./480.)*pi2)*nu - (21283907./1512.)*nu2 - (86910509./9720.)*nu3, -1149383987023./58212000. - (4588588./1575.)*np.euler_gamma + (42884./45.)*pi2 + ((11499615139./453600.) - (271871./960.)*pi2)*nu + (61093675./2016.)*nu2 - (2223241./90.)*nu3, 40262284807./4312000. - (20437./175.)*np.euler_gamma + (191./5.)*pi2 + (-(5028323./280.) - (6519./320.)*pi2)*nu + (24757667./1260.)*nu2 - (11792069./1215.)*nu3, 302322169./887040. - (1921387./5040.)*nu + (41179./108.)*nu2 - (386792./1215.)*nu3, 0.]) + (428./1575.)*c_kappae)
		self.sqrt_b6NS = my_cpoly(np.array([-22713049./7875. + (-(11053982./945.) + (8323./90.)*pi2)*nu + (108664./45.)*nu2, 178791374./7875. + (-(38295557./630.) + (94177./480.)*pi2)*nu + (681989./45.)*nu2, 5321445613./189000. + (-(26478311./756.) + (2501./1440.)*pi2)*nu + (450212./45.)*nu2, 186961./168. - (289691./252.)*nu + (3197./9.)*nu2]))
		self.one_m_sqrt_b6NS = 1460336./23625
		self.log_b6NS = my_cpoly(np.array([2633056./1575., 9525568./1575., 4588588./1575., 20437./175.]))
		
		#now store the e^{2n} coefficients appearing in the Spin-Orbit part of dy/dt
		self.chi_p_a3SO = my_cpoly(chi_eff*np.array([-752./15., -138., -611./30.]))
		self.dch_p_a3SO = my_cpoly(dmu*np.array([-152./15., -154./15., 17./30.]))
		
		self.chi_p_a5SO = my_cpoly(chi_eff*np.array([-5861./45. + (4004./15.)*nu, -968539./630. + (259643./135.)*nu, -4856917./2520. + (943721./540.)*nu, -64903./560. + (5081./45.)*nu]))
		self.chi_e2sqrt_a5SO = my_cpoly(chi_eff*np.array([-1416./5. + (1652./15.)*nu, 2469./5. - (5761./30.)*nu, 222./5. - (259./15.)*nu]))
		
		self.dch_p_a5SO = my_cpoly(dmu*np.array([-21611./315. + (632./15.)*nu, -55415./126. + (36239./135.)*nu, -72631./360. + (12151./108.)*nu, 909./560. - (143./45.)*nu]))
		self.dch_e2sqrt_a5SO = my_cpoly(dmu*np.array([-472./5. + (236./15.)*nu, 823./5. - (823./30.)*nu, 74./5. - (37./15.)*nu]))
		
		chi_p_a6SO_arr = -(3008./15.)*np.pi*chi_eff*c_thyc
		dch_p_a6SO_arr = -(592./15.)*np.pi*dmu*c_thyd

		#now store the e^{2n} coefficients appearing in the Spin-Orbit part of d(e^2)/dt
		self.chi_p_b3SO = my_cpoly(chi_eff*np.array([-3272./9., -26263./45., -812./15.]))
		self.dch_p_b3SO = my_cpoly(dmu*np.array([-3328./45., -1993./45., 23./15.]))
		
		self.chi_p_b5SO = my_cpoly(chi_eff*np.array([-13103./35. + (289208./135.)*nu, -548929./63. + (61355./6.)*nu, -6215453./840. + (1725437./270.)*nu, -87873./280. + (13177./45.)*nu]))
		self.chi_sqrt_b5SO = my_cpoly(chi_eff*np.array([-1184. + (4144./9.)*nu, -13854./5. + (16163./15.)*nu, -626./5. + (2191./45.)*nu]))
		
		self.dch_p_b5SO = my_cpoly(dmu*np.array([-32857./105. + (52916./135.)*nu, -1396159./630. + (126833./90.)*nu, -203999./280. + (56368./135.)*nu, 5681./1120. - (376./45.)*nu]))
		self.dch_sqrt_b5SO = my_cpoly(dmu*np.array([-1184./3. + (592./9.)*nu, -4618./5. + (2309./15.)*nu, -626./15. + (313./45.)*nu]))
		
		chi_p_b6SO_arr = -(92444./45.)*np.pi*chi_eff*c_thec
		dch_p_b6SO_arr = -(19748./45.)*np.pi*dmu*c_thed

		#now store the e^{2n} coefficients appearing in the precession averaged Spin-Spin part of dy/dt
		c_s2iS_a4SS = s2iS*np.array([8./5. - 8*dqS, 24./5. - (108./5.)*dqS, 3./5. - (63./20.)*dqS])
		c_s2iA_a4SS = dqA*s2iA*np.array([-8., -108./5., -63./20.])
		c_chi2_a4SS = chi_eff2*np.array([156./5. + 12*dqS, 84. + (162./5.)*dqS, 123./10. + (189./40.)*dqS])
		self.const_a4SS = my_cpoly(c_s2iS_a4SS + c_s2iA_a4SS + c_chi2_a4SS)
		self.sperp2_a4SS = my_cpoly(np.array([-84./5., -228./5., -33./5.]))
		self.chidch_a4SS = my_cpoly(chi_eff*dqA*np.array([24., 324./5., 189./20.]))
		self.dch2_a4SS = my_cpoly(np.array([-2./5. + 12*dqS, -6./5. + (162./5.)*dqS, -3./20. + (189./40.)*dqS]))
		
		chi2_p_a6SS_arr = chi_eff2*np.array([30596./105. + (2539./105.)*dqS + (443./30.)*dqAdmu +  (-(688./5.) - (172./5.)*dqS)*nu, 115078./45. + (21317./60.)*dqS + (3253./60.)*dqAdmu + (-(3962./3.) - (1981./6.)*dqS)*nu, 4476649./2520. + (133703./420.)*dqS + (481./48.)*dqAdmu + (-(53267./45.) - (53267./180.)*dqS)*nu, 17019./140. + (29831./1120.)*dqS + (29./160.)*dqAdmu + (-(1343./15.) - (1343./60.)*dqS)*nu, 0.])
		self.chi2_sqrt_a6SS = my_cpoly(chi_eff2*np.array([-(244./15.) - (52./15.)*dqS - (4./15.)*dqAdmu + (16./5. + (4./5.)*dqS)*nu, 6283./30. + (1339./30.)*dqS + (103./30.)*dqAdmu + (-(206./5.) - (103./10.)*dqS)*nu, -(48007./120.) - (10231./120.)*dqS - (787./120.)*dqAdmu + (787./10. + (787./40.)*dqS)*nu, -(183./20.) - (39./20.)*dqS - (3./20.)*dqAdmu + (9./5. + (9./20.)*dqS)*nu]))
		chidch_p_a6SS_arr = chi_eff*np.array([(3134./15. + (443./15.)*dqS)*dmu + (5078./105. - (344./5.)*nu)*dqA, (30421./45. + (3253./30.)*dqS)*dmu + (21317./30. - (1981./3.)*nu)*dqA, (-(111./5.) + (481./24.)*dqS)*dmu + (133703./210. - (53267./90.)*nu)*dqA, (-(149./40.) + (29./80.)*dqS)*dmu + (29831./560. - (1343./30.)*nu)*dqA, 0.])
		self.chidch_sqrt_a6SS = my_cpoly(chi_eff*np.array([(-(104./15.) - (8./15.)*dqS)*dmu + (-(104./15.) + (8./5.)*nu)*dqA, (1339./15. + (103./15.)*dqS)*dmu + (1339./15. - (103./5.)*nu)*dqA, (-(10231./60.) - (787./60.)*dqS)*dmu + (-(10231./60.) + (787./20.)*nu)*dqA, (-(39./10.) - (3./10.)*dqS)*dmu + (-(39./10.) + (9./10.)*nu)*dqA]))
		self.dch2_p_a6SS = my_cpoly(np.array([39./5. + (2539./105.)*dqS + (443./30.)*dqAdmu + (-(1163./15.) - (172./5.)*dqS)*nu, 659./15. + (21317./60.)*dqS + (3253./60.)*dqAdmu + (-(2399./15.) - (1981./6.)*dqS)*nu, 1769./90. + (133703./420.)*dqS + (481./48.)*dqAdmu + (2021./72. - (53267./180.)*dqS)*nu, 19./10. + (29831./1120.)*dqS + (29./160.)*dqAdmu + (-(3./10.) - (1343./60.)*dqS)*nu]))
		self.dch2_sqrt_a6SS = my_cpoly(np.array([-(4./15.) - (52./15.)*dqS - (4./15.)*dqAdmu + (32./15. + (4./5.)*dqS)*nu, 103./30. + (1339./30.)*dqS + (103./30.)*dqAdmu + (-(412./15.) - (103./10.)*dqS)*nu, -(787./120.) - (10231./120.)*dqS - (787./120.)*dqAdmu +  (787./15. + (787./40.)*dqS)*nu, -(3./20.) - (39./20.)*dqS - (3./20.)*dqAdmu + (6./5. + (9./20.)*dqS)*nu]))
		
		#now store the e^{2n} coefficients appearing in the precession averaged Spin-Spin part of d(e^2)/dt
		c_s2iS_b4SS = s2iS*np.array([-4./3., 34./3. - (938./15.)*dqS, 49./2. - (595./6.)*dqS, 9./4. - (37./4.)*dqS])
		c_s2iA_b4SS = dqA*s2iA*np.array([0., -938./15., -595./6., -37./4.])
		c_chi2_b4SS = chi_eff2*np.array([2./3., 3667./15. + (469./5.)*dqS, 4613./12. + (595./4.)*dqS, 287./8. + (111./8.)*dqS])
		self.const_b4SS = my_cpoly(c_s2iS_b4SS + c_s2iA_b4SS + c_chi2_b4SS)
		self.sperp2_b4SS = my_cpoly(np.array([2./3., -1961./15., -2527./12., -157./8.]))
		self.chidch_b4SS = my_cpoly(chi_eff*dqA*np.array([0., 938./5., 595./2., 111./4.]))
		self.dch2_b4SS = my_cpoly(np.array([2./3., 1./3. + (469./5.)*dqS, -13./4. + (595./4.)*dqS, -3./8. + (111./8.)*dqS]))
		
		chi2_p_b6SS_arr = chi_eff2*np.array([1468414./945. + (2852./105.)*dqS + (3461./30.)*dqAdmu + (-(57844./45.) - (14461./45.)*dqS)*nu, 47715853./3780. + (1464091./840.)*dqS + (11007./40.)*dqAdmu + (-(21865./3.) - (21865./12.)*dqS)*nu, 4255831./504. + (166844./105.)*dqS + (2941./48.)*dqAdmu + (-(222533./45.) - (222533./180.)*dqS)*nu, 414027./1120. + (365363./4480.)*dqS + (511./640.)*dqAdmu + (-(1287./5.) - (1287./20.)*dqS)*nu])
		self.chi2_sqrt_b6SS = my_cpoly(chi_eff2*np.array([49532./45. + (10556./45.)*dqS + (812./45.)*dqAdmu + (-(3248./15.) - (812./15.)*dqS)*nu, 140117./60. + (29861./60.)*dqS + (2297./60.)*dqAdmu + (-(2297./5.) - (2297./20.)*dqS)*nu, 3721./180. + (793./180.)*dqS + (61./180.)*dqAdmu + (-(61./15.) - (61./60.)*dqS)*nu]))
		chidch_p_b6SS_arr = chi_eff*np.array([(176426./135. + (3461./15.)*dqS)*dmu + (5704./105. - (28922./45.)*nu)*dqA, (387212./135. + (11007./20.)*dqS)*dmu + (1464091./420. - (21865./6.)*nu)*dqA, (2562./5. + (2941./24.)*dqS)*dmu + (333688./105. - (222533./90.)*nu)*dqA, (-(33./32.) + (511./320.)*dqS)*dmu + (365363./2240. - (1287./10.)*nu)*dqA])
		self.chidch_sqrt_b6SS = my_cpoly(chi_eff*np.array([(21112./45. + (1624./45.)*dqS)*dmu + (21112./45. - (1624./15.)*nu)*dqA, (29861./30. + (2297./30.)*dqS)*dmu + (29861./30. - (2297./10.)*nu)*dqA, (793./90. + (61./90.)*dqS)*dmu + (793./90. - (61./30.)*nu)*dqA]))
		self.dch2_p_b6SS = my_cpoly(np.array([8887./135. + (2852./105.)*dqS + (3461./30.)*dqAdmu + (-(13127./27.) - (14461./45.)*dqS)*nu, 161077./540. + (1464091./840.)*dqS + (11007./40.)*dqAdmu + (-(185723./270.) - (21865./12.)*dqS)*nu, 14827./90. + (166844./105.)*dqS + (2941./48.)*dqAdmu + (-(45373./360.) - (222533./180.)*dqS)*nu, 283./32. + (365363./4480.)*dqS + (511./640.)*dqAdmu + (-(117./20.) - (1287./20.)*dqS)*nu]))
		self.dch2_sqrt_b6SS = my_cpoly(np.array([812./45. + (10556./45.)*dqS + (812./45.)*dqAdmu + (-(6496./45.) - (812./15.)*dqS)*nu, 2297./60. + (29861./60.)*dqS + (2297./60.)*dqAdmu + (-(4594./15.) - (2297./20.)*dqS)*nu, 61./180. + (793./180.)*dqS + (61./180.)*dqAdmu + (-(122./45.) - (61./60.)*dqS)*nu]))
		
		#join the parts of a6 and b6 that do not depend on dchi
		self.const_p_a6SOSS = my_cpoly(chi_p_a6SO_arr + chi2_p_a6SS_arr)
		self.const_p_b6SOSS = my_cpoly(chi_p_b6SO_arr + chi2_p_b6SS_arr)
		
		#join the parts of a6 and b6 that are proportional to dchi
		self.dch_p_a6SOSS = my_cpoly(dch_p_a6SO_arr + chidch_p_a6SS_arr)
		self.dch_p_b6SOSS = my_cpoly(dch_p_b6SO_arr + chidch_p_b6SS_arr)
		
		#store the e^{2n} coefficients of the non-spinning part of the periastron precession k
		self.k0NS = 3
		
		self.p_k2NS = my_cpoly(np.array([27./2. - 7*nu, 51./4. - (13./2.)*nu]))
		
		self.p_k4NS = my_cpoly(np.array([105./2. + (-(625./4.) + (123./32.)*pi2)*nu + 7*nu2, 573./4. + (-(357./2.) + (123./128.)*pi2)*nu + 40*nu2, 39./2. - (55./4.)*nu + (65./8.)*nu2]))
		self.sqrt_k4NS = my_cpoly(np.array([15. - 6*nu, 30. - 12*nu]))
		
		#store the e^{2n} coefficients of the spin-orbit part of the periastron precession k
		self.chi_k1SO = -(7./2.)*chi_eff
		self.dch_k1SO = -(1./2.)*dmu
		
		self.chi_p_k3SO = my_cpoly(chi_eff*np.array([-26. + 8*nu, -(105./4.) + (49./4.)*nu]))
		self.dch_p_k3SO = my_cpoly(dmu*np.array([-8. + (1./2.)*nu, -(15./4.) + (7./4.)*nu]))
		
		#store the e^{2n} coefficients appearing in the precession averaged Spin-Spin part of the periastron precession k
		c_s2iS_k2SS = -(3./8.)*dqS*s2iS
		c_s2iA_k2SS = -(3./8.)*dqA*s2iA
		c_chi2_k2SS = (3./2. + (9./16.)*dqS)*chi_eff2
		self.const_k2SS = c_s2iS_k2SS + c_s2iA_k2SS + c_chi2_k2SS
		self.sperp2_k2SS = -(3./4.)
		self.chidch_k2SS = (9./8.)*dqA*chi_eff
		self.dch2_k2SS = (9./16.)*dqS
		
		self.chi2_p_k4SS = my_cpoly(chi_eff2*np.array([181./8. + (33./8.)*dqS + (3./4.)*dqAdmu + (-(5./2.) - (5./8.)*dqS)*nu, 369./16. + (75./16.)*dqS + (3./16.)*dqAdmu + (-(29./4.) - (29./16.)*dqS)*nu]))
		self.chidch_p_k4SS = my_cpoly(chi_eff*np.array([(43./4. + (3./2.)*dqS)*dmu + (33./4. - (5./4.)*nu)*dqA, (21./8. + (3./8.)*dqS)*dmu + (75./8. - (29./8.)*nu)*dqA]))
		self.dch2_p_k4SS = my_cpoly(np.array([1./8. + (33./8.)*dqS + (3./4.)*dqAdmu + (-(7./2.) - (5./8.)*dqS)*nu, -(3./16.) + (75./16.)*dqS + (3./16.)*dqAdmu - (29./16.)*dqS*nu]))

	#function to compute Dy and De^2
	def Dy_De2_Dl_Ddl(self, y, e2, dchi, dchi2, sperp2):

		#compute different things that will be needed
		sqrt1me2 = (1-e2)**0.5
		one_m_sqrt = 1 - sqrt1me2
		sqrt_a = (1 - sqrt1me2)/sqrt1me2
		e2sqrt = e2/sqrt1me2
		log_fact = np.log((1 + sqrt1me2)/(8*y*sqrt1me2*(1-e2)))
		y2 = y*y
		y8 = y**8

		#initialize the PN derivatives
		Dy  = 0
		De2 = 0
		k   = 0
		
		#add the required terms
		if self.pn_max_order>=6:
			if self.pn_phase_order>=6:
				Dy  += self.p_a6NS(e2) + sqrt_a*self.sqrt_a6NS(e2) + log_fact*self.log_a6NS(e2)
				De2 += e2*(self.p_b6NS(e2) + sqrt1me2*self.sqrt_b6NS(e2) + log_fact*self.log_b6NS(e2)) + one_m_sqrt*self.one_m_sqrt_b6NS
				k   += self.p_k4NS(e2) + sqrt1me2*self.sqrt_k4NS(e2)
			if self.pn_spin_order>=6:
				Dy  += self.const_p_a6SOSS(e2) + dchi*self.dch_p_a6SOSS(e2) + sqrt_a*self.chi2_sqrt_a6SS(e2) + dchi*sqrt_a*self.chidch_sqrt_a6SS(e2) + dchi2*(self.dch2_p_a6SS(e2) + sqrt_a*self.dch2_sqrt_a6SS(e2))
				De2 += e2*(self.const_p_b6SOSS(e2) + dchi*self.dch_p_b6SOSS(e2) + sqrt1me2*self.chi2_sqrt_b6SS(e2) + dchi*sqrt1me2*self.chidch_sqrt_b6SS(e2) + dchi2*(self.dch2_p_b6SS(e2) + sqrt1me2*self.dch2_sqrt_b6SS(e2)))
				k   += self.chi2_p_k4SS(e2) + dchi*self.chidch_p_k4SS(e2) + dchi2*self.dch2_p_k4SS(e2)
			Dy  *= y
			De2 *= y
			k   *= y
		if self.pn_max_order>=5:
			if self.pn_phase_order>=5:
				Dy  += self.p_a5NS(e2)
				De2 += e2*self.p_b5NS(e2)
			if self.pn_spin_order>=5:
				Dy  += self.chi_p_a5SO(e2) + e2sqrt*self.chi_e2sqrt_a5SO(e2) + dchi*(self.dch_p_a5SO(e2) + e2sqrt*self.dch_e2sqrt_a5SO(e2))
				De2 += e2*(self.chi_p_b5SO(e2) + sqrt1me2*self.chi_sqrt_b5SO(e2) + dchi*(self.dch_p_b5SO(e2) + sqrt1me2*self.dch_sqrt_b5SO(e2)))
				k   += self.chi_p_k3SO(e2) + dchi*self.dch_p_k3SO(e2)
			Dy  *= y
			De2 *= y
			k   *= y
		if self.pn_max_order>=4:
			if self.pn_phase_order>=4:
				Dy  += self.p_a4NS(e2) + sqrt_a*self.sqrt_a4NS(e2)
				De2 += e2*(self.p_b4NS(e2) + sqrt1me2*self.sqrt_b4NS(e2))
				k   += self.p_k2NS(e2)
			if self.pn_spin_order>=4:
				Dy  += self.const_a4SS(e2) + sperp2*self.sperp2_a4SS(e2) + dchi*self.chidch_a4SS(e2) + dchi2*self.dch2_a4SS(e2)
				De2 += self.const_b4SS(e2) + sperp2*self.sperp2_b4SS(e2) + dchi*self.chidch_b4SS(e2) + dchi2*self.dch2_b4SS(e2)
				k   += self.const_k2SS     + sperp2*self.sperp2_k2SS     + dchi*self.chidch_k2SS     + dchi2*self.dch2_k2SS
			Dy  *= y
			De2 *= y
			k   *= y
		if self.pn_max_order>=3:
			if self.pn_phase_order>=3:
				Dy  += self.p_a3NS(e2)
				De2 += e2*self.p_b3NS(e2)
			if self.pn_spin_order>=3:
				Dy  += self.chi_p_a3SO(e2) + dchi*self.dch_p_a3SO(e2)
				De2 += e2*(self.chi_p_b3SO(e2) + dchi*self.dch_p_b3SO(e2))
				k   += self.chi_k1SO + dchi*self.dch_k1SO
			Dy  *= y
			De2 *= y
			k   *= y
		if self.pn_phase_order>=2:
			Dy  = y2*(Dy  + self.p_a2NS(e2))
			De2 = y2*(De2 + e2*self.p_b2NS(e2))
			k   = y2*(k   + self.k0NS)
		
		#always take into account the 0PN terms of Dy and De2
		Dy = y*y8*self.nu*(self.p_a0NS(e2) + Dy)
		De2 = -y8*self.nu*(e2*self.p_b0NS(e2) + De2)

		#compute D\lambda from Eq.(103)
		Dl = y2*y

		#compute D\delta\lambda from Eq.(104)
		Ddl = k*Dl/(1+k)

		return Dy, De2, Dl, Ddl

