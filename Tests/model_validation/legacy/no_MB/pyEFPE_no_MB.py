import pyEFPE

#make a wrapper of pyEFPE that turns off amplitude interpolation 
def pyEFPE_no_MB(params):

	#make a copy of params
	p = params.copy()

	#turn off amplitude interpolation
	p['Interpolate_Amplitudes'] = False

	#now initialize pyEFPE with these parameters and return it
	return pyEFPE.pyEFPE(p)
