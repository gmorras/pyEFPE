#my lightweight cython class to evaluate polynomials using Horners method
cdef class my_cpoly:
	
	#Cython memory view for better performance
	cdef double[:] coefficients
	cdef int N
	
	#initialize polynomial class
	def __cinit__(self, double[:] coeffs):
		#Copy coefficients into internal array, fliping them for Horner method
		self.coefficients = coeffs[::-1].copy()
		self.N = len(self.coefficients)
		
	def __call__(self, double x) -> double:
		#initialize the result
		cdef double result = self.coefficients[0]
		cdef int i
			
		# Use Horner's method to evaluate the polynomial
		for i in range(1, self.N):
			result = result*x + self.coefficients[i]
			
		return result
