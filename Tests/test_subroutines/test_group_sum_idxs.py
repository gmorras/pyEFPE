import numpy as np
import cython
import time

############################################################################################

#function to sum over repeated values of idxs and put it in an array
def group_sum_idxs(arr, idxs, len_final=None):

	'''
	Function to sum values of arr wich have the same value of idxs and put it in array of len_f
	
	Parameters
	----------
	arr: array_like, shape (N,...)
		array to be summed
	idxs: 1-D array_like, len: N
		Indexes of final array each element of arr is going to be summed to
	len_final: int
		Length of final array to put the result into - Default: None
	'''
	
	#find the length needed for the result array
	if len_final is None: len_final = np.amax(idxs)
	else: len_final = int(len_final)
	
	#construct the result array
	result = np.zeros((len_final, *arr.shape[1:]), dtype=arr.dtype)
	
	#if not sorted, sort the input indices
	if np.any(idxs[1:]<idxs[:-1]):
		idxs_argsort = np.argsort(idxs)
		idxs = idxs[idxs_argsort]
		arr = arr[idxs_argsort]
	
	#compute sorted unique values of sorted input idxs
	u_idxs, u_idxs_inv, u_idxs_counts = np.unique(idxs, return_inverse=True, return_counts=True)
	
	#compute the final index of 'idxs' array corresponding to u_idxs value
	idxs_end = np.cumsum(u_idxs_counts)
	
	#initialize indexes in arr to be summed
	idxs_sum = idxs_end - u_idxs_counts
	
	#loop over repetitions
	for count in range(np.amax(u_idxs_counts)):
		
		#update the result
		result[u_idxs] += arr[idxs_sum]
		
		#go to next arr index to be summed
		idxs_sum += 1
		
		#see for which indices we have to keep summing
		keep_sum = (idxs_sum<idxs_end)
		
		#keep only the indexes that still have to be summed over
		if not np.all(keep_sum):
			idxs_sum = idxs_sum[keep_sum]
			idxs_end = idxs_end[keep_sum]
			u_idxs = u_idxs[keep_sum]
	
	#return the resulting array
	return result
	
#simple function to sum over repeated values of idxs and put it in an array
def simple_sum_repeated_idxs(arr, idxs, len_final=None):

	'''
	Function to sum values of arr wich have the same value of idxs and put it in array of len_f
	
	Parameters
	----------
	arr: array_like, shape (N,...)
		array to be summed
	idxs: 1-D array_like, len: N
		Indexes of final array each element of arr is going to be summed to
	len_final: int
		Length of final array to put the result into - Default: None
	'''
	
	#find the length needed for the result array
	if len_final is None: len_final = np.amax(idxs)
	else: len_final = int(len_final)
	
	#construct the result array
	result = np.zeros((len_final, *arr.shape[1:]), dtype=arr.dtype)

	#loop over indices
	for i_arr, i_res in enumerate(idxs):
		result[i_res] += arr[i_arr]
	
	return result

#function to sum over repeated values of idxs and put it in an array
def sum_repeated_idxs(arr, idxs, len_final=None):

	'''
	Function to sum values of arr wich have the same value of idxs and put it in array of len_f
	
	Parameters
	----------
	arr: array_like, shape (N,...)
		array to be summed
	idxs: 1-D array_like, len: N
		Indexes of final array each element of arr is going to be summed to
	len_final: int
		Length of final array to put the result into - Default: None
	'''
	
	#find the length needed for the result array
	if len_final is None: len_final = np.amax(idxs)
	else: len_final = int(len_final)
	
	#construct the result array
	result = np.zeros((len_final, *arr.shape[1:]), dtype=arr.dtype)
	
	#add things in place
	np.add.at(result, idxs, arr)
	
	#return the result which has been modified by in-place operation
	return result


#function to compute runtime
def compute_runtime(func, args, Ntries):
	
	#start a timer
	start = time.time()
	
	#run the function Ntries times
	for _ in range(Ntries):
		res = func(*args)
	
	#return the time taken
	total_time = time.time() - start
	return res, total_time/Ntries

############################################################################################

#Input
arr_len = 200000
arr_width = 5
len_final = 10000
Ntries = 10

#crete random array
arr_re = np.random.uniform(low=0, high=1, size=(arr_len, arr_width))
arr_im = np.random.uniform(low=0, high=1, size=(arr_len, arr_width))
arr = arr_re + 1j*arr_im

#create random indexes
idxs = np.random.uniform(low=0, high=len_final, size=(arr_len,)).astype(int)

#time and compute result with both methods
res_group, time_group = compute_runtime(group_sum_idxs, (arr, idxs, len_final), Ntries)
res_ez, time_ez = compute_runtime(simple_sum_repeated_idxs, (arr, idxs, len_final), Ntries)
res_np, time_np = compute_runtime(sum_repeated_idxs, (arr, idxs, len_final), Ntries)

print('Runtime with pyEFPE implementation: %.3g s'%(time_group))
print('Runtime with simple implementation: %.3g s'%(time_ez))
print('Runtime with np.add implementation: %.3g s'%(time_np))
print('Difference between both methods: %.3g -> Speedup: %.3g'%(np.linalg.norm((res_group-res_ez)/res_ez)/np.sqrt(np.prod(res_ez.shape)), time_ez/time_group))
print('Difference between both methods: %.3g -> Speedup: %.3g'%(np.linalg.norm((res_np-res_ez)/res_ez)/np.sqrt(np.prod(res_ez.shape)), time_ez/time_np))

#plot distribution of random indexes
from matplotlib import pyplot as plt
plt.rcParams.update({'font.size': 24})
plt.rcParams.update({'lines.linewidth': 2})

plt.figure(figsize=(12,8))
u_idxs, counts = np.unique(idxs, return_counts=True)
plt.hist(counts, bins=max(counts)-min(counts))
plt.xlabel('Index multiplicity')
plt.xlim(min(counts), max(counts))
plt.show()

plt.show()

