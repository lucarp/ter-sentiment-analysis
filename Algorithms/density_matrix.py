import numpy as np
from math import log
import scipy
from scipy import io
import sys

csr_dot = scipy.sparse.csr_matrix.dot

def compute_gradient(density_matrix, projectors, term_frequencies):
	#gradient = np.zeros(density_matrix.shape)
	gradient = scipy.sparse.csr_matrix(density_matrix.shape)
	for i in range(len(projectors)):
		projector = projectors[i]
		tf = term_frequencies[i]
		#delta = tf / np.trace(np.dot(projector, density_matrix)) * projector
		delta = tf / csr_dot(projector, density_matrix).diagonal().sum() * projector
		gradient += delta
	return gradient

def compute_q(trace_gdg, t):
	return 1 + 2 * t + t ** 2 * trace_gdg

def compute_direction_bar(density_matrix, gradient):
	#return (np.dot(gradient, density_matrix) + np.dot(density_matrix, gradient)) / 2 - density_matrix
	return (csr_dot(gradient, density_matrix) + csr_dot(density_matrix, gradient)) / 2 - density_matrix

def compute_direction_tild(density_matrix, gdg, trace_gdg):
	return gdg / trace_gdg - density_matrix

def compute_direction(density_matrix, projectors, term_frequencies, t):
	gradient = compute_gradient(density_matrix, projectors, term_frequencies)
	#gdg = np.dot(np.dot(gradient, density_matrix), gradient)
	gdg = csr_dot(csr_dot(gradient, density_matrix), gradient)
	#trace_gdg = np.trace(gdg)
	trace_gdg = gdg.diagonal().sum()
	
	d_bar = compute_direction_bar(density_matrix, gradient)
	d_tild = compute_direction_tild(density_matrix, gdg, trace_gdg)
	
	q_t = compute_q(trace_gdg, t)
		
	return 2 / q_t * d_bar + t * trace_gdg / q_t * d_tild

def compute_objective(density_matrix, projectors):
	obj = 0
	for i in range(len(projectors)):
		projector = projectors[i]
		#tr = np.trace( np.dot(projector, density_matrix) )
		tr = csr_dot(projector, density_matrix).diagonal().sum()
		if tr > 0.0:
			obj += log( tr )
	return obj

def gqlm(projectors, term_frequencies, t = 0.5):
	epsilon = 1e-5
	num_epoch = 200
	
	num_words = projectors[0].shape[0]
	
	diagonal_entries = np.random.rand(num_words)
	diagonal_entries /= np.sum(diagonal_entries)
	#density_matrix = np.diag(diagonal_entries)
	#density_matrix = scipy.sparse.csr_matrix(density_matrix)
	density_matrix = scipy.sparse.diags(diagonal_entries)

	old_obj = compute_objective(density_matrix, projectors)
	print(old_obj)

	i = 0	
	stop_criterion = False	
	while not stop_criterion:
		delta = t * compute_direction(density_matrix, projectors, term_frequencies, t)
		density_matrix += delta
		
		obj = compute_objective(density_matrix, projectors)
		print(obj)
		stop_criterion = abs(obj - old_obj) <= epsilon
		old_obj = obj
		
		i +=1 
		
	return density_matrix

def document_term_matrix_to_density_matrices(file_name):
	document_term_matrix = io.loadmat(file_name)
	document_term_matrix = document_term_matrix['X']
	
	num_documents = document_term_matrix.shape[0]
	num_words = document_term_matrix.shape[1]
	
	density_matrices = []
	
	for doc_idx in range(num_documents):
		print("doc {} / {}".format(doc_idx, num_documents))
		cx = scipy.sparse.coo_matrix(document_term_matrix[doc_idx])
		tf = []
		projectors = []
		for i,j,v in zip(cx.row, cx.col, cx.data):
			#print(doc_idx,j,v)
			tf.append(v)
			#pr = np.zeros((num_words,num_words))
			#pr[j][j] = 1
			pr = scipy.sparse.csr_matrix(([1], ([j], [j])), shape=[num_words, num_words])
			projectors.append(pr)
			
		dm = gqlm(projectors, tf)
		scipy.io.savemat("dataset/mat_files/density_matrices/doc_"+str(doc_idx)+".mat", {'X' : dm})
		density_matrices.append(dm)
			
	return density_matrices

if __name__ == '__main__':
	"""projectors = []
	tf = []
	pr = np.zeros((100,100))

	for i in range(10):
		temp = np.copy(pr)
		temp[i][i] = 1
		projectors.append(temp)
		tf.append(i+1)
	
	dm = gqlm(projectors, tf)
	print(dm)"""
	
	document_term_matrix_to_density_matrices(sys.argv[1])
