import numpy as np
import sys
from scipy import io
import scipy
import scipy.sparse
import pandas as pd
from preprocessing_tools import term_sentiment_matrix_to_context_matrix

def compute_loss(X, M, Z, S, W, Q, l_reg):
	ZSW_T = np.dot(np.dot(Z, S), W.T)
	WQ_T = np.dot(W, Q.T)
	
	return 1/2 * np.linalg.norm(X - ZSW_T) ** 2 + l_reg/2 * np.linalg.norm(M - WQ_T) ** 2

def wc_nmtf(X, M, g, m, l_reg = 1):
	epsilon = 1e-12

	n = X.shape[0]
	d = X.shape[1]

	Z = np.random.rand(n, g)
	S = np.random.rand(g, m)
	W = np.random.rand(d, m)
	Q = np.random.rand(d, m)
	
	# To sparse
	"""Z = scipy.sparse.csr_matrix(Z)
	S = scipy.sparse.csr_matrix(S)
	W = scipy.sparse.csr_matrix(W)
	Q = scipy.sparse.csr_matrix(Q)
	X = scipy.sparse.csr_matrix(X)
	M = scipy.sparse.csr_matrix(M)"""
	
	i = 0
	epoch = 500
	print_loss_frequency = epoch / 100
	stop_criterion = False
	while not stop_criterion:
		if i % print_loss_frequency == 0:
			loss = compute_loss(X, M, Z, S, W, Q, l_reg)
			
			print(i,"___",loss)
	
		# Compute Z
		denom = np.dot(np.dot(np.dot(Z, S), np.dot(W.T, W)), S.T)
		delta_Z = np.divide(np.dot(np.dot(X, W), S.T) + epsilon, denom + epsilon)
		Z = np.multiply(Z, delta_Z)
		
		# Compute W
		denom = np.dot(W, np.dot(np.dot(S.T, np.dot(Z.T, Z)), S) + l_reg * np.dot(Q.T, Q))
		delta_W = np.divide(np.dot(np.dot(X.T, Z), S) + l_reg * np.dot(M, Q) + epsilon, denom + epsilon)
		W = np.multiply(W, delta_W)

		# Compute S
		denom = np.dot(np.dot(np.dot(Z.T, Z), S), np.dot(W.T, W))
		delta_S = np.divide(np.dot(np.dot(Z.T, X), W) + epsilon, denom + epsilon)
		S = np.multiply(S, delta_S)
		
		# Compute Q
		denom = np.dot(Q, np.dot(W.T, W))
		delta_Q = np.divide(np.dot(M.T, W) + epsilon, denom + epsilon)
		Q = np.multiply(Q, delta_Q)
		
		i += 1
		stop_criterion = i > epoch
	
	# To dense
	"""Z = scipy.sparse.csr_matrix.todense(Z)
	S = scipy.sparse.csr_matrix.todense(S)
	W = scipy.sparse.csr_matrix.todense(W)
	Q = scipy.sparse.csr_matrix.todense(Q)"""		
	
	return {"Z": Z, "S": S, "W": W, "Q": Q}
	
if __name__ == '__main__':
	"""n = 50
	d = 100
	X = np.random.rand(n, d)
	M = np.random.rand(d, d)"""
	
	mat = io.loadmat(sys.argv[1])
	X = scipy.sparse.csr_matrix.todense(mat['X'])
	M = term_sentiment_matrix_to_context_matrix(sys.argv[2])

	g = int(sys.argv[3])
	m = int(sys.argv[4])
	l_reg = float(sys.argv[5])

	print(X.shape)

	res = wc_nmtf(X, M, g, m, l_reg = l_reg)
	
	Z = res["Z"]
	pd.DataFrame(Z).to_csv('wc-nmtf_Z.csv', index=False)
