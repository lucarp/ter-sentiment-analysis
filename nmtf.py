import numpy as np
import sys
from scipy import io
import scipy
from preprocessing_tools import term_sentiment_matrix_to_context_matrix

def compute_loss(X, M, Z, S, W, Q, l_reg):
	ZSW_T = np.dot(np.dot(Z, S), np.transpose(W))
	WQ_T = np.dot(W, np.transpose(Q))
	
	return 1/2 * np.linalg.norm(X - ZSW_T) ** 2 + l_reg/2 * np.linalg.norm(M - WQ_T) ** 2

def wc_nmtf(X, M, g = 5, m = 5, l_reg = 1):
	n = X.shape[0]
	d = X.shape[1]
	
	Z = np.random.rand(n, g)
	S = np.random.rand(g, m)
	W = np.random.rand(d, m)
	Q = np.random.rand(d, m)
	
	X_T = np.transpose(X)
	M_T = np.transpose(M)
	
	epsilon = 1e-12
	i = 0
	epoch = 1000
	stop_criterion = False
	while not stop_criterion:
		loss = compute_loss(X, M, Z, S, W, Q, l_reg)
		print(loss)
	
		# Compute Z
		S_T = np.transpose(S)
		W_T = np.transpose(W)
		denom = np.dot(np.dot(np.dot(np.dot(Z, S), W_T), W), S_T)
		denom[denom == 0] = epsilon
		delta_Z = np.divide(np.dot(np.dot(X, W), S_T), denom)
		Z = np.multiply(Z, delta_Z)
		
		# Compute W
		Z_T = np.transpose(Z)
		Q_T = np.transpose(Q)
		denom = np.dot(W, np.dot(np.dot(np.dot(S_T, Z_T), Z), S) + l_reg * np.dot(Q_T, Q))
		denom[denom == 0] = epsilon		
		delta_W = np.divide(np.dot(np.dot(X_T, Z), S) + l_reg * np.dot(M, Q), denom)
		W = np.multiply(W, delta_W)

		# Compute S
		denom = np.dot(np.dot(np.dot(np.dot(Z_T, Z), S), W_T), W)
		denom[denom == 0] = epsilon		
		delta_S = np.divide(np.dot(np.dot(Z_T, X), W), denom)
		S = np.multiply(S, delta_S)
		
		# Compute Q
		denom = np.dot(np.dot(Q, W_T), W)
		denom[denom == 0] = epsilon		
		delta_Q = np.divide(np.dot(M_T, W), denom)
		Q = np.multiply(Q, delta_Q)
		
		i += 1
		stop_criterion = i > epoch
	
	return {"Z": Z, "S": S, "W": W, "Q": Q}
	
if __name__ == '__main__':
	"""n = 10
	d = 20
	X = np.random.rand(n, d)
	M = np.random.rand(d, d)"""
	
	mat = io.loadmat(sys.argv[1])
	X = scipy.sparse.csr_matrix.todense(mat['X'])
	M = term_sentiment_matrix_to_context_matrix(sys.argv[2])
	
	wc_nmtf(X, M)
