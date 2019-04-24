import numpy as np
import sys
from scipy import io
import scipy
import scipy.sparse
import pandas as pd
#from preprocessing_tools import term_sentiment_matrix_to_context_matrix, sppmi_context_matrix
from sklearn.preprocessing import normalize

csr_dot = scipy.sparse.csr_matrix.dot
sparse_linalg_norm = scipy.sparse.linalg.norm

def compute_loss(X, M, Z, S, W, Q, l_reg):
	ZSW_T = csr_dot(csr_dot(Z, S), W.T)
	WQ_T = csr_dot(W, Q.T)
	
	return 1/2 * sparse_linalg_norm(X - ZSW_T) ** 2 + l_reg/2 * sparse_linalg_norm(M - WQ_T) ** 2

def wc_nmtf(X, M, g, m, l_reg = 1):
	epsilon = 1e-12

	n = X.shape[0]
	d = X.shape[1]

	Z = np.random.rand(n, g)
	S = np.random.rand(g, m)
	W = np.random.rand(d, m)
	Q = np.random.rand(d, m)
	
	# To sparse
	Z = scipy.sparse.csr_matrix(Z)
	S = scipy.sparse.csr_matrix(S)
	W = scipy.sparse.csr_matrix(W)
	Q = scipy.sparse.csr_matrix(Q)
	"""X = scipy.sparse.csr_matrix(X)
	M = scipy.sparse.csr_matrix(M)"""
	
	i = 0
	epoch = 300
	print_loss_frequency = epoch / 100
	stop_criterion = False
	while not stop_criterion:
		if i % print_loss_frequency == 0:
			loss = compute_loss(X, M, Z, S, W, Q, l_reg)
			
			print(i,"___",loss)
	
		# Compute Z
		denom = csr_dot(csr_dot(csr_dot(Z, S), csr_dot(W.T, W)), S.T)
		if(denom.sum() == 0):
			denom = epsilon
		delta_Z = (csr_dot(csr_dot(X, W), S.T)) / (denom)		
		Z = scipy.sparse.csr_matrix.multiply(Z, delta_Z)
		
		# Compute W
		denom = csr_dot(W, csr_dot(csr_dot(S.T, csr_dot(Z.T, Z)), S) + l_reg * csr_dot(Q.T, Q))
		if(denom.sum() == 0):
			denom = epsilon		
		delta_W = (csr_dot(csr_dot(X.T, Z), S) + l_reg * csr_dot(M, Q)) / (denom)
		W = scipy.sparse.csr_matrix.multiply(W, delta_W)

		# Compute S
		denom = csr_dot(csr_dot(csr_dot(Z.T, Z), S), csr_dot(W.T, W))
		if(denom.sum() == 0):
			denom = epsilon		
		delta_S = (csr_dot(csr_dot(Z.T, X), W)) / (denom)
		S = scipy.sparse.csr_matrix.multiply(S, delta_S)
		
		# Compute Q
		denom = csr_dot(Q, csr_dot(W.T, W))
		if(denom.sum() == 0):
			denom = epsilon		
		delta_Q = (csr_dot(M.T, W)) / (denom)
		Q = scipy.sparse.csr_matrix.multiply(Q, delta_Q)
		
		i += 1
		stop_criterion = i > epoch
	
	loss = compute_loss(X, M, Z, S, W, Q, l_reg)	
	
	return {"Z": Z, "S": S, "W": W, "Q": Q, "loss": loss}
	
if __name__ == '__main__':
	"""n = 50
	d = 100
	X = np.random.rand(n, d)
	X = scipy.sparse.csr_matrix(X)
	M = np.random.rand(d, d)
	M = scipy.sparse.csr_matrix(M)"""
	
	print("Usage: {} X_mat_file M_mat_file g m lambda iter_lamba_x10".format(sys.argv[0]))
	
	X = io.loadmat(sys.argv[1])
	
	#X = scipy.sparse.csr_matrix.todense(X['X'])
	#X = normalize(X)
	#M = term_sentiment_matrix_to_context_matrix(sys.argv[2], preprocess = True)
	#M = term_sentiment_matrix_to_context_matrix(sys.argv[2])
	#M = term_sentiment_matrix_to_context_matrix(sys.argv[2], method='cos')
	#M = pd.read_csv(sys.argv[2], index_col = 0)

	#M = sppmi_context_matrix(M, N = 1)
	
	M = io.loadmat(sys.argv[2])
	
	#M = scipy.sparse.csr_matrix.todense(M['X'])
	#M = normalize(X)	
	
	#M = scipy.sparse.csr_matrix((X.shape [1], X.shape[1]))

	g = int(sys.argv[3])
	m = int(sys.argv[4])
	l_reg = float(sys.argv[5])

	print(X.shape)

	num_iter = 10
	for _ in range(int(sys.argv[6])):
		best_loss = -1
		for i in range(num_iter):
			print("iter",i)
			res = wc_nmtf(X, M, g, m, l_reg = l_reg)
			if best_loss == -1 or best_loss > res["loss"]:
				best_loss = res["loss"]
				bestZ = res["Z"]
		pd.DataFrame(bestZ).to_csv("wc-nmtf_Z_l"+str(l_reg)+".csv", index=False)
		my_file = open("wc-nmtf_Z_loss.csv", "a")
		my_file.write(str(best_loss)+"\n")
		my_file.close()
		l_reg *= 10
