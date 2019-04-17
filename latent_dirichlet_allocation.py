import sys
from sklearn.decomposition import LatentDirichletAllocation
import scipy
import pandas as pd

file_name = sys.argv[1]
mat = scipy.io.loadmat(file_name)
X=mat['X'].todense()

print(X.shape)

lda = LatentDirichletAllocation(n_components = 5, verbose = 100, max_iter = 100)
lda.fit(X) 
res = lda.transform(X)
print(res)

pd.DataFrame(res).to_csv("latent_dirichlet_allocation_res.csv", index=False)
