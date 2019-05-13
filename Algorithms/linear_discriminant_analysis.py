import sys
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import scipy
import pandas as pd
from sklearn.utils import shuffle

# ----- Load Dataset ------
file_name = sys.argv[1]
mat = scipy.io.loadmat(file_name)
train_X = mat['X'].todense()

file_name = sys.argv[2]
train_y = pd.read_csv(file_name, header = None)
train_y = np.ravel(train_y)

ind = np.arange(train_X.shape[0])

train_X, train_y, ind = shuffle(train_X, train_y, ind, random_state=0)
# ------------------------

len_dataset = train_X.shape[0]
test_ratio = 0.20
test_size = int(len_dataset*test_ratio)

# ------ k-fold ------
k = int(sys.argv[3])
maxK = 1/test_ratio
if(k > maxK-1):
	k = maxK-1
# ------------------------

# ------ Data Splitting ------
if test_size != 0:
	b_split = int(len_dataset * test_ratio * k)
	e_split = int((b_split + len_dataset * test_ratio) % len_dataset)
	test_X, test_y = train_X[b_split:e_split], train_y[b_split:e_split]
	ind = ind[b_split:e_split]
	train_X = np.concatenate([train_X[0:b_split], train_X[e_split:]])
	train_y = np.concatenate([train_y[0:b_split], train_y[e_split:]])
# ------------------------

"""print(ind)
pd.DataFrame(ind).to_csv("lda_res_"+str(k)+"_ind.csv")
input()"""

# ------ Train ------
print("run LDA...")
clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
clf.fit(train_X, train_y)
print("done")
# ------------------------

# ------ Test ------
res = clf.predict(test_X)
print(res)
# ------------------------
