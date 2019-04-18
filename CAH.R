setwd("/media/matthieu/Data/Matthieu/##Etude/#M1/S2/TER/ter-sentiment-analysis")

library(aricode)
library(R.matlab)
library(skmeans)
library("FactoMineR")
library(NMF)

normalize <- function(x) {x / sqrt(rowSums(x^2))}
normalizeByCol <- function(df) { t( normalize( t(df) ) )}
sent_process <- function(x){ (x[1] - x[2]) * 10 + x[3] }

# -------------- Dataset loading --------------
#X <- readMat("mat_files/output_30.csv_tf-idf-l2.mat")
#X <- readMat("mat_files/output_not_original_30.csv_bow.mat")
#X <- readMat("mat_files/output_not_original_30.csv_tf-idf.mat")
#X <- readMat("mat_files/output_not_original_30.csv_tf-idf-l2.mat")
#X <- readMat("mat_files/output_not_original_50.csv_tf-idf-l2.mat")
X <- readMat("mat_files/output_not_original_10.csv_tf-idf-l2.mat")
#X <- readMat("mat_files/output_not_original_5.csv_tf-idf-l2.mat")
#X <- readMat("mat_files/output_not_original_no_clean.csv_tf-idf-l2.mat")
#X <- readMat("mat_files/output_not_original_100.csv_tf-idf-l2.mat")

#X <- readMat("../density_matrices.mat")

#X <- readMat("mat_files/output_not_original_most_1000.csv_tf-idf-l2.mat")

#X <- read.csv("doc2vec_matrix.csv", header = FALSE)

df <- X$X
dim(df)
mat_df <- as.matrix(df)
mat_df <- normalize(mat_df)
dim(mat_df)
label <- read.csv("mat_files/dataset_LABEL.csv", header = FALSE)
#label

k <- 5
labelK <- apply(label, MARGIN = 1, FUN=function(x) max(1, ceiling(x*k))) # true label (1 to k)
# ----------------------------------------

# -------------- Hierarchical Clustering --------------
dc <- dist(mat_df, method ="euclidean")
res_hclust <- hclust(dc)

svg(filename="cah_tree.svg")
plot(hc,hang=-1)
dev.off()

class=cutree(hc,k)
label_res <- class

svg(filename="cah_label.svg")
layout(matrix(1:2))
plot(labelK)
plot(label_res)
dev.off()
# ----------------------------------------