setwd("/media/matthieu/Data/Matthieu/##Etude/#M1/S2/TER/ter-sentiment-analysis")

library(aricode)
library(R.matlab)
library(skmeans)
library("FactoMineR")
library(NMF)

normalize <- function(x) {x / sqrt(rowSums(x^2))}
normalizeByCol <- function(df) { t( normalize( t(df) ) )}

# ------- Dataset loading -------
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

X <- readMat("mat_files/output_not_original_most_1000.csv_tf-idf-l2.mat")

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



S <- read.csv("dataset/output_not_original_10_term_sentiment.csv")[,3:5]
S <- as.matrix(head(S, -1))
S <- normalize(S)
dim(S)
M <- S %*% t(S)
M <- normalize(M)
mat_df <- mat_df %*% M
mat_df <- normalize(mat_df)
dim(mat_df)
# ----------------------------------------

#svds(df)

# ------- Split, if wanted, per autor -------
# Get ID for each author
temp <- label$V1[1]
for(i in 2:length(label$V1)) { 
  temp <- label$V1[i-1]
  if(label$V1[i] < temp)
    print(i)
}
#1028
#2335
#3237

label <- matrix(label[1:1027,])
df <- df[1:1027,]
label <- matrix(label[1028:2334,])
df <- df[1028:2334,]
label <- matrix(label[2335:3236,])
df <- df[2335:3236,]
#label <- matrix(label[3237:length(label),])
#df <- df[3237:length(label),]
# ----------------------------------------


# ------- K-means -------
# - Elbow criteron
rng <- 2:10 #K from 2 to 20
tries <- 10 #Run the K Means algorithm x times
avg.totw.ss <- integer(length(rng)) #Set up an empty vector to hold all of points
avg.rsquared <- integer(length(rng)) #Set up an empty vector to hold all of points
for(v in rng){ # For each value of the range variable
  v.totw.ss <- integer(tries) #Set up an empty vector to hold the x tries
  v.rsquared <- integer(tries) #Set up an empty vector to hold the x tries
  print(sprintf("-- v = %i", v))
  for(i in 1:tries){
    print(sprintf("i = %i", i))
    k.temp <- kmeans(df, centers=v) #Run kmeans
    v.totw.ss[i] <- k.temp$tot.withinss #Store the total withinss
    v.rsquared[i] <- k.temp$betweenss / k.temp$tot.withinss
  }
  avg.totw.ss[v-1] <- mean(v.totw.ss) #Average the total withinss
  avg.rsquared[v-1] <- mean(v.rsquared) #Average the R-squared
}
plot(rng, avg.totw.ss, type="b", main="Total Within SS by Various K",
     ylab="Average Total Within Sum of Squares",
     xlab="Value of K")
plot(rng, avg.rsquared, type="b", main="R squared by Various K",
     ylab="Average R squared",
     xlab="Value of K")

# - K means clustering
res <- kmeans(df, centers = k)
length(res$cluster)
length(labelK)

# - Plot 
layout(matrix(1:2))
plot(labelK, xlab = "Documents", ylab = "Cluster")
plot(res$cluster, xlab = "Documents", ylab = "Cluster")

# - Compute NMI and ARI
# TODO - Do correclty the NMI and ARI (match cluster value with label value)
NMI(res$cluster, labelK)
ARI(res$cluster, labelK)
# ----------------------------------------


# ------- Spherical K-means -------
res2 <- skmeans(mat_df, k)
res2$cluster

layout(matrix(1:2))
plot(labelK, xlab = "Documents", ylab = "Cluster")
plot(res2$cluster, xlab = "Documents", ylab = "Cluster")

NMI(res2$cluster, labelK)
ARI(res2$cluster, labelK)
# ----------------------------------------

layout(matrix(1:3))
plot(labelK, xlab = "Documents", ylab = "Cluster", main = "Real clusters")
plot(res$cluster, xlab = "Documents", ylab = "Cluster", main = "K-Means clusters")
plot(res2$cluster, xlab = "Documents", ylab = "Cluster", main = "Spherical K-Means clusters")

# ------- PCA -------

layout(matrix(c(1,2), ncol=2))
resPCA <- PCA(mat_df, scale.unit = FALSE)
plot.PCA(resPCA, choix="ind", habillage = 9, label = "none")

# ----------------------------------------


# ------- NMF -------
res_nmf <- nmf(mat_df, 4000)

# ----------------------------------------


# ------- WC-NMTF -------
res_wc_nmtf <- read.csv("result_wc_nmtf/wc-nmtf_Z_l1_cos_xnorm.csv", header = TRUE)
res_wc_nmtf <- read.csv("result_wc_nmtf/wc-nmtf_Z_l1_cos.csv", header = TRUE)
res_wc_nmtf <- read.csv("result_wc_nmtf/wc-nmtf_Z_l1_cos_2.csv", header = TRUE)
res_wc_nmtf <- read.csv("result_wc_nmtf/wc-nmtf_Z_l1_cos_2_200.csv", header = TRUE)
res_wc_nmtf <- read.csv("result_wc_nmtf/wc-nmtf_Z_l1_cos_5_200.csv", header = TRUE)
res_wc_nmtf <- read.csv("result_wc_nmtf/wc-nmtf_Z_l1_p.csv", header = TRUE)
res_wc_nmtf <- read.csv("result_wc_nmtf/wc-nmtf_Z_l1_p_2.csv", header = TRUE)
res_wc_nmtf <- read.csv("result_wc_nmtf/wc-nmtf_Z_l1.csv", header = TRUE)
res_wc_nmtf <- read.csv("result_wc_nmtf/wc-nmtf_Z_l1_5_200.csv", header = TRUE)

# cos files
res_wc_nmtf <- read.csv("result_wc_nmtf/lambda/cos/wc-nmtf_Z_l1000.0.csv", header = TRUE)
res_wc_nmtf <- read.csv("result_wc_nmtf/lambda/cos/wc-nmtf_Z_l100.0.csv", header = TRUE)
res_wc_nmtf <- read.csv("result_wc_nmtf/lambda/cos/wc-nmtf_Z_l10.0.csv", header = TRUE)
res_wc_nmtf <- read.csv("result_wc_nmtf/lambda/cos/wc-nmtf_Z_l1.0.csv", header = TRUE)
res_wc_nmtf <- read.csv("result_wc_nmtf/lambda/cos/wc-nmtf_Z_l0.1.csv", header = TRUE)
res_wc_nmtf <- read.csv("result_wc_nmtf/lambda/cos/wc-nmtf_Z_l0.01.csv", header = TRUE)
res_wc_nmtf <- read.csv("result_wc_nmtf/lambda/cos/wc-nmtf_Z_l0.001.csv", header = TRUE)
res_wc_nmtf <- read.csv("result_wc_nmtf/lambda/cos/wc-nmtf_Z_l0.0001.csv", header = TRUE)
res_wc_nmtf <- read.csv("result_wc_nmtf/lambda/cos/wc-nmtf_Z_l1e-05.csv", header = TRUE)

# tra files

# print results
res_wc_nmtf <- t( normalize( t(res_wc_nmtf) ) )

#apply(res_wc_nmtf, MARGIN = 1, FUN=normalize)
#sqrt(sum(res_wc_nmtf[,1]^2))

label_res <- apply(res_wc_nmtf, MARGIN = 1, FUN=which.max)
#res_std <- apply(res_wc_nmtf, MARGIN = 1, FUN=sd)
#res_std

layout(matrix(1:2))
plot(labelK)
plot(label_res)

# compute nmi and ari for each file

lambdas <- c("10000.0", "1000.0", "100.0", "10.0", "1.0", "0.1", "0.01", "0.001", "0.0001", "1e-05", "0.0")
lambdas <- rev(lambdas)

cos_nmi <- c()
cos_ari <- c()
for(lambda in lambdas){
  #file <- paste("result_wc_nmtf/lambda/cos/wc-nmtf_Z_l", lambda, ".csv", sep = "")
  file <- paste("result_wc_nmtf/lambda/p_tra/wc-nmtf_Z_l", lambda, "_p_tra.csv", sep = "")
  res_wc_nmtf <- read.csv(file, header = TRUE)
  res_wc_nmtf <- t( normalize( t(res_wc_nmtf) ) )
  label_res <- apply(res_wc_nmtf, MARGIN = 1, FUN=which.max)
  t_nmi <- NMI(label_res, labelK)
  #t_nmi <- NMI:NMI(label_res, labelK)
  t_ari <- ARI(label_res, labelK)
  cos_nmi <- c(cos_nmi, t_nmi)
  cos_ari <- c(cos_ari, t_ari)
}

plot(cos_nmi, type = "b", ylab = "NMI", xlab = "Lambda", xaxt = "n")
axis(1, at=1:length(lambdas), labels=lambdas)
plot(cos_ari, type = "b", ylab = "ARI", xlab = "Lambda", xaxt = "n")
axis(1, at=1:length(lambdas), labels=lambdas)

# ----------------------------------------


# ------- LDA -------
res_lda <- read.csv("latent_dirichlet_allocation_res.csv", header = TRUE)
# print results
res_lda <- t( normalize( t(res_lda) ) )

label_res <- apply(res_lda, MARGIN = 1, FUN=which.max)

layout(matrix(1:2))
plot(labelK)
plot(label_res)
# ----------------------------------------

# ---------- Autoencoder results ----------
k0 <- read.csv("result_autoencoder/autoencoder_0_code.csv", row.names = 1)
k0_ind <- read.csv("result_autoencoder/autoencoder_0_ind.csv", row.names = 1)
k1 <- read.csv("result_autoencoder/autoencoder_1_code.csv", row.names = 1)
k1_ind <- read.csv("result_autoencoder/autoencoder_1_ind.csv", row.names = 1)
k2 <- read.csv("result_autoencoder/autoencoder_2_code.csv", row.names = 1)
k2_ind <- read.csv("result_autoencoder/autoencoder_2_ind.csv", row.names = 1)
k3 <- read.csv("result_autoencoder/autoencoder_3_code.csv", row.names = 1)
k3_ind <- read.csv("result_autoencoder/autoencoder_3_ind.csv", row.names = 1)
k4 <- read.csv("result_autoencoder/autoencoder_4_code.csv", row.names = 1)
k4_ind <- read.csv("result_autoencoder/autoencoder_4_ind.csv", row.names = 1)

res_ae = rbind(k0, k1, k2, k3, k4)
k_ind = rbind(k0_ind, k1_ind, k2_ind, k3_ind, k4_ind)
res_ae = res_ae[order(k_ind),]
label_res_ae <- apply(res_ae, MARGIN = 1, FUN=which.max)

layout(matrix(1:2))
plot(labelK)
plot(label_res_ae)

# ----------------------------------------

# ----------------------------------------

#library(cluster)
#library(factoextra)
#fviz_cluster(res, data = df)