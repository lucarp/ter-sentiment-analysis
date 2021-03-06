#setwd("/media/matthieu/Data/Matthieu/##Etude/#M1/S2/TER/ter-sentiment-analysis")
setwd(dirname(rstudioapi::getSourceEditorContext()$path))

library(aricode)
library(R.matlab)
library(skmeans)
library("FactoMineR")
library(NMF)
library(caret)

normalize <- function(x) {x / sqrt(rowSums(x^2))}
normalizeByCol <- function(df) { t( normalize( t(df) ) )}
sent_process <- function(x){ x[1] - x[2] + 1e-12 }
sent_process2 <- function(x){ if(x[1] > x[2]) {
                                1}
                              else{ if(x[1] < x[2]){
                                -1
                              }else {0}
                              }  
}

mat_path <- "../dataset/mat_files/"

# -------------- Dataset loading --------------
mat_name <- paste(mat_path, "output_not_original_30.csv_tf-idf-l2.mat", sep = "")

X <- readMat(mat_name)

#X <- readMat("mat_files/output_30.csv_tf-idf-l2.mat")
#X <- readMat("mat_files/output_not_original_30.csv_bow.mat")
#X <- readMat("mat_files/output_not_original_30.csv_tf-idf.mat")
#X <- readMat("../mat_files/output_not_original_30.csv_tf-idf-l2.mat")
#X <- readMat("mat_files/output_not_original_50.csv_tf-idf-l2.mat")
#X <- readMat("mat_files/output_not_original_10.csv_tf-idf-l2.mat")
#X <- readMat("mat_files/output_not_original_5.csv_tf-idf-l2.mat")
#X <- readMat("mat_files/output_not_original_no_clean.csv_tf-idf-l2.mat")
#X <- readMat("mat_files/output_not_original_100.csv_tf-idf-l2.mat")

#X <- readMat("../density_matrices.mat")

#X <- readMat("mat_files/output_not_original_most_1000.csv_tf-idf-l2.mat")

df <- X$X

#df <- read.csv("Word_Doc2Vec/doc2vec_matrix.csv", header = FALSE)

#df <- read.csv("dataset/output_not_original_10.csv_doc2Vec.csv", header = TRUE, row.names = 1)
#df <- read.csv("dataset/output_not_original_10.csv_doc2Vec_e300.csv", header = TRUE, row.names = 1)
#df <- read.csv("dataset/output_not_original_10.csv_doc2Vec_w10.csv", header = TRUE, row.names = 1)
#df <- read.csv("dataset/output_not_original_10.csv_doc2Vec_w15.csv", header = TRUE, row.names = 1)
#df <- read.csv("dataset/output_not_original_10.csv_doc2Vec_w15_2.csv", header = TRUE, row.names = 1)
#df <- read.csv("dataset/output_not_original_10.csv_doc2Vec_w15_v100.csv", header = TRUE, row.names = 1)
#df <- read.csv("dataset/output_not_original_10.csv_doc2Vec_w15_e50.csv", header = TRUE, row.names = 1)
#df <- read.csv("dataset/output_not_original_10.csv_doc2Vec_w30.csv", header = TRUE, row.names = 1)
#df <- read.csv("dataset/output_not_original_10.csv_doc2Vec_w30_v5.csv", header = TRUE, row.names = 1)
#df <- read.csv("dataset/doc2vec/output_not_original_10.csv_doc2Vec_w50.csv", header = TRUE, row.names = 1)
#df <- read.csv("dataset/output_not_original_10.csv_doc2Vec_w2.csv", header = TRUE, row.names = 1)

dim(df)
mat_df <- as.matrix(df)
mat_df <- normalize(mat_df)
dim(mat_df)
label_name <- paste(mat_path, "dataset_LABEL.csv", sep = "")
label <- read.csv(label_name, header = FALSE)
#label

k <- 5
labelK <- apply(label, MARGIN = 1, FUN=function(x) max(1, ceiling(x*k))) # true label (1 to k)


# ---
S <- read.csv("dataset/output_not_original_10_term_sentiment.csv")[,3:5]
S <- as.matrix(head(S, -1))
S <- normalize(S)
dim(S)

M <- apply(S, MARGIN = 1, FUN = sent_process2)
M <- diag(M)

M <- S %*% t(S)
M <- normalize(M)

mat_df <- mat_df %*% M
mat_df <- normalize(mat_df)
dim(mat_df)
# ----------------------------------------

#svds(df)

# -------------- Split, if wanted, per autor --------------
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
mat_df <- mat_df[1:1027,]
label <- matrix(label[1028:2334,])
mat_df <- mat_df[1028:2334,]
label <- matrix(label[2335:3236,])
mat_df <- mat_df[2335:3236,]
label <- matrix(label[3237:dim(label)[1],])
mat_df <- mat_df[3237:dim(mat_df)[1],]
# ----------------------------------------

labelK <- apply(label, MARGIN = 1, FUN=function(x) max(1, ceiling(x*k))) # true label (1 to k)


# -------------- K-means --------------
# - Elbow criteron
rng <- 2:10 #K from 2 to 10
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
res <- kmeans(mat_df, centers = k)
length(res$cluster)
length(labelK)

# - Plot 
layout(matrix(1:2))
plot(labelK, xlab = "Documents", ylab = "Cluster")
plot(res$cluster, xlab = "Documents", ylab = "Cluster")

# - Compute NMI and ARI
NMI(res$cluster, labelK)
ARI(res$cluster, labelK)
# ----------------------------------------


# -------------- Spherical K-means --------------
res2 <- skmeans(mat_df, k)
res2$cluster

layout(matrix(1:2))
plot(labelK, xlab = "Documents", ylab = "Cluster")
plot(res2$cluster, xlab = "Documents", ylab = "Cluster")

aricode::NMI(res2$cluster, labelK)
ARI(res2$cluster, labelK)
# ----------------------------------------

svg(filename="kmeans_skmeans.svg")
layout(matrix(1:3))
plot(labelK, xlab = "Documents ID", ylab = "Rating", main = "Real clusters")
plot(res$cluster, xlab = "Documents ID", ylab = "Cluster", main = "K-Means clusters")
plot(res2$cluster, xlab = "Documents ID", ylab = "Cluster", main = "Spherical K-Means clusters")
dev.off()

# -------------- PCA --------------

layout(matrix(c(1,2), ncol=2))
resPCA <- PCA(mat_df, scale.unit = FALSE)
plot.PCA(resPCA, choix="ind", habillage = 9, label = "none")

# ----------------------------------------


# -------------- NMF --------------
res_nmf <- nmf(mat_df, 5)

# ----------------------------------------


# -------------- WC-NMTF --------------
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
res_wc_nmtf <- read.csv("result_wc_nmtf/lambda/cos/wc-nmtf_Z_l0.0.csv", header = TRUE)

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

mode <- 0

#lambdas <- c("10000.0", "1000.0", "100.0", "10.0", "1.0", "0.1", "0.01", "0.001", "0.0001", "1e-05", "0.0")
#lambdas_name <- c("1e04", "1000", "100", "10", "1", "0.1", "0.01", "0.001", "1e-04", "1e-05", "0")
lambdas <- c("10000.0", "1000.0", "100.0", "10.0", "1.0", "0.1", "0.01", "0.001", "0.0001", "1e-05")
lambdas_name <- c("1e04", "1000", "100", "10", "1", "0.1", "0.01", "0.001", "1e-04", "1e-05")
lambdas <- rev(lambdas)
lambdas_name <- rev(lambdas_name)

cos_nmi <- c()
cos_ari <- c()
for(lambda in lambdas){
  #if(mode == 0){
    #file <- paste("result_wc_nmtf/lambda/cos/wc-nmtf_Z_l", lambda, ".csv", sep = "")
  #} else {
    #file <- paste("result_wc_nmtf/lambda/p_tra/wc-nmtf_Z_l", lambda, "_p_tra.csv", sep = "") 
  #}
  
  file <- paste("result_temp/cos2/wc-nmtf_Z_l", lambda, ".csv", sep = "")
  
  res_wc_nmtf <- read.csv(file, header = TRUE)
  res_wc_nmtf <- t( normalize( t(res_wc_nmtf) ) )
  label_res <- apply(res_wc_nmtf, MARGIN = 1, FUN=which.max)
  t_nmi <- NMI(label_res, labelK)
  #t_nmi <- NMI:NMI(label_res, labelK)
  t_ari <- ARI(label_res, labelK)
  cos_nmi <- c(cos_nmi, t_nmi)
  cos_ari <- c(cos_ari, t_ari)
}

min <- min(min(cos_nmi), min(cos_ari))
max <- max(max(cos_nmi), max(cos_ari))

if(mode == 0){
  svg(filename="nmi_ari_wcnmtf_cos.svg") 
} else {
  svg(filename="nmi_ari_wcnmtf_tra.svg") 
}
plot(cos_nmi, type = "b", ylab = "", xlab = "Lambda", xaxt = "n", ylim = c(min , max))
lines(cos_ari, type= "b", pch = 2, lty = 4)
legend("topright",legend=c("NMI", "ARI"), lty = c(1,4), pch = 1:2)
axis(1, at=1:length(lambdas_name), labels=lambdas_name)
dev.off()

plot(cos_ari, type = "b", ylab = "ARI", xlab = "Lambda", xaxt = "n")
axis(1, at=1:length(lambdas), labels=lambdas)

# ----------------------------------------


# -------------- LDA --------------
res_lda <- read.csv("latent_dirichlet_allocation_res.csv", header = TRUE)
res_lda <- read.csv("latent_dirichlet_allocation_res_1000.csv", header = TRUE)
# print results
res_lda <- t( normalize( t(res_lda) ) )

label_res <- apply(res_lda, MARGIN = 1, FUN=which.max)

layout(matrix(1:2))
plot(labelK)
plot(label_res)
# ----------------------------------------


# -------------- Hierarchical Clustering --------------
dc <- dist(mat_df, method ="euclidean", diag=FALSE, upper=FALSE)
res_hclust <- hclust(dc, method = "ward.D2")

class=cutree(res_hclust,k)
label_res <- class

layout(matrix(1:2))
plot(labelK)
plot(label_res)
aricode::NMI(label_res, labelK)
ARI(label_res, labelK)
# ----------------------------------------


# ----------------- Autoencoder results -----------------
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


# ----------------- Correspondance analysis -----------------

dim(mat_df[1:100,])

res_CA <- CA(mat_df[1:100,])

# ----------------------------------------


# ----------------------------------------
df <- read.csv("dataset/output_not_original_10_doc_sentiment.csv", row.names = 1)
mat_df <- as.matrix(df)
mat_df <- normalize(mat_df)
mat_df <- normalizeByCol(mat_df)

temp <- apply(mat_df, MARGIN = 1, FUN = sent_process)
temp <- (temp-mean(temp)) / sd(temp)
temp <- (temp - min(temp)) / (max(temp) - min(temp))

temp <- as.matrix(temp)
k <- 10
label_res <- apply(temp, MARGIN = 1, FUN=function(x) max(1, ceiling(x*k))) # true label (1 to k)

layout(matrix(1:2))
plot(labelK)
plot(label_res)

NMI(label_res, labelK)
ARI(label_res, labelK)
# ----------------------------------------

#library(cluster)
#library(factoextra)
#fviz_cluster(res, data = df)


plot(labelK, xlab = "Document ID", ylab = "Rating", width = 1000, height = 500)
abline(v = 1027)
abline(v = 2334)
abline(v = 3236)
text(1027/2, 5, "Author 1")
text((2334-1027)/2 + 1027, 5, "Author 2")
text((3236-2334)/2 + 2334, 5, "Author 3")
text((5006-3236)/2 + 3236, 5, "Author 4")



# ----------------------------------------
a_nmi <- 0
a_ari <- 0
conf_mat <- 0
f1 <- 0
for(i in 0:4){
  print("-------")
  print(i)
  print("-------")
  #file <- paste("result_linearDA/lda_res_", i, ".csv", sep = "")
  #res <- read.csv(file, row.names = 1)
  file <- paste("result_MLR/y_pred_fold_", i, ".csv", sep = "")
  res <- read.csv(file, header = FALSE)
  res <- unlist(res, use.names = FALSE)
  
  #file <- paste("result_linearDA/lda_res_", i, "_ind.csv", sep = "")
  file <- paste("result_MLR/y_pred_fold_", i, "_ind.csv", sep = "")
  ind <- read.csv(file, row.names = 1)  
  ind <- ind + 1
  ind <- unlist(ind, use.names = FALSE)
  
  lab <- labelK[ind]
  
  nmi <- NMI(res, lab)
  ari <- ARI(res, lab)
  print(nmi)
  print(ari)
  a_nmi <- a_nmi + nmi
  a_ari <- a_ari + ari
  
  conf <- confusionMatrix(factor(res), factor(lab))
  conf_mat <- conf_mat + conf$table
  f1 <- f1 + conf$byClass[,7]
}
print("-------")
print("mean")
print("-------")
print(a_nmi/5)
print(a_ari/5)

