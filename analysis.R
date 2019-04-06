setwd("/media/matthieu/Data/Matthieu/##Etude/#M1/S2/TER/ter-sentiment-analysis")

library(aricode)
library(R.matlab)
library(skmeans)
library("FactoMineR")
library(NMF)

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
df <- X$X
#df
dim(df)
mat_df <- as.matrix(df)
dim(mat_df)
label <- read.csv("mat_files/dataset_LABEL.csv", header = FALSE)
#label
# ----------------------------------------

svds(df)


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


k <- 5
labelK <- apply(label, MARGIN = 1, FUN=function(x) max(1, ceiling(x*k))) # true label (1 to k)


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
plot(labelK)
plot(res$cluster)

# - Compute NMI and ARI
# TODO - Do correclty the NMI and ARI (match cluster value with label value)
NMI(res$cluster, labelK)
ARI(res$cluster, labelK)
# ----------------------------------------


# ------- Spherical K-means -------
res2 <- skmeans(df, k)
res2$cluster

layout(matrix(1:2))
plot(labelK)
plot(res2$cluster)

NMI(res2$cluster, labelK)
ARI(res2$cluster, labelK)
# ----------------------------------------


# ------- PCA -------

layout(matrix(c(1,2), ncol=2))
resPCA <- PCA(mat_df, scale.unit = FALSE)
plot.PCA(resPCA, choix="ind", habillage = 9, label = "none")

# ----------------------------------------


# ------- NMF -------
res_nmf <- nmf(mat_df, 4000)

# ----------------------------------------


# ------- WC-NMTF -------
res_wc_nmtf <- read.csv("wc-nmtf_Z_l1_cos_xnorm.csv", header = TRUE)
label_res <- apply(res_wc_nmtf, MARGIN = 1, FUN=which.max)

layout(matrix(1:2))
plot(labelK)
plot(label_res)

NMI(label_res, labelK)
ARI(label_res, labelK)

# ----------------------------------------


# ----------------------------------------

#library(cluster)
#library(factoextra)
#fviz_cluster(res, data = df)
