#setwd("/media/matthieu/Data/Matthieu/##Etude/#M1/S2/TER/ter-sentiment-analysis")
#setwd("/home/lucarp/master/ter")
setwd(dirname(rstudioapi::getSourceEditorContext()$path))

library(aricode)
library(R.matlab)
library("FactoMineR")
library(factoextra)

# ---------- Read Dataset ----------
normalize <- function(x) {x / sqrt(rowSums(x^2))}
X <- readMat("mat_files/output_not_original_10.csv_tf-idf-l2.mat")

df <- X$X
dim(df)
mat_df <- as.matrix(df)
mat_df <- normalize(mat_df)
dim(mat_df)
label <- read.csv("mat_files/dataset_LABEL.csv", header = FALSE)
#label

k <- 5
labelK <- apply(label, MARGIN = 1, FUN=function(x) max(1, ceiling(x*k))) # true label (1 to k)

#mat_df <- cbind(mat_df, labelK)

mat_df <- mat_df[1:100,]
labelK <- labelK[1:100]
dim(mat_df)
mat_df <- mat_df[,(!colSums(mat_df) == 0)]
dim(mat_df)
# ------------------------------


# ---------- Correspondence Analysis ----------
print("run CA...")
res_CA <- CA(mat_df)
print("done")

#saveRDS(res_CA, file = "CA.rds")

res_CA <- readRDS(file = "CA.rds")

authors = as.factor(c(rep("Dennis Schwartz",1027), rep("James Berardinelli", 1307), rep("Scott Renshaw", 902), rep("Steve Rhodes", 1770)))
labelK <- as.factor(labelK)

svg("CA_authors.svg")
#fviz_ca_row(res_CA, label = "none", col.row = labelK, addEllipses = TRUE, xlim=c(-0.5, 0.5), ylim=c(-1.5, 1.5))
fviz_ca_row(res_CA, label = "none", col.row = authors, addEllipses = TRUE, xlim=c(-0.5, 0.5), ylim=c(-1.5, 1.5))
dev.off()
# ------------------------------