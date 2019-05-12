library(R.matlab)

setwd("/Users/lucasrodriguespereira/Projects/ter/mat_files/")
X <- readMat("output_not_original_10.csv_tf-idf-l2.mat")
y <- read.table("dataset_LABEL.csv")
df <- X$X
df<-as.data.frame(as.matrix(df))
dim(df)
df<-cbind(df,y)
colnames(df)<-c(1:9749,"y")
fit <- lm(y~., data=df)

summary(fit) # show results
