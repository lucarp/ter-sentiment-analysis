library(R.matlab)

#setwd("/Users/lucasrodriguespereira/Projects/ter/mat_files/")
setwd(dirname(rstudioapi::getSourceEditorContext()$path))

mat_name <- paste(mat_path, "output_not_original_30.csv_tf-idf-l2.mat", sep = "")

X <- readMat(mat_name)

label_name <- paste(mat_path, "dataset_LABEL.csv", sep = "")
y <- read.csv(label_name, header = FALSE)
#y <- read.table("dataset_LABEL.csv")

df <- X$X
df<-as.data.frame(as.matrix(df))
dim(df)
df<-cbind(df,y)
colnames(df)<-c(1:9749,"y")
fit <- lm(y~., data=df)

summary(fit) # show results
