setwd("/media/matthieu/Data/Matthieu/##Etude/#M1/S2/TER/ter-sentiment-analysis")

library(R.matlab)
X <- readMat("mat_files/output_not_original.csv_tf-idf-l2.mat")
df <- X$X
df

# K-means
#res <- kmeans(df, 5)

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

# Spherical K-means
library(skmeans)
x <- skmeans(df, 5)


#library(cluster)
#library(factoextra)
#fviz_cluster(res, data = df)
