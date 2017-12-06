df <- read.table(url("https://archive.ics.uci.edu/ml/machine-learning-databases/00233/CNAE-9.data"), sep = ',')
dplyr::glimpse(df)
write.csv(df,"CNAE.csv")

 #libraries
library(tm)
library(plyr)
library(class)
library(caret)
# convert to categorical
jk <- lapply(df,as.factor)
jk <- as.data.frame(jk)

df$V1 <- as.factor(df$V1)

# Feature Extraction & Labels for the model
corp <- Corpus(VectorSource(df))
tdm <- TermDocumentMatrix(corp)
# splitting of the data
set.seed(123)
trainrows <- sample(nrow(jk), 0.7*nrow(jk))
train = jk[trainrows,]
test = jk[-trainrows,]


# splitting
trainrows <- sample(nrow(df), 0.7*nrow(df))
trains = df[trainrows,]
tests = df[-trainrows,]

# Building Model : Naive Bayes
library(e1071)
na <- naiveBayes(train$V1 ~., data = train)
pre <- predict(na, test[,-1])
con <- confusionMatrix(pre, test$V1)
con

# Building Model : SVM
# Polynomial
library(e1071)
super <- svm(trains$V1 ~., data = trains, kernel = 'polynomial')  
pred <- predict(super, tests[,-1])
dj <- table(pred,tests[,1])
co <- confusionMatrix(dj)
co

# radial
man <- svm(trains$V1 ~., data = trains, kernel = 'radial')  
pred <- predict(man, tests[,-1])
jd <- table(pred,tests[,1])
com <- confusionMatrix(jd)
com

# linear
superman <- svm(trains$V1 ~., data = trains, kernel = 'linear')  
pred <- predict(superman, tests[,-1])
dj <- table(pred,tests[,1])
cm <- confusionMatrix(dj)
cm

# Decision Trees
library(C50)
tree <- C5.0(trains$V1~., data = trains) 
trepre <- predict(tree, tests[,-1])
con <- confusionMatrix(trepre,tests[,1])
con


# logistic Regression
library(nnet)
model <- nnet(V1~.,data = trains,size=10,MaxNWts=80000,maxit=80000)
modelpre <- predict(model,tests[,-1],type = "class")
confu <- confusionMatrix(modelpre,tests[,1])
confu

