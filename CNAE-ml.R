# Reading the data file and converting to csv format
df <- read.table(url("https://archive.ics.uci.edu/ml/machine-learning-databases/00233/CNAE-9.data"), sep = ',')
write.csv(df,"CNAE.csv")


# convert to categorical
jd <- as.factor(df)
df$V1 <- as.factor(df$V1)
library(caret)


# splitting
trainrows <- sample(nrow(df), 0.7*nrow(df))# sampling the data
trains = df[trainrows,]# splitting the train from sample
tests = df[-trainrows,]# splitting the test from sample

# Building Model : Logistic regression
library(glmnet)
log <- glmnet(trains[,-1],trains[,1], family = "multinomial")
pr <- predict(log, tests[,-1])
cx <- confusionMatrix(pr, tests[,1])
cx


# Building Model : Naive Bayes
library(e1071) #loading the library naive bayes
naive <- naiveBayes(train$V1 ~., data = trains) # building the model on train data
pre <- predict(naive, test[,-1])# Predicting the model on test data removing target variable
con <- confusionMatrix(pre, test$V1)# forming the confusion matrix on predicted and test target variable
con # Printing the results

# Building Model : SVM
# Polynomial
library(e1071)#loading the library for svm with kernel trick as polynomial
super <- svm(trains$V1 ~., data = trains, kernel = 'polynomial') # building the model on train data 
pred <- predict(super, tests[,-1])# Predicting the model on test data removing target variable
co <- confusionMatrix(pred,tests[,1]) # forming the confusion matrix on predicted and test target variable
co # Printing the results

# radial
man <- svm(trains$V1 ~., data = trains, kernel = 'radial')# building the model on train data  
pred <- predict(man, tests[,-1])# Predicting the model on test data removing target variable
com <- confusionMatrix(pred,tests[,1])# forming the confusion matrix on predicted and test target variable
com # Printing the results

# linear
superman <- svm(trains$V1 ~., data = trains, kernel = 'linear')# building the model on train data   
pred <- predict(superman, tests[,-1])# Predicting the model on test data removing target variable
cm <- confusionMatrix(pred,tests[,1])# forming the confusion matrix on predicted and test target variabl
cm # Printing the results

# Decision Trees
library(C50) #loading the library for decision tree
tree <- C5.0(trains$V1~., data = trains) # building the model on train data 
trepre <- predict(tree, tests[,-1]) # Predicting the model on test data removing target variable
con <- confusionMatrix(trepre,tests[,1])# forming the confusion matrix on predicted and test target variable
con # Printing the results

library(rpart)
part <- rpart(trains$V1~., data = trains) # building the model on train data
papre <- predict(tree, tests[,-1]) # Predicting the model on test data removing target variable
cma <- confusionMatrix(papre,tests[,1]) # forming the confusion matrix on predicted and test target variable
cma

# logistic Regression
library(nnet) # loading the library for Logistic Regression
model <- nnet(V1~.,data = trains,size=10,MaxNWts=80000,maxit=80000)# building the model on train data
modelpre <- predict(model,tests[,-1],type = "class")# Predicting the model on test data removing target variable
confu <- confusionMatrix(modelpre,tests[,1])# forming the confusion matrix on predicted and test target variable
confu # Printing the results

