# Johns Hopkins University PML Project
# Antonio I. M. Guerreiro
# August 20, 2016

# The goal of the project is to predict the manner in which people did barbell lift exercises.  
# Additional information should have been available at http://groupware.les.inf.puc-rio.br/har  
# but this site related to Human Activity Recognition seems down now.  
# Training data from:  
# https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv  
# Quizz data from:   
# https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv  

# Required Libraries
# I use the "caret" package and, as random forests are quite computer intensive,  
# I also use the "doMC" package to run several cores on my machine.  
# It's very easy to install and use on Linux and OS X.
# http://topepo.github.io/caret/parallel.html
library(caret)
library(doMC)

# Preparing a tidy data set
# This is my way to get a tidy data set:
# remove all these empty or with NAs columns,
# keep only movement related columns plus "user_name" and "classe".
pdat<-read.csv("pml-training.csv", stringsAsFactors=FALSE, na.strings=c("","NA"))
pdat<-pdat[ , apply(pdat, 2, function(x) !any(is.na(x)))]
pdat<-subset(pdat,select=-c(X,new_window,num_window,raw_timestamp_part_1,raw_timestamp_part_2,cvtd_timestamp))
pdat$user_name <- as.factor(pdat$user_name); pdat$classe <- as.factor(pdat$classe)

# Some exploratory data analysis...
# Just curious to see if one user performs better by creating a spineplot...   
plot(pdat$user_name, pdat$classe, xlab="User", ylab="Classe")

# Splitting the project data for building & evaluation of the model
inTrain <- createDataPartition(y = pdat$classe, p = 0.7, list = FALSE)
train_data <- pdat[inTrain,]; test_data <- pdat[-inTrain,]

# Trying first the caret "rpart" method 
# "classe" is a factor with 5 levels so I try first a quick recursive partitioning method for classification.  
# I include all features. It uses random sampling with replacement (bootstrap).  
# Try no preprocessing as it's not a linear model.
modFit_rpart <- train(classe ~ ., data = train_data, method = "rpart")
modFit_rpart

# What about performance on the test set?
pred <- predict(modFit_rpart, test_data)
confusionMatrix(pred, test_data$classe)

# Fitting a random forest model using all the features of the training tidy data set
# 5 cores on a 2.26 GHz old mac pro with 20 GB RAM.  
# I use the defaults for the method, so it's a random sampling with replacement (bootstrap), 
# and all features. It takes around 20 min to complete.
registerDoMC(cores = 5)
modFit_rf <- train(classe ~ ., data = train_data, method = "rf")
modFit_rf

## Evaluating out of sample error using the tidy data set part kept for testing
pred <- predict(modFit_rf, test_data)
confusionMatrix(pred, test_data$classe)

# Applying this model to the prediction of Quizz 20 test cases. 
qdat<-read.csv("pml-testing.csv", stringsAsFactors=FALSE, na.strings=c("","NA"))
qdat<-qdat[ , apply(qdat, 2, function(x) !any(is.na(x)))]
qdat<-subset(qdat,select=-c(X,new_window,num_window,raw_timestamp_part_1,raw_timestamp_part_2,cvtd_timestamp))
qdat$user_name <- as.factor(qdat$user_name)
pred <- predict(modFit_rf, qdat)
pred

# Variable importance
rfImp <- varImp(modFit_rf)
rfImp
