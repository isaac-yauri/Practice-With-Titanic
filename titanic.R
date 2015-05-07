setwd("c:/emc/cursos/Kaggle")

train <- read.csv("C:/EMC/Cursos/Kaggle/train.csv")
test <- read.csv("C:/EMC/Cursos/Kaggle/test.csv")

str(train)

library(rpart)
library(rpart.plot)
library(caTools)
library(ROCR)
library(caret)
library(e1071)
library(randomForest)
library(flexclust)

train$Survived <- as.factor(train$Survived)
train$Pclass <- as.factor(train$Pclass)

test$Survived <- as.factor(test$Survived)
test$Pclass <- as.factor(test$Pclass)


set.seed=200
split <- sample.split(train$Survived, SplitRatio = 0.7)
Titanictrain <- subset(train, split == TRUE)
Titanictest <- subset(train, split == FALSE)


TitanicRF <- randomForest(Survived ~ Pclass + Sex + SibSp + Parch, data = Titanictrain, nodesize = 24, ntree = 2000)
PredictRF <- predict(TitanicRF, newdata = Titanictest)
tabla <- table(Titanictest$Survived, PredictRF)
acc <- (tabla[1,1] + tabla[2,2])/nrow(Titanictest)
cat("acc=",acc,"\n")
tabla


PredTest <- predict(TitanicRF, newdata=test, type="response")
MySubmission <- data.frame(PassengerID = test$PassengerId, Survived = PredTest)
write.csv(MySubmission, "Submission.csv", row.names=FALSE)
