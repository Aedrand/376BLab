#Andrew Riggs
#CS376B, Spring 2017
#Lab 8

###Setup###
#Install packages#
install.packages("car")
install.packages("lattice")
install.packages("Hmisc")
install.packages("caret")
install.packages("RWeka")
install.packages("rpart")
install.packages("partykit")
install.packages("arules")
install.packages("e1071")

#Load libraries#
library(car)
library(lattice)
library(Hmisc)
library(caret)
library(RWeka)
library(rpart)
library(partykit)
library(arules)
library(e1071)

#Load the training and test data
train <- read.csv("Lab8.Train.csv")
test <- read.csv("Lab8.Test.csv")

###Summary and Structure###
#Report on structure for train and summarize
writeLines("Structure report for training set:")
str(train)
writeLines("Summary of training set:")
summary(train)

#Report on structure for test and summarize
writeLines("Structure report for testing set:")
str(train)
writeLines("Summary of testing set:")
summary(train)

###Data Preparation###
#Create sets of data where the target is nominal to allow C4.5 creation
#Remove id column and a2-7
train.nominal <- train[,2:13]
train.nominal$a1 <- as.factor(train.nominal$a1)
test.nominal <- test[,2:13]
test.nominal$a1 <- as.factor(test.nominal$a1)

#Remove id column and a1,a3-7
train.nominal2 <- train[,2:14]
train.nominal2$a1 <- NULL
train.nominal2$a2 <- as.factor(train.nominal2$a2)
test.nominal2 <- test[,2:14]
test.nominal2$a1 <- NULL
test.nominal2$a2 <- as.factor(test.nominal2$a2)

#Create a combined dataset for binning
whole <- rbind(train,test)
whole2 <- rbind(train,test)

#Use equal frequency binning to bin a1 in whole and a2 in whole2
whole$a1 <- discretize(whole$a1, categories = 10)
whole2$a2 <- discretize(whole$a2, categories = 10)

#Remove the target features that are not being used in each set
whole <- whole[,2:13]
whole2 <- whole2[2:14]
whole2$a1 <- NULL

#Split the whole dataset into training and testing again
set.seed(1)
split <- createDataPartition(whole$a1, p=0.6)[[1]]
split.train <- whole[split,]
split.test <- whole[-split,]
split2 <- createDataPartition(whole2$a2, p=0.6)[[1]]
split.train2 <- whole2[split2,]
split.test2 <- whole2[-split2,]
rm(split)
rm(split2)

###Model Creation###
##C4.5##
# Build a decision tree for a1 using C4.5 (Weka's J48 implementation)
train.model.nom <- J48(a1 ~ ., data=train.nominal)

# View details of the constructed tree
#summary(train.model.nom)

# Plot the decision tree
#plot(train.model.nom)

# Build a decision tree for a2 using C4.5 (Weka's J48 implementation)
train.model.nom2 <- J48(a2 ~ ., data=train.nominal2)

# View details of the constructed tree
#summary(train.model.nom2)

# Plot the decision tree
#plot(train.model.nom2)

##RIPPER##
#Create RIPPER models for a1 and a2 using the split training set
train.model.rules <- JRip(a1 ~ ., data=split.train)
train.model.rules2 <- JRip(a2 ~ ., data=split.train2)

#Print the models
writeLines("RIPPER Model created for a1:")
print(train.model.rules)
writeLines("RIPPER Model created for a2:")
print(train.model.rules2)

###Evaluation###
##C4.5##--------------------------------------------------------------------------NOT WORKING
#Create predicitions for a1 and a2 using the test set
test.predict.nom <- predict(train.model.nom, test.nominal)

#Calculate the performance of a1 and a2 C4.5 models
test.eval.nom <- confusionMatrix(test.predict.nom, test.nominal$a1)

##RIPPER##
#Create predictions for a1 and a2 using the new test set
test.predict.rules <- predict(train.model.rules, split.test)
test.predict.rules2 <- predict(train.model.rules2, split.test2)

#Calculate the performance of the a1 and a2 RIPPER models
test.eval.rules <- confusionMatrix(test.predict.rules, split.test$a1)
test.eval.rules2 <- confusionMatrix(test.predict.rules2, split.test2$a2)

print(test.eval.rules)
