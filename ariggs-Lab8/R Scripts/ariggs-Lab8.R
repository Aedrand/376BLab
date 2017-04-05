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
#Create a combined dataset
whole <- rbind(train,test)
whole2 <- rbind(train,test)

#Use equal frequency binning to bin a1 in whole and a2 in whole2
whole$a1 <- discretize(whole$a1, "frequency", categories = 5)
whole2$a2 <- discretize(whole$a2, "frequency", categories = 5)

#Remove the target features that are not being used in each set
whole <- whole[,2:13]
whole2 <- whole2[2:14]
whole2$a1 <- NULL

#Split the whole dataset into training and testing again
split.train <- whole[1:198,]
split.test <- whole[199:338,]
split2.train <- whole2[1:198,]
split2.test <- whole2[199:338,]

#Remove unneeded datasets
rm(whole)
rm(whole2)

###Model Creation###
##C4.5##
# Build a decision tree for a1 using C4.5 (Weka's J48 implementation)
train.model.nom <- J48(a1 ~ ., data=split.train)

# View details of the constructed tree
summary(train.model.nom)

# Plot the decision tree
png("a1C4.5Tree.png", width = 1000, height = 1000)
plot(train.model.nom)
dev.off()

# Build a decision tree for a2 using C4.5 (Weka's J48 implementation)
train.model.nom2 <- J48(a2 ~ ., data=split2.train)

# View details of the constructed tree
summary(train.model.nom2)

# Plot the decision tree
png("a2C4.5Tree.png", width = 1000, height = 1000)
plot(train.model.nom2)
dev.off()

##RIPPER##
#Create RIPPER models for a1 and a2 using the split training set
train.model.rules <- JRip(a1 ~ ., data=split.train)
train.model.rules2 <- JRip(a2 ~ ., data=split2.train)

#Print the models
writeLines("RIPPER Model created for a1:")
print(train.model.rules)
writeLines("RIPPER Model created for a2:")
print(train.model.rules2)

###Evaluation###
##C4.5##
#Create predicitions for a1 and a2 using the test set
test.predict.nom <- predict(train.model.nom, split.test)
test.predict.nom2 <- predict(train.model.nom2, split2.test)

#Calculate the performance of a1 and a2 C4.5 models
test.eval.nom <- confusionMatrix(test.predict.nom, split.test$a1)
test.eval.nom2 <- confusionMatrix(test.predict.nom2, split2.test$a2)

#Print the result of the evaluations
writeLines("C4.5 a1:")
print(test.eval.nom)
writeLines("C4.5 a2:")
print(test.eval.nom2)

##RIPPER##
#Create predictions for a1 and a2 using the new test set
test.predict.rules <- predict(train.model.rules, split.test)
test.predict.rules2 <- predict(train.model.rules2, split2.test)

#Calculate the performance of the a1 and a2 RIPPER models
test.eval.rules <- confusionMatrix(test.predict.rules, split.test$a1)
test.eval.rules2 <- confusionMatrix(test.predict.rules2, split2.test$a2)

#Print the result of the evaluations
writeLines("RIPPER a1:")
print(test.eval.rules)
writeLines("RIPPER a2:")
print(test.eval.rules2)