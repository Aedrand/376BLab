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

#Load libraries#
library(car)
library(lattice)
library(Hmisc)
library(caret)
library(RWeka)
library(rpart)
library(partykit)
library(arules)

#Load the training and test data#
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
#Create sets of data where the target is nominal to allow C4.5 creation#
#Remove id column and a2-7
train.nominal <- train[,2:13]
train.nominal$a1 <- as.factor(train.nominal$a1)

#Remove id column and a1,a3-7
train.nominal2 <- train[,2:14]
train.nominal2$a1 <- NULL
train.nominal2$a2 <- as.factor(train.nominal2$a2)

#Create a combined dataset for binning#
whole <- rbind(train,test)
whole2 <- rbind(train,test)

#Use equal frequency binning to bin a1
whole$a1 <- discretize(whole$a1, categories = 10)
whole2$a2 <- discretize(whole$a2, categories = 10)

#Split the whole dataset into training and testing again

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


