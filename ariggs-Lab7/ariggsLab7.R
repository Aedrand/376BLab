#Andrew Riggs
#CS376B, Spring 2017
#Lab 7

#SETUP
#Set up the necessary libraries
library(corrplot)
library(aplpack)
library(modes)
library(OneR)
library(mlbench)
library(e1071)
library(caret)

#Load the diabetes data set from diabetes.csv
diabetes <- read.csv('diabetes.csv')

#Set up another dataset where the test result is an integer
#Maps positive to 1, negative to 0
intdiabet <- diabetes
intdiabet$class <- as.integer(intdiabet$class)
intdiabet[intdiabet$class == 1,]$class <- 0
intdiabet[intdiabet$class == 2,]$class <- 1

#SUMMARIES
#Report on the structure of the diabetes data set
str(diabetes)

#Summary for the entire dataset
summary(diabetes)

#Create variable summaries for each of the variables in diabetes
for(varnum in 1:ncol(diabetes))
{
  writeLines(paste("Summary for", colnames(diabetes)[varnum],sep = " "))
  print(summary(diabetes[,varnum]))
}

#PLOT1
#Create a scatter plot matrix for the entire dataset
png("plotmatrix.png", width = 1920, height = 1080)
plot(diabetes, main = "Scatter Plot Matrix for Diabetes")
dev.off()

#Create a correlation plot for the entire dataset
png("corrplot.png", width = 1920, height = 1080)
corIntDiabet <- cor(intdiabet)
corrplot(corIntDiabet, main = "Correlation Plot for Diabetes")
dev.off()
rm(corIntDiabet)

#CREATE TRAINING AND TEST SETS
# Apply supervised binning to the continuous data
intdiabet.binned <- optbin(intdiabet)

# Create separate training and test sets
# Use a 60:40 split of data for train:tes
set.seed(0)
trainSet <- sample(seq_len(nrow(intdiabet.binned)), nrow(intdiabet.binned) * .6)
intdiabet.binned.train <- intdiabet.binned[trainSet,]
intdiabet.binned.test <- intdiabet.binned[-trainSet,]

#PLOT2
# Verify the class splits in the training set
# to explain the priors in the model
# Instead of raw counts, lets look at proportions
tbl.data <- table(intdiabet$class) / nrow(intdiabet)
tbl.train <- table(intdiabet.binned.train$class) / nrow(intdiabet.binned.train)
tbl.test <- table(intdiabet.binned.test$class) / nrow(intdiabet.binned.test)

# Plot the original and training set class proportions
png("classprop.png", width = 1920, height = 1080)
op <- par(mfrow=c(1,3))
bp <- barplot(tbl.data, 
              ylim=c(0, 1),
              main = "Class Proportions in the\nDiabetes Data Set",
              xlab="Proportion",
              ylab="Class")
text(x = bp, y = tbl.data, label = round(tbl.data, digits=2), 
     pos = 3, cex = 2, col = "red")

bp <- barplot(tbl.train, 
              ylim=c(0, 1),
              main = "Class Proportions in the\nDiabetes Training Set",
              xlab="Proportion",
              ylab="Class")
text(x = bp, y = tbl.train, label = round(tbl.train, digits=2), 
     pos = 3, cex = 2, col = "red")

bp <- barplot(tbl.test, 
              ylim=c(0, 1),
              main = "Class Proportions in the\nDiabetes Testing Set",
              xlab="Proportion",
              ylab="Class")
text(x = bp, y = tbl.test, label = round(tbl.test, digits=2), 
     pos = 3, cex = 2, col = "red")
par(op)
dev.off()

#ONER
# Create a 1R classification model
intdiabet.oner.model <- OneR(intdiabet.binned.train, verbose = TRUE)

# Look at the raw model (e.g. the tree's decisions)
print(intdiabet.oner.model)

# Show the structure of the model
str(intdiabet.oner.model)

# Show details regarding the model
summary(intdiabet.oner.model)

# Create predictions based on the model
intdiabet.oner.pred <- predict(intdiabet.oner.model, intdiabet.binned.test)

# Evaluate the model
eval_model(intdiabet.oner.pred, intdiabet.binned.test)

#NAIVE BAYES
# Create a Naive Bayes classification model
intdiabet.nb.model <- naiveBayes(class ~ .,data = intdiabet.binned.train)

# Look at the raw model 
# Note the priors for the classes and the conditional 
# probabilities for the DFs
print(intdiabet.nb.model)

# Show the structure of the model
str(intdiabet.nb.model)

# Show an overview of the model
summary(intdiabet.nb.model)

# Create predictions based on the model
intdiabet.nb.pred <- predict(intdiabet.nb.model, intdiabet.binned.test)

# Evaluate the model
eval_model(intdiabet.nb.pred, intdiabet.binned.test)

