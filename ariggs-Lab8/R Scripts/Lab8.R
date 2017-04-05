#############################################################
# A script to go through the process of data understanding, #
# data preparation, modeling, evaluation and documenting    #
# the choice of model for deployment                        #
#                                                           #
# This file serves as a template for the lab 8 assignment   #
# using the algae data set. Due to the simplicity of the    #
# iris data set, no feature selection or data preparation   #
# work was done. For the algae data you should use basic    #
# feature selection and preparation steps (such as          #
# removing unimportant features or scaling features).       #
#############################################################

# Note that you can run this script from within R using the
# source command. e.g. source("Lab8.R")


##################
## Preconditions #
##################

# Lab4.iris.data.csv files in R's working directory 
# for this script to work properly


####################################
## Install packages - if necessary #
####################################

# If any of these following packages have not been installed
# uncomment and run the necessary package installations

install.packages("car")
install.packages("lattice")
install.packages("Hmisc")
install.packages("caret")
install.packages("RWeka")
install.packages("rpart")


###################
## Load libraries #
###################

library(car)
library(lattice)
library(Hmisc)
library(caret)
library(RWeka)
library(rpart)


##################
## Load the data #
##################

iris <- read.csv("Lab4.iris.data.csv")


####################################
## Data Understanding              #
##                                 #
## Visualization and Summarization #
####################################

# Summarize the features
summary(iris)

# Plot a histogram of sepal length
# Showing probabilities rather than frequencies
hist(iris$sepal.length, prob=TRUE, main="Histogram of\nIris Sepal Length")

# Combine a histogram with a density curve and "rug" showing 
# the data value distribution
# Add a Quantile-Quantile (Q-Q) plot to check sepal length 
# distribution against a normal distribution
op <- par(mfrow=c(1,2))
hist(iris$sepal.length, prob=TRUE, main="Histogram of\nIris Sepal Length", ylim=0:1)
lines(density(iris$sepal.length, na.rm=TRUE))
rug(jitter(iris$sepal.length))
qqPlot(iris$sepal.length, main="Normal QQ Plot of\nIris Sepal Length")
par(op)

# Combine a boxplot of petal width with a rug, again to show the 
# distribution of values, and a dashed line showing the mean
boxplot(iris$petal.width, main="Box plot of\nIris Petal Width")
rug(jitter(iris$petal.width), side=2)
abline(h=mean(iris$petal.width, na.rm=TRUE), lty=2)

# Identify a specific instance based on a plotted value
# Use the left mouse button to select values and
# retrieve the instance row number in the data frame
# Click the right mouse button to exit interactive mode
# Note that the identify function returns a vector of the 
# selected row numbers
i = iris$sepal.width
plot(i, xlab="Sepal Width")
abline(h=mean(i, na.rm=TRUE), lty=1)
abline(h=mean(i, na.rm = TRUE) + sd(i, na.rm=TRUE_, lty=2)
abline(h=median(i, na.rm=TRUE), lty=3)
chosen <- identify(i)
# Show the selected instances
print(iris[chosen,])


# Use the lattice library to create a 
# boxplot of sepal length conditioned on species
bwplot(species ~ sepal.length, data=iris, ylab="Species", xlab="Sepal Length")

# Use the lattice and Hmisc libraries to create a box 
# percentile plot of sepal length conditioned on species
bwplot(species ~ petal.length, data=iris, 
       panel=panel.bpplot, probs=seq(.01, .49, by=.01),
       datadensity=TRUE,
       ylab="Species", xlab="Petal Length")

# Discretize petal width and then create a strip plot
# of species conditioned on petal length and petal width
PetalWidth <- equal.count(na.omit(iris$petal.width), number=4, overlap=1/5)
stripplot(species~petal.length|PetalWidth, data=iris[!is.na(iris$petal.width),],
          main="Strip Plot of Iris Species\n (conditioned on petal height and width)")

# View a histogram of petal lengths conditioned by species
histogram(~petal.length | species, data=iris, 
          main="histogram of Iris Petal Lengths\n(conditioned by species")

# Show the histograms with setosa last (top)
iris$species <- factor(iris$species, levels=c("Iris-versicolor", "Iris-virginica", "Iris-setosa"))
histogram(~petal.length | species, data=iris, 
          main="histogram of Iris Petal Lengths\n(conditioned by species")


# Show class distribution
iris.species.table <- table(iris$species) / nrow(iris)
bp <- barplot(iris.species.table, 
	ylim=c(0, 1),
	main = "Class Proportions in the\nIris Data Set",
	xlab="Proportion",
	ylab="Class")
text(x = bp, y = iris.species.table, label = round(iris.species.table, digits=2), 
	pos = 3, cex = 0.8, col = "red")

###########################################################
## Data Preparation                                       #
##                                                        #
## Includes feature selection and transformations such as #
## discretization, normalization, standardization...      #
###########################################################

# No feature selection was done on the Iris data set

# No changes were made to the data set prior to building the classifiers


################################################
## Modeling                                    #
##                                             #
## Decision Tree, Rule Set and Regression Tree #
################################################

# Split iris data into train and test sets
set.seed(1)
trainSet <- createDataPartition(iris$species, p=.6)[[1]]
iris.train <- iris[trainSet,]
iris.test <- iris[-trainSet,]

# Build a decision tree for species using C4.5 (Weka's J48 implementation)
iris.model.nom <- J48(species ~ ., data=iris.train)

# View details of the constructed tree
# Be sure you understand each of the reported measures
# (how it is calculated and what it means)
summary(iris.model.nom)

# Plot the decision tree
plot(iris.model.nom)

# Create a regression tree for petal.length using rpart (CART implementation)
iris.model.reg <- rpart(petal.length ~ ., data=iris.train[,1:4])

# View details of the constructed tree
# Be sure you understand what the output means
summary(iris.model.reg)

# Plot the regression tree
plot(iris.model.reg, uniform=TRUE,
   main="Regression Tree for Iris Petal Length")
text(iris.model.reg, use.n=TRUE, all=TRUE, cex=.8)

# Attempt post-pruning of the regression tree to see if a better
# classifier can be created. The approach repeatedly prunes the tree
# generating a set of iteratively pruned trees and displays the cost
# complexity (CP) along with error rates of each
# Note that rpart will calculate the error rate 
# using 10-fold cross validation
printcp(iris.model.reg)

# Obtain one of the pruned trees using the cp value (round up)
# This will retrieve the 2nd tree (cp:0.082403)
iris.model.reg.prune <- prune(iris.model.reg, cp=0.09)

# Verify it is the correct pruned tree (will be last listed)
printcp(iris.model.reg.prune)

# Interactively (manually) prune a tree
# Click once with the left mouse button to shows the impact
# of removing the node. Click the left mouse button a second
# time on the same node to actually remove it.
# Click with the right mouse button to end the process
plot(iris.model.reg, uniform=TRUE,
   main="Regression Tree for Iris Petal Length")
text(iris.model.reg, use.n=TRUE, all=TRUE, cex=.8)
iris.model.reg.manualprune <- snip.rpart(iris.model.reg)

# Display the pruned tree (assuming the user chose to remove any nodes)
plot(iris.model.reg.manualprune, uniform=TRUE,
   main="Regression Tree for Pruned Iris Petal Length")
text(iris.model.reg.manualprune, use.n=TRUE, all=TRUE, cex=.8)

#
# Build a Rule Set Using RIPPER
# Remember that RIPPER creates a default rule for the majority class
# and then creates rules to cover the other classes
#

# Build the rule set
iris.model.rules <- JRip(species ~ ., data=iris.train)

# Display the rule set
print(iris.model.rules)


###############
## Evaluation #
###############

# Create predictions from the decision tree model using the test set
iris.predict.nom <- predict(iris.model.nom, iris.test)

# Calculation of performance for nominal values uses a confusion matrix
# and related measures. 
iris.eval.nom <- confusionMatrix(iris.predict.nom, iris.test$species)

# Display the evaluation results for the decision tree
# You should understand all of these measures (calculation and meaning)
print(iris.eval.nom)

# Create predictions from the rule set using the test set
iris.predict.rules <- predict(iris.model.rules, iris.test)

# Calculation of performance for nominal values uses a confusion matrix
# and related measures.
iris.eval.rules <- confusionMatrix(iris.predict.rules, iris.test$species)

# Display the evaluation results for the rule set
# You should understand all of these measures (calculation and meaning)
# Notes: sensitivity is a synonym for recall;
#        positive predictive value is a synonym for precision
print(iris.eval.rules)

# Create predictions from the regression tree model using the test set
iris.predict.reg <- predict(iris.model.reg, iris.test)

# Need to use a numeric measures for accuracy of a regression tree
# MSE
iris.predict.reg.mse <- mean((iris.predict.reg - iris.test$petal.length)^2)
print(paste("Mean Squared Error (MSE):", iris.predict.reg.mse))

# RMSE
iris.predict.reg.rmse <- sqrt(iris.predict.reg.mse)
print(paste("Root Mean Squared Error (RMSE)", iris.predict.reg.rmse))

# MAE
iris.predict.reg.mae <- mean(abs(iris.predict.reg - iris.test$petal.length))
print(paste("Mean Absolute Error (MAE):", iris.predict.reg.mae))

# Plot the predictions vs. the actuals and add a line showing 
# The location for a "perfect" model where each prediction
# would equal the actual value (e.g. line y=x; a.k.a. 0-intercept, slope 1)
plot(iris.predict.reg, iris.test$petal.length,
     main="Regression Tree Preditions vs. Actual",
     xlab="Predicted", ylab="Actual")
abline(0, 1, lty=2)
legend("topleft", c("Data", "Perfect Model"), pch=c(1, NA), lty=c(NA, 2))


###################################################################
## Deployment                                                     #
##                                                                #
## Choose the best model for deployment based on the evaluations. #
## You should draw a conclusion as to which model is better       #
## (between the decision tree and rule set) using measures like   #
## accuracy vs. recall vs. precision vs. FPR                      #
###################################################################


