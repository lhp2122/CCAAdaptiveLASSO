# CCAAdaptiveLASSO
# Example Adaptive LASSO Code used for CCA Research. Modeled after Yoshida's code https://rpubs.com/kaz_yos/alasso and Carvalho's https://ricardocarvalho.ca/post/lasso/  

# LASSO Regression
# This study aimed to explore the conditions that influenced an instructor in their prescriptive (top-down) or elicitive (bottom-up) approach
# Using LASSO Regression to generate a predictive model rather than Stepwise
# Packages
library(lattice)
library(psych)
library(mlbench)
library(caret)
library(glmnet)
library(mice)

# Create different data set for each DVs
# There are too many DVs so I will only take DV1 as an example; the same process can be applied to the remaining
dv1dat <- as_tibble(select(mydata, 7, 1:6)) # The first 6 IVs and 1 DV
# Put the DV in the front for convenience later on

# Impute missing data for LASSO analysis
# Load mice package
md.pattern(dv1dat)

# Imputing with 5 imputed data sets and 50 iterations
# Method: predictive mean matching
impute01 <- function(x){
  fivetimes <- mice(x, m=5, maxit = 50, method = 'pmm', seed = 1234) 
  x <- complete(fivetimes,4) #select set number 4 out of 5 imputed sets
}
dv1dat <- impute01(dv1dat)

# Summary Matrix of the IVs
dv1dat[c(-1)] %>%
  pairs.panels(cex = 2)

# Data partition; separate into a training set and a testing set

set.seed(1234) # Consistent randomization 
dv1ind <- sample(2, nrow(dv1dat), replace = T, prob = c(0.7, 0.3)) #splitting the data into 2 groups
# One group is 70% of the data is the training set
dv1train <- dv1dat[dv1ind == 1,]
# The remaining 30% of the data is the testing set
dv1test <- dv1dat[dv1ind == 2,]

# Custom control parameters
custom <- trainControl(method = "repeatedcv",
                       number = 10,
                       repeats = 5,
                       verboseIter = T)

# LASSO Process
# Work up a LASSO regression fit
# Already set seed above 1234
# Call it LASSO to separate from Adaptive LASSO later
dv1lasso <- train(DV1_traineffect ~.,
                  dv1train,
                  method = 'glmnet',
                  tuneGrid = expand.grid(alpha = 1, lambda = seq(0.001, 1, length = 5)),
                  trControl = custom) 

# Summary results of the model generated
dv1lasso
plot(dv1lasso$finalModel, xvar = 'dev', label = T) #DV1 LASSO Model plot

# Which predictors are in the model?
plot(varImp(dv1lasso, scale = T)) #use this to visually compare to adaptive LASSO 
dv1lasso$bestTune #the final alpha and lambda used for the model
# Instructor's cultural familiarity, collectivist orientation, frequency of practice, and time spent in cultural context
# Are significant predictors of training effectiveness
coef(dv1lasso$finalModel, dv1lasso$bestTune$lambda) #coefficient results

# Save
saveRDS(dv1lasso$finalModel, "dv1lasso.rds")

# Prediction testing for training set
p1dv1 <- predict(dv1lasso, dv1train)
sqrt(mean((dv1train$DV1_traineffect - p1dv1)^2))
# The model can predict 87% of the training set

# Prediction testing for test set
p2dv1 <- predict(dv1lasso, dv1test)
sqrt(mean((dv1test$DV1_traineffect - p2dv1)^2))
# The model can predict 90% of the training set
# This LASSO model has high predictability

# Adaptive LASSO
# Why not LASSO?
# According to stats enthusiasts, LASSO often includes too many variables when selecting the tuning paramenter
# The true model is very likely a subset of these variables
# Adaptive LASSO controls the bias by using the prediction-optimal tuning parameter
# "Adaptive LASSO and its oracle properties" (Zou, 2006)
summary(dv1dat) 
# Set up matrix
x1 <- as.matrix(dv1dat[,-1]) #Choose all but the DV
y1 <- as.double(as.matrix(dv1dat[,1])) #Only choose DV

# Run Ridge Regression to find the optimal tuning parameter
# I cannot say that I understand 100% of the process
# Run Ridge Regression to find the optimal tuning parameter
ridgepenalty <- function(x, y) {
  set.seed(1234)
  ridgeval <- cv.glmnet(x, y, alpha = 0, standardize = TRUE)
  w <- 1/abs(matrix(coef(ridgeval, s = ridgeval$lambda.min)[,1][2:(ncol(x)+1)]))^1 #gamma=1
}
w1 <- ridgepenalty(x1,y1)
w1[w1[,1] == Inf] <- 99999999 # Penalty value

# Adaptive LASSO
# Adaptive LASSO
adaptlasso <- function(x, y ,w){
  cv.glmnet(x, y, alpha = 1, standardize = TRUE,
            type.measure = 'mse',
            penalty.factor = w)
}
dv1adaptlasso <- adaptlasso(x1, y1, w1)
plot(dv1adaptlasso)
plot(dv1adaptlasso$glmnet.fit, xvar = "lambda", label = TRUE)
abline(v = log(dv1adaptlasso$lambda.min)) # Using min value of lambda that gives minimum mean cross-validated error
abline(v = log(dv1adaptlasso$lambda.1se)) # lambda 1 Standard Error from the minimum

coef(dv1adaptlasso, s = dv1adaptlasso$lambda.min) # 
coef1 <- coef(dv1adaptlasso, s = 'lambda.min')
saveRDS(coef1, "dv1adaptlasso.rds")
# LASSO selected 4 variables in the prediction model, while adaptive LASSO selected 2
