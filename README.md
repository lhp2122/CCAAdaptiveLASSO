# CCAAdaptiveLASSO
# Example Adaptive LASSO Code used for CCA Research. Modeled after Yoshida's code https://rpubs.com/kaz_yos/alasso and Carvalho's https://ricardocarvalho.ca/post/lasso/  

#import data
DV3Survey2 <- read_excel("Desktop/Academics/PhD Research/Cross Cultural Adaptivity/SPSS/DV3Survey2.xlsx")
View(DV3Survey2)                                                         
dv3dat <- data.frame(DV3Survey2)

#set up matrix
x3 <- as.matrix(dv3dat[,-1]) #dependent variable is on the first column #choose all but DV
y3 <- as.double(as.matrix(dv3dat[,1])) #only choose DV

#run Ridge regression
set.seed(1234)
dv3ridge <- cv.glmnet(x3,y3, alpha = 0, standardize = TRUE)
w3 <- 1/abs(matrix(coef(dv3ridge, s = dv3ridge$lambda.min) [,1][2:(ncol(x3)+1)]))^1 #gamma = 1
w3[w3[,1]==Inf] <- 99999999

#Adaptive Lasso
set.seed(1234)
dv3lasso <- cv.glmnet(x3,y3, alpha = 1, standardize = TRUE, type.measure ='mse', penalty.factor = w3)
plot(dv3lasso)
plot(dv3lasso$glmnet.fit, xvar = "lambda", label = TRUE)
abline(v = log(dv3lasso$lambda.min))
abline(v = log(dv3lasso$lambda.1se))

coef(dv3lasso, s=dv3lasso$lambda.1se) #produce significant predictors
coef3 <- coef(dv3lasso, s = 'lambda.1se')
saveRDS(coef3, "dv3adlasso.rds")


#get model stats (only significant predictors)
dv3step = lm(DV3_PEProcess ~ only significant predictor, data = dv3dat)
#get model (all predictors just in case)
dv3step3 = lm(DV3_PEProcess ~ Gender+IV1_insculfam+IV2_inscollect+IV3_insambi+PracFreqQ90+TimeSpentQ21, data = dv3dat)
summary(dv3step)
summ(dv3step, vifs = TRUE)
summary (dv3step3)
