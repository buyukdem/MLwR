# This work utilizes Decision Tree Algorithm whether to reject or accept a customer's loan application
# using their financial information and characteristics. Decision tree is very useful for this problem
# because it provides a parsimonious model with a high accuracy. 

#read the data
setwd("C:/Users/U085452/Desktop/ML with R/MLwR-master")
credit <- read.csv(paste0(getwd(),"/datasets/credit.csv"))
 
str(credit)
table(credit$checking_balance)
table(credit$savings_balance)
summary(credit[,c("months_loan_duration","amount")])
table(credit$default)
credit$default <- factor(credit$default, levels = c("1","2"), labels = c("No","Yes"))

#training and test sets
# the data should be ordered randomly before splitting. To make it random, we do the following:
set.seed(12345)
credit_rand <- credit[order(runif(1000)), ]
summary(credit$amount)
summary(credit_rand$amount)
head(credit$amount)
head(credit_rand$amount)

credit_train <- credit_rand[1:900, ]
credit_test <- credit_rand[901:1000, ]

#checking target variable has similar proportions in the training and test
prop.table(table(credit_train$default)); prop.table(table(credit_test$default))

#training a model
#install.packages("C50")
library(C50)
credit_model <- C5.0(credit_train[-17], credit_train$default)
credit_model
summary(credit_model)

#evaluating model performance
credit_pred <- predict(credit_model, credit_test)
library(gmodels)
CrossTable(credit_test$default, credit_pred,
           prop.c = F, prop.chisq = F, prop.t = F,
           dnn = c("actual_default","predicted_default"))

#improving the model performance, adaptive boosting
# many decision trees are built, and trees vote on the best class for each example. We could build a model
# with better predictive power with boosting since it ensembles a bunch of weak learners to have a better one.
credit_boost10 <- C5.0(credit_train[-17], credit_train$default, trials = 10)
credit_boost10
summary(credit_boost10)

credit_boots10_pred <- predict(credit_boost10, credit_test)
library(gmodels)
CrossTable(credit_test$default, credit_boots10_pred,
           prop.c = F, prop.chisq = F, prop.t = F,
           dnn = c("actual_default","predicted_default"))

#assigning a penalty with cost matrix: 
#false positive (predicted:'No Loan Default', actual:'Yes Loan Default') is more costly than false negative
error_cost <- matrix(c(0,1,4,0), nrow = 2)
error_cost
credit_cost <- C5.0(credit_train[-17], credit_train$default, costs = error_cost)
credit_cost_pred <- predict(credit_cost, credit_test)
CrossTable(credit_test$default, credit_cost_pred,
           prop.c = F, prop.chisq = F, prop.t = F,
           dnn = c("actual_default","predicted_default"))

#partitioning data with holdout method
random_ids <- order(runif(1000))
credit_train2 <- credit[random_ids[1:500], ]
credit_validation2 <- credit[random_ids[501:750], ]
credit_test2 <- credit[random_ids[751:1000], ]
prop.table(table(credit_train2$default)); prop.table(table(credit_validation2$default)); prop.table(table(credit_test2$default));

#in some cases, a class may have very different proportions in each partition
#stratified random sampling is used to address this problem
library(caret)
in_train <- createDataPartition(credit$default, p = 0.75, list = FALSE)
credit_train3 <- credit[in_train, ]
credit_test3 <- credit[-in_train, ]

#k-fold cross-validation: a repeated holdout method
#empirical evidence suggests that k > 10 has little added benefit to the model performance
#mostly 10-fold cv is used
set.seed(123)
folds <- createFolds(credit$default, k = 10) 
str(folds)
#there is no function to perform cv so a loop is needed to automate the model construction and performance evaluation
library(irr)
cv_results <- lapply(folds, function(x) {
  credit_train_cv <- credit[x, ]
  credit_test_cv <- credit[-x, ]
  credit_model_cv <- C5.0(default ~ ., data = credit_train_cv)
  credit_pred_cv <- predict(credit_model_cv, credit_test_cv)
  credit_actual <- credit_test_cv$default
  kappa <- kappa2(data.frame(credit_actual, credit_pred_cv))$value
  return(kappa)
})
str(cv_results)
mean(unlist(cv_results)) #cv suggests it is actually a poor model

#improving model performance with caret
library(caret)
set.seed(300)
m <- train(default ~ ., data = credit, method = "C5.0")
m

#When there are numerous alternatives for each test in the tree or ruleset,
#it is likely that at least one of them will appear to provide valuable predictive information. 
#In applications like these it can be useful to pre-select a subset of the attributes 
#that will be used to construct the decision tree or ruleset. 
#The C5.0 mechanism to do this is called "winnowing" by analogy with the process for separating wheat from chaff 
#(or, here, useful attributes from unhelpful ones).
#winnow = TRUE -> decision tree is built upon with variables that have the highest predictive power
#variables with low predictive power are not included in the model
p <- predict(m, credit)
table(p, credit$default)
#only 2 mistakes but this is not a correct performance metric just a resubstitution error
#0.73 accuracy estimate of bootstrap is more realistic
head(predict(m, credit))
head(predict(m, credit, type = "prob"))

#fine tuning
ctrl <- trainControl(method = "cv", number = 10,
                     selectionFunction = "oneSE") 
#resampling method and model selection criteria are specified. 
#oneSE: selects the simplest model within one standard error. 
#tolerance: selects the simplest model with user-defined performance
#best: selects the best performing model
grid <- expand.grid(.model = "tree",
                    .trials = c(1,5,10,15,20,25,30,35),
                    .winnow = FALSE)
#grid: different model paramaters to be tried while model building
set.seed(300)
m <- train(default ~ ., data = credit, method = "C5.0",
           metric = "Kappa",
           trControl = ctrl,
           tuneGrid = grid)
m

#ensemble-learning (meta-learning: learning how to learn)
#bagging: bootstrapped aggregating
library("ipred")
set.seed(300)
mybag <- bagging(default ~ ., data = credit, nbagg = 25)
credit_pred <- predict(mybag, credit)
table(credit_pred, credit$default)

#let's try with 10-CV
library(caret)
set.seed(300)
ctrl <- trainControl(method = "cv", number = 10)
train(default ~ ., data = credit, method = "treebag", #treebag: bagging algorithm in caret
      trControl = ctrl)

bagctrl <- bagControl(fit = svmBag$fit,
                      predict = svmBag$pred,
                      aggregate = svmBag$aggregate)
set.seed(300)
svmbag <- train(default ~ ., data = credit,
                trControl = ctrl, bagControl = bagctrl)
svmbag

#boosting
#differences from bagging: 
#1- resampled datasets are constructed specifically to generate complementary learners
#2- vote is weighted based on each model's performance
#AdaBoost (adaptive boosting) resampling and training starting 
#from easy datasets (easy to correctly predict) to difficult ones

#random forests: bagging with random feature selection then aggregating with votes
#can handle extremely large datasets (handles curse of dimensionality)
#selects only the most important features
library(randomForest)
set.seed(300)
rf <- randomForest(default ~ ., data = credit)
rf

#evaluating and comparing (with boosted C5.0) random forest model performance
ctrl <- trainControl(method = "repeatedcv", number = 10, repeats = 10)
grid.rf <- expand.grid(.mtry = c(2,4,8,16))
set.seed(300)
m_rf <- train(default ~ ., data = credit, method = "rf",
              metric = "Kappa", trControl = ctrl, tuneGrid = grid.rf)

grid_c50 <- expand.grid(.model = "tree", 
                        .trials = c(10,20,30,40),
                        .winnow = FALSE)
set.seed(300)
m_c50 <- train(default ~ ., data = credit, method = "C5.0",
               metric = "Kappa", trControl = ctrl, tuneGrid = grid_c50)
m_rf
m_c50 #with 0.36 kappa (50 trials boosted C5.0) is winner
