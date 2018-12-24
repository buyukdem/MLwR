#read the data
credit <- read.csv(paste0(getwd(),"/datasets/credit.csv"))

str(credit)
table(credit$checking_balance)
table(credit$savings_balance)
summary(credit[,c("months_loan_duration","amount")])
table(credit$default)
credit$default <- factor(credit$default, levels = c("1","2"), labels = c("No","Yes"))

#training and test sets
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
CrossTable(credit_test$default, credit_pred,
           prop.c = F, prop.chisq = F, prop.t = F,
           dnn = c("actual_default","predicted_default"))

#improving the model performance, boosting
credit_boost10 <- C5.0(credit_train[-17], credit_train$default, trials = 10)
credit_boost10
summary(credit_boost10)

credit_boots10_pred <- predict(credit_boost10, credit_test)
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
