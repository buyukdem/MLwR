##read the dataset
wine <- read.csv("whitewines.csv")

##dataset info
str(wine)
hist(wine$quality)
summary(wine)
head(wine)

##splitting the data: 75% to 25%
wine_train <- wine[1:3750, ]
wine_test <- wine[3751:4898, ]

##training a model
library(rpart)
m.rpart <- rpart(quality ~ ., data = wine_train)
m.rpart
summary(m.rpart)

##tree visualization
library(rpart.plot)
rpart.plot(m.rpart, digits = 3)
rpart.plot(m.rpart, digits = 4, fallen.leaves = TRUE, type = 3, extra = 101)


##model performance
p.rpart <- predict(m.rpart, wine_test)
summary(p.rpart);summary(wine_test$quality)
#predicted values are narrower, not performing well for worst and best quality wines
cor(p.rpart, wine_test$quality)
#simply looking at correlation between actual and predicted values to see the performance
MAE <- function(actual, predicted) {
  mean(abs(actual-predicted))
}
MAE(wine_test$quality, p.rpart)
#on average the error is 0.57 which is fairly well considering the scale is 0 to 10
MAE(wine_test$quality, mean(wine_train$quality))
#however model estimate is slightly better than when the estimates are mean of actual values

##improving the model performance: 
#building a model tree, a tree is built and estimations are made by linear models in each terminal node
library(RWeka)
m.m5p <- M5P(quality ~ ., data = wine_train)
m.m5p
summary(m.m5p) ## performance diagnostics on training data
p.m5p <- predict(m.m5p, wine_test)
summary(p.m5p)
cor(wine_test$quality, p.m5p) #correlation is better
MAE(wine_test$quality, p.m5p) #average absolute error is lower
