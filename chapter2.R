#install.packages("gmodels")
library(gmodels)

data("mtcars")
str(mtcars)
mtcars$carb <- as.factor(mtcars$carb)
mtcars$gear <- as.factor(mtcars$gear)
mtcars$am <- as.factor(mtcars$am)
str(mtcars)

CrossTable(x = mtcars$gear, y = mtcars$carb)
CrossTable(x = mtcars$gear, y = mtcars$carb, chisq = TRUE)
