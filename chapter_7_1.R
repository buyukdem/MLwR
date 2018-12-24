#reading dataset
concrete <- read.csv("concrete.csv")
str(concrete)

#feature distributions
lapply(concrete, hist) #non-normal distribution

#normalizing data between 0 and 1 to fit a neural network
normalize <- function(x) {
  
  return((x-min(x)) / (max(x)-min(x)))
  
}

concrete_norm <- as.data.frame(lapply(concrete, normalize))
summary(concrete_norm$strength)

#partitioning concrete dataset
concrete_train <- concrete_norm[1:773, ]
concrete_test <- concrete_norm[774:1030, ]

#fitting a neural network to training data
#other packages: nnet, RSNNS
install.packages("neuralnet")
library(neuralnet)

concrete_model <- neuralnet(strength ~ cement + slag +
                            ash + water + superplastic +
                            coarseagg + fineagg + age, 
                            data = concrete_train) #with one hidden node

plot(concrete_model) 
#8 input nodes: one node per input
#1 hidden node
#1 output node
#bias terms for hidden node and output node

#evaluating model performance
model_results <- compute(concrete_model, concrete_test[1:8]) #only the predioctors in the test set
predicted_strength <- model_results$net.result
cor(predicted_strength, concrete_test$strength)
#very similar peformance to linear regression model, 
#weights of input nodes are similar to weights if we built regression model 
#and bias terms are similar to intercept of a regression model

#improving model performance
concrete_model2 <- neuralnet(strength ~ cement + slag +
                              ash + water + superplastic +
                              coarseagg + fineagg + age, 
                              data = concrete_train, hidden = 5) #with 5 hidden nodes

plot(concrete_model2)
#more connections, less Sum of Squared Error: 5.66 -> 1.57

model_results2 <- compute(concrete_model2, concrete_test[1:8])
predicted_strength2 <- model_results2$net.result
cor(predicted_strength2, concrete_test$strength)
