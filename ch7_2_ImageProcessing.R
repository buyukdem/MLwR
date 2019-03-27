#reading the dataset
letters <- read.csv("letterdata.csv")
str(letters)

#splitting the dataset
letters_training <- letters[1:16000, ]
letters_test <- letters[16001:20000, ]

#SVM learning - packages: e1071, klaR, kernlab 
install.packages("kernlab") #developed natively in R, can be used with caret package
library(kernlab)

letter_classifier <- ksvm(letter ~ ., data = letters_training,
                          kernel = "vanilladot") #linear kernel

letter_classifier #cost c = 1 by default: cost of violating the constraints, soft margins
                  #larger values result in narrower margins

#evaluating model performance
letter_predictions <- predict(letter_classifier, letters_test, type = "response")
head(letter_predictions)

table(letter_predictions, letters_test$letter) #to see how well it performed

agreement <- letter_predictions == letters_test$letter
table(agreement) #how many classified correctly
prop.table(table(agreement))

#improving the model performance
letter_classifier_rbf <- ksvm(letter ~ ., data = letters_training,
                              kernel = "rbfdot")  #Gaussian RBF Kernel, default kernel 
                                                  #RBF: Radial Basis Functiom

letter_predictions_rbf <- predict(letter_classifier_rbf, letters_test, type = "response")
agreement_rbf <- letter_predictions_rbf == letters_test$letter
table(agreement_rbf)
prop.table(table(agreement_rbf))  #with different kernel, performance improved 9 pts
                                  #for more improved model, cost parameter can be modified
