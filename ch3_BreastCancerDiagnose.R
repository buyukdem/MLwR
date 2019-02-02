# This work shows how to use kNN algorithm to diagnose breast cancer benefiting from
# the data consists of measurements of biopsy mass taken from patients breasts. 

#read the data
wbcd <- read.csv(paste0(getwd(),"/datasets/wisc_bc_data.csv"), stringsAsFactors = FALSE)
str(wbcd)

wbcd <- wbcd[-1]
table(wbcd$diagnosis)
wbcd$diagnosis <- factor(wbcd$diagnosis, levels = c("B","M"), labels = c("Benign","Malignant"))
round(prop.table(table(wbcd$diagnosis))*100, digits = 1)

summary(wbcd[c("radius_mean","smoothness_mean","area_mean")])

# dataset should be scaled before applying kNN algorithm
normalize <- function(x) {
  return ((x - min(x)) / (max(x) - min(x)))
}

wbcd_n <- as.data.frame(lapply(wbcd[2:31], normalize))
summary(wbcd_n$radius_mean)

wbcd_train <- wbcd_n[1:469, ]
wbcd_test <- wbcd_n[470:569, ]

wbcd_train_labels <- wbcd[1:469, 1]
wbcd_test_labels <- wbcd[470:569, 1]

library(class)

# kNN is a lazy learning algorithm. It does not actually learn from data but specifies the class
# of an observation based on neighbours' distance. Higher number of votes given by the closest k-neighbors 
# specifies the class of an observation in the test dataset. 
# Rule of thumb k = square root of n, n = no of records in training data, k = 21 (21^2 = 469)
# Choosing the right k is very important! 
# Bias-variance tradeoff: large k means less variance more bias or vice versa
#training and prediction
wbcd_test_pred <- knn(train = wbcd_train, test = wbcd_test, cl = wbcd_train_labels, k = 21)

#model performance
library(gmodels)
CrossTable(x = wbcd_test_labels, y = wbcd_test_pred, prop.chisq = F)
# 2 examples are diagnosed benign falsely (FN) which is very costly mistake.

#new trial with z-score transformation
# z-score standardized values have no predefined minimum or maximum, extreme values are not
# compressed towards center (outliers have more weight)
wbcd_z <- as.data.frame(scale(wbcd[-1]))
summary(wbcd_z$area_mean)

wbcd_train <- wbcd_z[1:469,]
wbcd_test <- wbcd_z[470:569,]
wbcd_train_labels <- wbcd[1:469, 1]
wbcd_test_labels <- wbcd[470:569, 1]

#training and prediction
wbcd_test_pred <- knn(train = wbcd_train, test = wbcd_test, cl = wbcd_train_labels, k = 21)

#model performance
CrossTable(x = wbcd_test_labels, y = wbcd_test_pred, prop.chisq = F)
# no change in the performance

# Testing alternative values of k might be useful to find the best performance.
fp_fn <- data.frame(k=integer(), FP=integer(), FN=integer())
k = 1:100

for (i in 1:length(k)) {
  #build a model for the i-th k-value
  knn.pred <- knn(train = wbcd_train, test = wbcd_test, cl = wbcd_train_labels, k = i)
  
  #calculate the accuracy
  fp_fn[i, 1] <- i
  fp_fn[i, 2] <- table(knn.pred, wbcd_test_labels)[1,2]
  fp_fn[i, 3] <- table(knn.pred, wbcd_test_labels)[2,1]
}

fp_fn[with(fp_fn, order(FN, FP)), ]

library(ggplot2)
library(reshape2)
melt_fp_fn <- melt(fp_fn, id = "k")
ggplot(melt_fp_fn, aes(x = k, y = value, color = variable)) +
  geom_line() +
  ylab(label="number of mistakes") + 
  xlab("k value") + 
  scale_colour_manual(values=c("blue", "red")) + 
  scale_x_continuous(breaks=k*10) 

# k=14 gives 2 FP but 0 FN which results in the least cost for this scenario. (we may choose k=15 to prevent ties while voting) 
# In other scenarios, False Positives may be as costly as False Negatives if not more!
