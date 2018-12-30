##kNN practice
#read the data
wbcd <- read.csv(paste0(getwd(),"/datasets/wisc_bc_data.csv"), stringsAsFactors = FALSE)
str(wbcd)

wbcd <- wbcd[-1]
table(wbcd$diagnosis)
wbcd$diagnosis <- factor(wbcd$diagnosis, levels = c("B","M"), labels = c("Benign","Malignant"))
round(prop.table(table(wbcd$diagnosis))*100, digits = 1)

summary(wbcd[c("radius_mean","smoothness_mean","area_mean")])

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

#training and prediction
wbcd_test_pred <- knn(train = wbcd_train, test = wbcd_test, cl = wbcd_train_labels, k = 21)

#model performance
library(gmodels)
CrossTable(x = wbcd_test_labels, y = wbcd_test_pred, prop.chisq = F)

#new trial: z-transformation
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
