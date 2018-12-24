##read dataset
insurance <- read.csv("insurance.csv", stringsAsFactors = TRUE)

##info about dataset
str(insurance)
summary(insurance$charges)
hist(insurance$charges) ##linear regression assumes the target var. is distributed normally.
                        ##here this assumption is violated but in practice it's violated most of the time

table(insurance$region) 

##correlation matrix and visualizations
cor(insurance[c("age","bmi","children","charges")])
pairs(insurance[c("age","bmi","children","charges")])
#install.packages("psych")
library(psych)
pairs.panels(insurance[c("age","bmi","children","charges")]) 
#correlation ellipse shows the correlation strength, the more it's streched the stronger
#red dot shows the point where mean of x and y
#the curve is loess smooth

##construct the model
ins_model <- lm(formula = charges ~ ., data = insurance)
ins_model
#reference group: female non-smoker in the northeast region
#males have 131.3 dollar less medical costs than females

##model performance
summary(ins_model)
#residual = actual-predicted
#50% of errors fall between -2.848$ and 1.393$ (1Q and 3Q), overestimated and underestimated respectively

##improving model performance
#relation between dependent and independent var is assumed to be linear in linear regression.
#but in some cases the relation is non-linear such as the charges~age relation.
insurance$age2 <- insurance$age^2

#transforming numeric var. to categorical:
#bmi >= 30 means obese people, which might be more explanatory for medical cost than numeric bmi values
insurance$bmi30 <- ifelse(insurance$bmi>=30, 1, 0)

#adding an interaction effects:
#obese and smoker people might have more medical costs 
#so it's reasonable to look at the combined effect of two: bmi30*smoker

#finally:
ins_model2 <- lm(formula = charges ~ age + age2 + children + bmi
                 + sex + bmi30*smoker + region, data = insurance)
summary(ins_model2)
#age2 (nonlinear) variable is statistically significant
#obese and smoker interaction variable is very strong