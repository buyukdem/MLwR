#read the dataset
teens <- read.csv("snsdata.csv")
str(teens)

##data preparation
table(teens$gender, useNA = "ifany")
summary(teens$age)  #min and max values don't seem to fit the date
                    #they need to be cleaned. high school studets can be between 13-19 yo
teens$age <- ifelse(teens$age >= 13 & teens$age < 20, teens$age, NA)
summary(teens$age)

#one hot encoding: creating dummy variables
teens$female <- ifelse(teens$gender == "F" & !is.na(teens$gender), 1, 0) #NA and M is 0
teens$no_gender <- ifelse(is.na(teens$gender), 1, 0) #F and M is 0
#rest is male

#control
table(teens$gender, useNA = "ifany")
table(teens$female, useNA = "ifany")
table(teens$no_gender, useNA = "ifany")

#imputing missing data - numeric variable
mean(teens$age, na.rm = TRUE)
aggregate(data = teens, age ~ gradyear, mean, na.rm = TRUE)
ave_age <- ave(teens$age, teens$gradyear, FUN = 
                 function(x) mean(x, na.rm = TRUE)) #returns gradyear average age for each student 

teens$age <- ifelse(is.na(teens$age), ave_age, teens$age)
summary(teens$age)

##clustering
interests <- teens[5:40] #subsetting features including words used
#z-score transformation
interests_z <- as.data.frame(lapply(interests, scale))
teen_clusters <- kmeans(interests_z, 5) #princesses, brains, criminals, athletes, basket cases

##evaluation
teen_clusters$size
teen_clusters$centers #cluster 3 members are below the mean for all features
                      #they might be the members who post very little or nothing -- basket cases
                      #cluster 1 can be athletes
                      #cluster 5 can be princesses
                      #cluster 4 can be brains

##further analysis of clusters
teens$cluster <- teen_clusters$clusters
teens[1:5, c("cluster","gender","age","friends")]

aggregate(data = teens, age ~ cluster, mean) #age does not differ much by cluster
aggregate(data = teens, female ~ cluster, mean) #notable differences in gender and gender is not used in clustering
aggregate(data = teens, friends ~ cluster, mean) #princess(cl 5) have the most friends then athletes then brains
