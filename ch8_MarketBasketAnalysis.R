#reading the dataset as sparse matrix
install.packages("arules")
library(arules)

groceries <- read.transactions("groceries.csv", sep = ",")  #I don't want to change the order of txns
                                                            #so I can't create a regular matrix to fill the data
                                                            #so I created a sparse matrix which stores only the items that appear in the txn
                                                            #more efficient memory-wise

summary(groceries)
inspect(groceries[1:5, ]) #to see the items in the first 5 txns
itemFrequency(groceries[ ,1:3]) #proportion of txns that contain the items

#visualtzation - items
itemFrequencyPlot(groceries, support = 0.1) #plotting items by their proportions in overall txns
                                            #support parameter sets the minimum support to be shown in the plot
                                            #don't show the items with support under 0.1
                                            #support(X): count(X)/#TXN 
                                            #confidence(X->Y): support(X,Y)/support(X) where X,Y are items

itemFrequencyPlot(groceries, topN = 20) #show 20 items that have the highest support

#visualtzation - txns
image(groceries[1:5, ])
image(sample(groceries, 100))

#training a model - unsupervised learning algorithm: apriori which is an association rules algorithm
groceryrules <- apriori(groceries, parameter = list(support = 0.006, #if an item purchased twice a day (30days*2=60) 
                                                                     #then support = 60/9386 = 0.006
                                                    confidence = 0.25, #rule has to be correct at least 25% of the time
                                                    
                                                    minlen = 2)) #avoidimg unnecessary rules like {} -> "whole milk
                                                                 #eliminate rules that contain fewer than 2 items

#evaluating the model performance
summary(groceryrules) #lift(x->y) = confidence(x->y)/support(y), how much more likely one item is to be purchased
                      #relative to its typical purchase rate

inspect(groceryrules[1:3]) #prints rules and their stats, support, confidence and lift

#improving model performance-sorting the rules based on their value
inspect(sort(groceryrules, by = "lift")[1:5])

berryrules1 <- subset(groceryrules, items %in% "berries") #to select the rules that contain berries
inspect(berryrules1)

berryrules2 <- subset(groceryrules, items %in% c("berries","yogurt")) #to select the rules that contain berries or yogurt
inspect(berryrules2)

berryrules3 <- subset(groceryrules, items %ain% c("berries","yogurt")) #to select the rules that contain berries and yogurt (complete match)
inspect(berryrules3)

berryrules4 <- subset(groceryrules, items %pin% "fruit") #to select the rules where "fruit" word passes (partial match)
inspect(berryrules4)

#writing the rules to a CSV
write(groceryrules, file = "groceryrules.csv", 
      sep = ",", quote = TRUE, row.names = FALSE)

#converting rules to a dataframe
groceryrules_df <- as(groceryrules, "data.frame")
str(groceryrules_df)

