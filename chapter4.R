#read the data
sms_raw <- read.csv(paste0(getwd(),"/datasets/sms_spam.csv"), stringsAsFactors = F)

str(sms_raw)
sms_raw$type <- factor(sms_raw$type)
table(sms_raw$type)

#text mining package
#install.packages("tm")
library(tm)

sms_corpus <- Corpus(VectorSource(sms_raw$text))
print(sms_corpus)
inspect(sms_corpus[1:3])

#processing the text data
corpus_clean <- tm_map(sms_corpus, tolower)
corpus_clean <- tm_map(corpus_clean, removeNumbers)
corpus_clean <- tm_map(corpus_clean, removeWords, stopwords())
corpus_clean <- tm_map(corpus_clean, removePunctuation)
corpus_clean <- tm_map(corpus_clean, stripWhitespace)
inspect(corpus_clean[1:3])

# word tokenization
sms_dtm <- DocumentTermMatrix(corpus_clean)

#training and test sets
sms_raw_train <- sms_raw[1:4169, ]
sms_raw_test <- sms_raw[4170:5574, ]

sms_dtm_train <- sms_dtm[1:4169, ]
sms_dtm_test <- sms_dtm[4170:5574, ]

sms_corpus_train <- corpus_clean[1:4169]
sms_corpus_test <- corpus_clean[4170:5574]

prop.table(table(sms_raw_train$type))
prop.table(table(sms_raw_test$type))

#word cloud
#install.packages("wordcloud")
library(wordcloud)
wordcloud(sms_corpus_train, min.freq = 40, random.order = F)

spam <- subset(sms_raw_train, type == "spam")
ham <- subset(sms_raw_train, type == "ham")

wordcloud(spam$text, max.words = 40, scale = c(3, .5))
wordcloud(ham$text, max.words = 40, scale = c(3, .5))

findFreqTerms(sms_dtm_train, 5)
sms_dict <- findFreqTerms(sms_dtm_train, 5)

#train and test sets include words appearing at least 5 times in all the messages
sms_train <- DocumentTermMatrix(sms_corpus_train, list(dictionary = sms_dict))
sms_test <- DocumentTermMatrix(sms_corpus_test, list(dictionary = sms_dict))
dim(sms_train)
dim(sms_test)

#converting numeric features to factor
convert_counts <- function(x) {
  x <- ifelse(x > 0, 1, 0)
  x <- factor(x, levels = c(0, 1), labels = c("No","Yes"))
  return(x)
}

sms_train <- apply(sms_train, 2, convert_counts)
sms_test <- apply(sms_test, 2, convert_counts)

#training a model
#install.packages("e1071")
library(e1071)
sms_classifier <- naiveBayes(sms_train, sms_raw_train$type)

#evaluating model performance
sms_test_pred <- predict(sms_classifier, sms_test)
library(gmodels)
CrossTable(sms_test_pred, sms_raw_test$type, 
           prop.chisq = F, prop.t = F, 
           dnn = c("predicted","actual"))

#improving the model
sms_classifier2 <- naiveBayes(sms_train, sms_raw_train$type, laplace = 1)
sms_test_pred <- predict(sms_classifier2, sms_test)
CrossTable(sms_test_pred, sms_raw_test$type, 
           prop.chisq = F, prop.t = F, 
           dnn = c("predicted","actual"))
