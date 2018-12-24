#read the data
sms_raw <- read.csv("sms_spam.csv", stringsAsFactors = F)

str(sms_raw)
sms_raw$type <- factor(sms_raw$type)
table(sms_raw$type)

install.packages("tm")
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