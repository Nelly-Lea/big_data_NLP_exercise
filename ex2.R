#Chirly Sfez 342687647
#Nelly Lea Amar 341289106

library(stringr)
library(dplyr)
library(ggplot2)
library(mosaic)
library(dplyr)
library(stringr)
library(xtable)
library(gridExtra)
library(stopwords)
library(quanteda)
library(caret)
library(rpart.plot)
library(rpart)
library(limma)
library(tidyverse)
library(cluster)
library(factoextra)
# Set rounding to 2 digits
options(digits=2)

## ----cache=TRUE, warning=FALSE, message=FALSE----------------------------
profiles <- read.csv( file.path( 'C:\\Users\\nelly\\Documents\\big data\\exercise2', 'okcupid_profiles.csv' ), header=TRUE, stringsAsFactors=FALSE)
n <- nrow(profiles)

str(profiles)

#profiles <- filter(profiles.full, height>=60)
#profiles.test<-filter(profiles.full, height<60)
#object.size(profiles.full)
#object.size(profiles)
#object.size(profiles.test)
#rm(profiles.full)

essays <- select(profiles, starts_with("essay"))
essays <- apply(essays, MARGIN = 1, FUN = paste, collapse=" ")

html <- c( "<a[^>]+>", "class=[\"'][^\"']+[\"']", "&[a-z]+;", "\n", "\\n", "<br ?/>", "</[a-z]+ ?>" )
stop.words <-  c( "a", "am", "an", "and", "as", "at", "are", "be", "but", "can", "do", "for", "have", "i'm", "if", "in", "is", "it", "like", "love", "my", "of", "on", "or", "so", "that", "the", "to", "with", "you", "i" )

html.pat <- paste0( "(", paste(html, collapse = "|"), ")" )
html.pat
stop.words.pat <- paste0( "\\b(", paste(stop.words, collapse = "|"), ")\\b" )
stop.words.pat
essays <- str_replace_all(essays, html.pat, " ")
essays <- str_replace_all(essays, stop.words.pat, " ")


# Tokenize essay texts
all.tokens <- tokens(essays, what = "word",
                     remove_numbers = TRUE, remove_punct = TRUE,
                     remove_symbols = TRUE, remove_hyphens = TRUE)

# Take a look at a specific message and see how it transforms.
all.tokens[[357]]

# Lower case the tokens.
all.tokens <- tokens_tolower(all.tokens)

# Use quanteda's built-in stopword list for English.
# NOTE - You should always inspect stopword lists for applicability to
#        your problem/domain.
all.tokens <- tokens_select(all.tokens, stopwords(),
                            selection = "remove")
all.tokens[[357]]


# Perform stemming on the tokens.
all.tokens <- tokens_wordstem(all.tokens, language = "english")
# remove single-word tokens after stemming. Meaningless
all.tokens <- tokens_select(all.tokens, "^[a-z]$",
                            selection = "remove", valuetype = "regex")
all.tokens[[357]]

# Create a bag-of-words model (document-term frequency matrix)
all.tokens.dfm <- dfm(all.tokens, tolower = FALSE)
rm(all.tokens)

all.tokens.dfm

sparsity(all.tokens.dfm)
# meaning that 99.90% of the cells are zeros. Even if you could
# create the data frame, fitting a model is not going to work
# because of this extreme lack of information in the features.
# Solution? Trim some features.

dfm.trimmed <- dfm_trim(all.tokens.dfm, min_docfreq = 25, min_termfreq = 35, verbose = TRUE)
dfm.trimmed



# Transform to a matrix and inspect.
all.tokens.matrix <- as.matrix(dfm.trimmed)
object.size(all.tokens.matrix)
#View(all.tokens.matrix[1:20, 1:100])
dim(all.tokens.matrix)


# Investigate the effects of stemming.
# [A]
colnames(all.tokens.matrix)[1:50]

# [B]
sort(colnames(all.tokens.matrix))[1:100]

# clear some space
gc()


memory.limit(size=20000) 

#Implementation of tf-idf 
# Our function for calculating relative term frequency (TF)
term.frequency <- function(row) {
  row / sum(row)
}

# Our function for calculating inverse document frequency (IDF)
inverse.doc.freq <- function(col) {
  corpus.size <- length(col)
  doc.count <- length(which(col > 0))
  
  log10(corpus.size / doc.count)
}

# Our function for calculating TF-IDF.
tf.idf <- function(x, idf) {
  x * idf
}
gc()
#rm(all.tokens.dfm)
# First step, normalize all documents via TF.
all.tokens.df <-apply(all.tokens.matrix, 1, term.frequency)
dim(all.tokens.df)
View(all.tokens.df[1:20, 1:100])


# Second step, calculate the IDF vector that we will use - both
# for training data and for test data!
all.tokens.idf <-apply(all.tokens.matrix, 2, inverse.doc.freq)
str(all.tokens.idf)


# Lastly, calculate TF-IDF for our training corpus.
all.tokens.tfidf <-apply(all.tokens.df, 2, tf.idf, idf = all.tokens.idf)
dim(all.tokens.tfidf)
View(all.tokens.tfidf[1:25, 1:25])


# Transpose the matrix
all.tokens.tfidf <- t(all.tokens.tfidf)
dim(all.tokens.tfidf)
View(all.tokens.tfidf[1:25, 1:25])

# Setup a the feature data frame with labels.
all.tokens.df <- cbind(Label = profiles$sex, data.frame(dfm.trimmed))
rm(profiles)
#rajout 
rm(dfm.trimmed)

# Often, tokenization requires some additional pre-processing
names(all.tokens.df)[c(146, 148, 235, 238)]


# Cleanup column names.
names(all.tokens.df) <-make.names(names(all.tokens.df))

#tokens.subset <- filter(all.tokens.df, height>=55 & height <=80)
#object.size(tokens.subset)
# Use caret to create stratified folds for 10-fold cross validation repeated
# 3 times (i.e., create 30 random stratified samples)
set.seed(48743)
cv.folds <-createMultiFolds(labels, k = 10, times = 3)

cv.cntrl <-trainControl(
  method = "repeatedcv",
  number = 10,
  repeats = 3,
  index = cv.folds
)

rm(cv.folds)
rm(all.tokens.dfm)

#start.time <- Sys.time()
#rpart.cv.1 <- train(x = all.tokens.matrix,y=labels, method = "rpart", trControl = cv.cntrl, tuneLength = 7)

#total.time <- Sys.time() - start.time
#total.time

#summary(rpart.cv.1)


#pred = predict(rpart.cv.1, type="raw")
#pred
#labels
#table(pred, labels)


start.time <- Sys.time()
tree <- rpart(Label~., data=all.tokens.df, cp=.02)
total.time <- Sys.time() - start.time
total.time
tree
pred = predict(tree, type="class")

table(pred,all.tokens.df$Label)
rpart.plot(tree, box.palette="RdBu", shadow.col="gray", nn=TRUE)

profiles <- read.csv( file.path( 'C:\\Users\\nelly\\Documents\\big data\\exercise2', 'okcupid_profiles.csv' ), header=TRUE, stringsAsFactors=FALSE)
n <- nrow(profiles)

str(profiles)
male.words <- subset(essays, profiles$sex == "m") %>%
  str_split(" ") %>%
  unlist() %>%
  table() %>%
  sort(decreasing=TRUE) %>%
  names()
female.words <- subset(essays, profiles$sex == "f") %>%
  str_split(" ") %>%
  unlist() %>%
  table() %>%
  sort(decreasing=TRUE) %>%
  names()

# Top 25 male words:
print( male.words[1:25] )
# Top 25 female words
print( female.words[1:25] )

## ----cache=TRUE, warning=FALSE, message=FALSE----------------------------
# Words in the males top 500 that weren't in the females' top 500:
male_words<-as.data.frame(setdiff(male.words[1:500], female.words[1:500]))
# Words in the female top 500 that weren't in the males' top 500:
female_words<-as.data.frame(setdiff(female.words[1:500], male.words[1:500]))
all.tokens.df.sub<-all.tokens.df[,-c(1,2)]

#make a list of male word (these words are in male_words)
drop_male_word<-c("video","company","sports","internet","computer","star","science","business"
                  ,"us","couple","bar","here","started","lost","three","run","become"
                  ,"beer","now","less","isn't","south","words","point","stuff","woman","show"
                  ,"thought","games","car","during","self","done","seem","said","daily"
                  ,"history","years" ,"3","breaking")

#make a list of female words (these words in female_word)
all.tokens.df<-all.tokens.df[,!names(all.tokens.df)%in% drop_male_word]
drop_female_word<-c("loving","dancing,","dog","hair","laughing",";)","laugh","please"
                    ,"kids","adventure","family","healthy","explore","hiking","laugh"
                    ,"men","smile","nature","comfortable","crazy","chocolate","harry"
                    ,"dating","mad","loved","positive","laughter","modern","sunshine"
                    ,"active","yoga","fresh","ready","art","glass","except","loves"
                    ,"planning","half","strong")
all.tokens.df<-all.tokens.df[,!names(all.tokens.df)%in% drop_female_word]

#cluster
pca_word = prcomp(all.tokens.df, center = TRUE, scale = TRUE)
summary(pca_word)

for( i in c(2,3,4,10))
{
  kmeans_word = kmeans(all.tokens.df, centers = k, nstart = 50)
  fviz_cluster(kmeans_word, data = all.tokens.df)
}
