# Reading txt file
library(readtext)
txt <- readtext("text.txt")
text <- Corpus(VectorSource(txt))

# Text PreProcessing
library(tm)
tokenizewhitespace <- tm_map(text,stripWhitespace)
tokenizelowercase <- tm_map(tokenizewhitespace,content_transformer(tolower))
extratokenize <- tm_map(tokenizelowercase,removeNumbers)
stopword <- tm_map(extratokenize,removeWords, stopwords('en'))
library(SnowballC)
Canonicalform <- tm_map(extratokenize, stemDocument)
preProcessedtext <- Canonicalform

# n grams
library(tokenizers)
grams <- tokenize_ngrams(preProcessedtext, lowercase = TRUE, n = 3L, n_min = 2,
                         stopwords = character(), ngram_delim = " ", simplify = FALSE)
grams 


# Matrix Form
dtm <- DocumentTermMatrix(preProcessedtext)
tdm <- TermDocumentMatrix(preProcessedtext)

tfidf <- as.matrix(dtm)
head(tfidf)
dim(tfidf)
write.csv(tfidf, file = 'texting.csv', row.names = 'frequency')

#Wordcloud
library(wordcloud)
dtm.matrix = as.matrix(dtm)
wordcloud(colnames(dtm.matrix), dtm.matrix[1,], max.words = 124, rot.per = 0.4,colors = c('violet','indigo','blue','green','yellow','orange','red'))

# H Clustering
library(fpc)
library(cluster)
dis <- dist(t(dtm), method = 'euclidian')
groups <- hclust(dis,method="ward.D")
plot(groups, cex=0.9, hang=-1)
rect.hclust(groups, k=4)

library("lsa")
getwd()
dir()
library(tm) 
docs <- Corpus(DirSource("C:/Users/hp/text"), readerControl = list(reader=readPlain)) 

# Constructs or coerces to a document-term matrix
dt_matrix <- DocumentTermMatrix(docs) 
dt_matrix <- DocumentTermMatrix(docs,         control = list(weighting = weightTfIdf, 
                                               wordLengths = c(6, Inf),
                                               removeNumbers = TRUE,
                                               removePunctuation = TRUE,
                                               stopwords = TRUE
                                ))
inspect(dt_matrix)                          
dim(dt_matrix)
lsaSpace = lsa(t(dt_matrix), dims=3)

# document-term matrix M = T S t(D)decomposed via a singular value decomposition
# into: term vector matrix T (constituting left singular vectors), the 
# document vector matrix D (constituting right singular vectors) being both
# orthonormal, and the diagonal matrix S (constituting singular values)
sk=diag(lsaSpace$sk)
lsaSpace$sk
lsaSpace$tk
lsaSpace$dk

#Creating query string
q=query("analysts", rownames(t(dt_matrix)))
q=query("engines", rownames(t(dt_matrix)))


# q = t(q)U(S)-1
qv=t(q)%*%lsaSpace$tk%*%solve(sk)

# Cosine Similarity 

score= 0
doc = 0
for (i in 1:nrow(lsaSpace$dk)) {
  score[i] =cosine(as.vector(qv),as.vector(lsaSpace$dk[i,]))
  doc[i] = i
}
Result=cbind(doc,score)
Result

