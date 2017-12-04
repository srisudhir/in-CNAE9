rm(list=ls(all=TRUE))
library("lsa")
setwd("C:/Users/Classroom2/Desktop/Material")
dir()
library(tm) 
docs <- Corpus(DirSource("C:/Users/Classroom2/Desktop/Material/LSIDocs"), readerControl = list(reader=readPlain)) 

# Constructs or coerces to a document-term matrix
dt_matrix <- DocumentTermMatrix(docs) 
dt_matrix <- DocumentTermMatrix(docs,
                                control = list(weighting = weightTfIdf, 
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


#The following code accomplishes q = t(q)U(S)-1
qv=t(q)%*%lsaSpace$tk%*%solve(sk)

#In this step, we are finding cosine similarity with all document matrices

score= 0
doc = 0
for (i in 1:nrow(lsaSpace$dk)) {
  score[i] =cosine(as.vector(qv),as.vector(lsaSpace$dk[i,]))
  doc[i] = i
}
Result=cbind(doc,score)
Result

#For n-grams, use the following link to explore further.
#http://stackoverflow.com/questions/28033034/r-and-tm-package-create-a-term-document-matrix-with-a-dictionary-of-one-or-two


