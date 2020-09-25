import pandas as pd
import numpy as np
import re # regular expression
import string 
import nltk
from nltk.stem.porter import PorterStemmer
from timeit import default_timer

nltk.download('punkt')

# runtime clock
start = default_timer()

def importData(csv):
    df = pd.read_csv(csv)
    return df.review, df.sentiment, df.rate

def removePunctuation(sentences):
    return sentences.apply(lambda s : re.sub('[^a-zA-Z]', ' ', s))

def wordStemmer(words):
    ps = PorterStemmer()
    words = words.apply(lambda x: x.split())
    words = words.apply(lambda x: ' '.join([ps.stem(word) for word in x]))
    return words

# tokenizer for pandas dataframe
def sentenceTokenizer(sentences):
    return sentences.apply(nltk.sent_tokenize)

def wordTokenizer(words):
    return words.apply(nltk.word_tokenize)

X, y, z = importData('labeledTrain.csv')

#pre-process
cleanup = removePunctuation(X)
print(X[0])
stem = wordStemmer(cleanup)
print('\nRuntime: {}'.format((default_timer()-start)))

#itung bobot score tf-idf

from sklearn.feature_extraction.text import TfidfVectorizer

print(1)
tv = TfidfVectorizer(max_features = 25000)
features = list(stem)
features = tv.fit_transform(features).toarray() 
print(features.shape)

print(2)
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier


#train test split
x_train,x_test,y_train,y_test = train_test_split(features,y,test_size= 0.1)

print(3)
# #Using linear support vector classifier
lsvc = LinearSVC()
# training the model
lsvc.fit(x_train, y_train)
# getting the score of train and test data
print(lsvc.score(x_train, y_train)) #0.98705
print(lsvc.score(x_test, y_test))   #0.8856 


# model 2:-
#Using Gaussuan Naive Bayes
gnb = GaussianNB()
gnb.fit(x_train, y_train)
print(gnb.score(x_train, y_train))  # 
print(gnb.score(x_test, y_test))    #
 
# model 3:-
# Logistic Regression
lr = LogisticRegression()
lr.fit(x_train, y_train)
print(lr.score(x_train, y_train))   # 
print(lr.score(x_test, y_test))     # 

# model 4:-
# Random Forest Classifier
rfc = RandomForestClassifier(n_estimators = 10, random_state = 0)
rfc.fit(x_train, y_train)
print(rfc.score(x_train, y_train))  # 
print(rfc.score(x_test, y_test))    # 

# model 5
# Decision Tree Classifier
DST = DecisionTreeClassifier(criterion="gini").fit(x_train,y_train)
print(DST.score(x_train, y_train))  # 
print(DST.score(x_test, y_test))    # 

# model 6
KNN
KNN = KNeighborsClassifier(n_neighbors = 3).fit(x_train, y_train)
print(KNN.score(x_train, y_train))  # 
print(KNN.score(x_test, y_test))    # 

from sklearn.metrics import confusion_matrix