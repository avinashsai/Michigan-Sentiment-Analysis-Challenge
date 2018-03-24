#get_ipython().system(u'conda install gensim -y')
import os
import re
import numpy as np
import pandas as pd

import nltk
import sklearn
from sklearn.metrics import accuracy_score,f1_score,confusion_matrix
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from sklearn.utils import shuffle

nltk.download('punkt')
nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

stopword = stopwords.words('english')

with open('rt-polarity.pos','r',encoding='latin1') as f:
    pos = f.readlines()

with open('rt-polarity.neg','r',encoding='latin1') as f:
    neg = f.readlines()


def preprocess(sentence):
    sentence = re.sub(r'[^\w\s]'," ",str(sentence))
    sentence = re.sub(r'[^a-zA-Z]'," ",str(sentence))
    sents = word_tokenize(sentence)
    new_sents = " "
    for i in range(len(sents)):
        if(sents[i].lower() not in stopword and len(sents[i])>1):
            new_sents+=sents[i].lower()+" "
    return new_sents


labels = np.zeros(10662)
labels[0:5331] = 1


corpus1 = []


for i in range(5331):
    sen = pos[i]
    sen = sen[0:len(sen)-1]
    corpus1.append(preprocess(sen))


for i in range(5331):
    sen = neg[i]
    sen = sen[0:len(sen)-1]
    corpus1.append(preprocess(sen))


train_data,test_data,train_labels,test_labels = train_test_split(corpus1,labels,test_size=0.3,random_state=42)


import gensim
from gensim.models import doc2vec
from gensim.models.doc2vec import Doc2Vec,TaggedDocument


def tagsentences(sentence,labels):
    return TaggedDocument(sentence,labels)


make_sentences = []

for i in range(len(train_data)):
    make_sentences.append(tagsentences(train_data[i],["train_"+str(i)]))

for i in range(len(test_data)):
    make_sentences.append(tagsentences(test_data[i],["test_"+str(i)]))

print(make_sentences[0:10])


model_sent = Doc2Vec(min_count=1,vector_size=100,window=15,workers=1,seed=0)

model_sent.build_vocab(make_sentences)

for epoch in range(50):
    model_sent.train(make_sentences,total_examples=model_sent.corpus_count,epochs=model_sent.iter)


train_length = len(train_data)
test_length = len(test_data)
print(train_length)
print(test_length)


train_arrays = np.zeros((train_length,100))
label_train = np.zeros(train_length)


test_arrays = np.zeros((test_length,100))
label_test = np.zeros(test_length)



for i in range(train_length):
    vects = model_sent.docvecs['train_'+str(i)]
    for j in range(100):
        train_arrays[i][j] = vects[j]


for i in range(test_length):
    vects = model_sent.docvecs['test_'+str(i)]
    for j in range(100):
        test_arrays[i][j] = vects[j]


print(train_arrays.shape)
print(test_arrays.shape)


lr_classifier = LogisticRegression()
lr_classifier.fit(train_arrays,train_labels)
lr_predict = lr_classifier.predict(test_arrays)
lr_accuracy = accuracy_score(lr_predict,test_labels)
lr_cm = confusion_matrix(test_labels,lr_predict)
lr_f1 = f1_score(test_labels,lr_predict)
print("The Logistic Regression Classifier Accuracy is :" ,lr_accuracy)
print("F1 Score is :" ,lr_f1)
print("The Confusion Matrix is :")
print(lr_cm)


rbf_classifier = SVC()
rbf_classifier.fit(train_arrays,train_labels)
rbf_predict = rbf_classifier.predict(test_arrays)
rbf_accuracy = accuracy_score(rbf_predict,test_labels)
rbf_cm = confusion_matrix(test_labels,rbf_predict)
rbf_f1 = f1_score(test_labels,rbf_predict)
print("The Support Vector Classifier Accuracy is :" ,rbf_accuracy)
print("F1 Score is :" ,rbf_f1)
print("The Confusion Matrix is :")
print(rbf_cm)

