from __future__ import print_function
import os
import re
import numpy as np 
import pandas as pd 

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

stopword = stopwords.words('english')

def preprocess(text):
	text = re.sub(r"it\'s","it is",str(text))
	text = re.sub(r"i\'d","i would",str(text))
	text = re.sub(r"don\'t","do not",str(text))
	text = re.sub(r"he\'s","he is",str(text))
	text = re.sub(r"there\'s","there is",str(text))
	text = re.sub(r"that\'s","that is",str(text))
	text = re.sub(r"can\'t", "can not", text)
	text = re.sub(r"cannot", "can not ", text)
	text = re.sub(r"what\'s", "what is", text)
	text = re.sub(r"What\'s", "what is", text)
	text = re.sub(r"\'ve ", " have ", text)
	text = re.sub(r"n\'t", " not ", text)
	text = re.sub(r"i\'m", "i am ", text)
	text = re.sub(r"I\'m", "i am ", text)
	text = re.sub(r"\'re", " are ", text)
	text = re.sub(r"\'d", " would ", text)
	text = re.sub(r"\'ll", " will ", text)
	text = re.sub(r"\'s"," is",text)
	text = re.sub(r"[0-9]"," ",str(text))
	words = text.split()

	return " ".join(word.lower() for word in words if word.lower() not in stopword)

  
def read_train_data():
	X = []
	y = np.zeros(7086)

	with open('../Dataset/training.txt','r',encoding='latin1') as f:
		i = 0
		for line in f:
			label,text = line.split("	")
			X.append(preprocess(text[:-1]))
			y[i] = int(label)
			i+=1
	return X,y

def read_test_data():
	X = []

	with open('../Dataset/testdata.txt','r',encoding='latin1') as f:
		for line in f:
			X.append(preprocess(line[:-1]))
	return X