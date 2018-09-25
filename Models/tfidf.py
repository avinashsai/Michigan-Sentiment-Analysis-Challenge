import sys
import os
import re
import numpy as np 
from sklearn.feature_extraction.text import TfidfVectorizer

def generate_features(Xtrain,Xtest):
	
	vectorizer = TfidfVectorizer(min_df=1,max_df=0.8,use_idf=True,sublinear_tf=True,stop_words='english')
	train_tf = vectorizer.fit_transform(Xtrain)
	test_tf = vectorizer.transform(Xtest)

	return train_tf,test_tf