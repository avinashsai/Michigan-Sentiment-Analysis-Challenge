from __future__ import print_function
import os
import re
import numpy as np 
import pandas as pd 

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import accuracy_score

def predict_labels(Xtrain,ytrain,Xtest):

	classifiers = [LogisticRegression(),SVC(gamma=0.3),SVC(kernel='linear'),MultinomialNB()]
	classify = ["LR","SVM-RBF","SVM-L","MNB"]
	filenames = ['lr_pred.txt','svmrbf_pred.txt','svmlinear_pred.txt','mnb_pred.txt']
	i = 0

	for classifier in classifiers:
		classi = classifier
		classi.fit(Xtrain,ytrain)
		train_pred = classi.predict(Xtrain)
		train_acc = accuracy_score(train_pred,ytrain)
		print(classify[i]+" "+"Accuracy score is :",train_acc)
		ypred = classi.predict(Xtest)
		np.savetxt(filenames[i],ypred,fmt="%d",delimiter=" ")
		i+=1
