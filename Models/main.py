from __future__ import print_function
import os
import re
import numpy as np 
import pandas as pd 

from loader import *

from tfidf import *

from predict import *

def main():
	train_data,train_labels = read_train_data()

	test_data = read_test_data()

	train_features,test_features = generate_features(train_data,test_data)


	predict_labels(train_features,train_labels,test_features)



if __name__ == '__main__':
	main()