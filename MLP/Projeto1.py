import sys
import math
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import argparse

from scipy import stats
from sklearn import preprocessing
from MLPClass import MLPClass


global ARGS

def parse_arguments():
	parser = argparse.ArgumentParser()
	parser.add_argument('dataset', type=str, help='Describes the name of dataset(wine or tracks)')
	parser.add_argument('lowWeight', type=float, default = -1, help='Describes the low weight for initialization of parameters.')
	parser.add_argument('highWeight', type=float, default = 1, help='Describes the high weight for initialization of parameters.')
	parser.add_argument('neuronsOnHiddenLayer', type=str,  help='Describes the number of neurons on the hidden layers.')
	parser.add_argument('maxNumberOfIterations', type=int, default=500, help='Describes the max number of iterations')	
	parser.add_argument('tol1', type=float, default = 0.01, help='Describes the tolerance for example on every element from training set.')
	parser.add_argument('tol2', type=float, default = 0.05, help='Describes the tolerance for mean squared error.')
	parser.add_argument('learningRate', type=float, default = 0.3, help='Describes the value for learning Rate')
	parser.add_argument('alphaMomentum', type=float, default = 0.9, help='Describes the value for alpha on momentum.')
	parser.add_argument('sizeMiniBatch', type=int, default = 100, help='Describes the size of minibatch.')
	ARGStemp = parser.parse_args()
	return ARGStemp


# Outliers Treatment
def Outliers_Treatment(data):
	Predictive      = data
	Mean            = []
	Z_score         = []
	Outliers        = []
	threshold       = 3
	data_Predictive = []
	
	# Data standarization 
	for i in range(Predictive.shape[1]):
		z = np.abs(stats.zscore(Predictive.iloc[:,i]))
		Z_score.append(z)

	# Detecting outliers and deleting  
	for i in range(len(Z_score)): 
		Z_score_current = Z_score[i]
		outliers        = []
		new_Predictive  = []

		for j in range(len(Z_score_current)):
			value                 = Z_score_current[j]
			Predictive_current    = Predictive.iloc[:,i]
			Predictive_current[j] = np.where(value > threshold, 0 , Predictive_current[j])
			
			if (value > threshold):
				outliers.append(value)

		# Calculating the mean of a attribute without outliers        
		mean = Predictive_current.mean()
		Mean.append(mean)
		
		# Changing outliers by mean
		for j in range(len(Z_score_current)):
			value                 = Z_score_current[j]
			Predictive_current    = Predictive.iloc[:,i]
			new_Predictive = np.where(value > threshold, Mean[i] , Predictive_current)
   
		data_Predictive.append(new_Predictive)	
		Outliers.append(outliers)		
	return Outliers, data_Predictive

def getMiniBatches(x, y, size): 
	mini_batches  = []    
	try:
		aux = y.shape[1]        
	except:
		aux = 1        
	
	data          = np.hstack((x, np.array(y).reshape(y.shape[0], aux)))    
	n_minibatches = data.shape[0] // size 
	i             = 0
	
   # np.random.shuffle(data) # Random data
	for i in range(n_minibatches + 1):
		mb     = np.array(data[i * size:(i + 1)*size, :])         
		x_mini = mb[:, :-aux]
		y_mini = mb[:, x.shape[1]:].reshape((-aux, aux))    
		mini_batches.append((x_mini, y_mini)) 

	if data.shape[0] % size != 0: # If there needs to create one adittional mini batch
		mb     = data[i * size:data.shape[0]] 
		x_mini = mb[:, :-aux] 
		y_mini = mb[:, x.shape[1]:].reshape((-aux, aux))
		mini_batches.append((x_mini, y_mini))

	return np.array(mini_batches)


def preprocess_dataset(name_dataset):
	if name_dataset == 'wine':
		# Load data 
		dataset_1 = pd.read_csv("winequality-red.csv")

		# Defining predictive and target attributes
		Predictive_1  = dataset_1.drop(['Unnamed: 0','category'], axis = 1)
		data_Target_1 = dataset_1['category']
		new_data_Target_1 = []
		for i in range(len(data_Target_1)):
			if data_Target_1[i] == 'Bad':
				element = [1, 0, 0]
			elif data_Target_1[i] == 'Mid':
				element = [0, 1, 0]
			elif data_Target_1[i] == 'Good':
				element = [0, 0, 1]
			new_data_Target_1.append(element)

		# PREPROCESSING
		# Outliers Treatment
		Outliers, data_Predictive_1 = Outliers_Treatment(Predictive_1)		 
		# Standardize the data attributes
		std_data_1 = preprocessing.normalize(data_Predictive_1)

				# The number of classes that has dataset wine
		return np.array(std_data_1.T), np.array(new_data_Target_1)

	elif name_dataset == 'tracks':
		# Load data
		data_2 = pd.read_csv("default_features_1059_tracks.txt", header = None)

		# Defining predictive and target attributes
		Predictive_2  = data_2.drop([68,69], axis = 1)
		data_Target_2 = data_2.iloc[:, [68,69]]

		# PREPROCESSING
		# Outliers Treatment
		Outliers, data_Predictive_2 = Outliers_Treatment(Predictive_2)

		# Standardize the data attributes
		std_data_2 = preprocessing.normalize(data_Predictive_2)
		 # The number of class that has tracks dataset
		return np.array(std_data_2.T), np.array(data_Target_2)
	else:
		sys.exit('The name of dataset is invalid.')

def split_dataset(data, target):
	if len(data) > 1200:
		size_train = 1200
		# size_test  = len(data) - 1200
	else:
		size_train = 700
		# size_test  = len(data) - 700
	
	try:
		target_train = target[0:size_train, :]
		target_test  = target[size_train:, :]
	except:
		target_train = target[0:size_train]
		target_test  = target[size_train:]

	data_train   = data[0:size_train,:]	
	data_test    = data[size_train:,:]
	

	return data_train, data_test, target_train, target_test



if __name__ == '__main__':
	global ARGS
	print('Processing arguments...')
	ARGS                      = parse_arguments()
	neuronsOnHiddenLayer      = [int(strDim) for strDim in ARGS.neuronsOnHiddenLayer[1:-1].split(',')]
	ARGS.neuronsOnHiddenLayer = neuronsOnHiddenLayer

	print('Preprocessing dataset selected...')
	data, target = preprocess_dataset(ARGS.dataset)

	print('Splitting into training and testing...')
	data_train, data_test, target_train, target_test = split_dataset(np.array(data), np.array(target))

	print('Getting mini-batches')
	mini_batches  = getMiniBatches(data_train, target_train, ARGS.sizeMiniBatch)
	size_x = len(mini_batches[0][0][0])

	if ARGS.dataset == 'wine':
		size_y = 3
	elif ARGS.dataset == 'tracks':
		size_y = 2

	MLPClassifier = MLPClass(ARGS.lowWeight, ARGS.highWeight, ARGS.neuronsOnHiddenLayer, size_x, size_y)
	# print('len(mini_batches)')
	# print((mini_batches[12]))	
	for mb in mini_batches:
		if len(mb[0]) != 0:	# Checks that mini batch has data
			x_mini = mb[0]
			y_mini = mb[1]

			# print('x_mini')
			# print(x_mini)			
			MLPClassifier.fit_by_mini_batch(x_mini, y_mini, ARGS.learningRate, ARGS.alphaMomentum)

