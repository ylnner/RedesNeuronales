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
from sklearn.metrics import confusion_matrix


global ARGS

def parse_arguments():
	parser = argparse.ArgumentParser()
	parser.add_argument('dataset', type=str, help='Describes the name of dataset(wine or tracks)')
	parser.add_argument('lowWeight', type=float, default = -1, help='Describes the low weight for initialization of parameters.')
	parser.add_argument('highWeight', type=float, default = 1, help='Describes the high weight for initialization of parameters.')
	parser.add_argument('neuronsOnHiddenLayer', type=str,  help='Describes the number of neurons on the hidden layers.')
	parser.add_argument('maxNumberOfIterations', type=int, default=500, help='Describes the max number of iterations')		
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
	
	np.random.shuffle(data) # Random data
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
		return np.array(std_data_1.T), np.array(new_data_Target_1), data_Target_1

	elif name_dataset == 'tracks':
		# Load data
		data_2 = pd.read_csv("default_features_1059_tracks.txt", header = None)

		x = data_2.values.astype(float)
		min_max_scaler = preprocessing.MinMaxScaler()
		x_scaled = min_max_scaler.fit_transform(x)
		data_2 = pd.DataFrame(x_scaled)

		# print(data_2)

		# # Defining predictive and target attributes
		Predictive_2  = data_2.drop([68,69], axis = 1)
		data_Target_2 = data_2.iloc[:, [68,69]]


		# Create x, where x the 'scores' column's values as floats
		# std_data_2 = []

	
		# PREPROCESSING
		# Outliers Treatment
		Outliers, data_Predictive_2 = Outliers_Treatment(Predictive_2)

		# Standardize the data attributes
		std_data_2 = preprocessing.normalize(data_Predictive_2)
		# data_Target_2 = preprocessing.normalize(data_Target_2)
		 # The number of class that has tracks dataset
		return np.array(std_data_2.T), np.array(data_Target_2)
	else:
		sys.exit('The name of dataset is invalid.')

def split_dataset(data, target, old_target = []):
	size_train = int(np.ceil(len(data)*0.75))
	print('size_train')
	print(size_train)

	try:
		target_train = target[0:size_train, :]
		target_test  = target[size_train:, :]
	except:
		target_train = target[0:size_train]
		target_test  = target[size_train:]
	
	data_train   = data[0:size_train,:]	
	data_test    = data[size_train:,:]
	
	if len(old_target) != 0:
		old_target_test = old_target[size_train:]
		return data_train, data_test, target_train, target_test, old_target_test
	
	return data_train, data_test, target_train, target_test

def save_value(x,y):
    if y == 0:
        return 0
    return x / y

# Multiclass Confusion Matrix
# Entries:
# y_true: true values of the classification
# y_predict: predict values of the classification
# C: quantity of classes 

# Making confuxion matrix
def MultiClassConfusionMatrix(y_true,y_pred):

    C           = np.unique(y_true)
    D           = len(C)
    CM          = confusion_matrix(y_true, y_pred)     # Matriz de confusion general 
    print('CM')
    print(CM)
    accuracy    = np.zeros(D)
    precision   = np.zeros(D)
    recall      = np.zeros(D)
    specificity = np.zeros(D)
    
    for i in range(D):
        atributo = C[i]
        row_i    = CM[i,:]
        col_i    = CM[:,i]
        
        row_i_without_i = np.delete(row_i,i,0)
        col_i_without_i = np.delete(col_i,i,0)
        del_row_i       = np.delete(CM,i,0)
        del_col_i       = np.delete(del_row_i,i,1)
        
        TP = CM[i,i]
        FN = np.sum(row_i_without_i)
        FP = np.sum(col_i_without_i)
        TN = np.sum(del_col_i)
       
        CM_new = [[TN,FP],[FN,TP]]
        CM_new = np.array(CM_new) 

        div1 = TP+TN+FP+FN
        div2 = TP+FP
        div3 = TP+FN
        div4 = TN+FP
        
        accuracy[i]    = save_value((TP+TN),div1)
    return accuracy



if __name__ == '__main__':
	
	global ARGS
	# print('Processing arguments...')
	ARGS                      = parse_arguments()
	neuronsOnHiddenLayer      = [int(strDim) for strDim in ARGS.neuronsOnHiddenLayer[1:-1].split(',')]
	ARGS.neuronsOnHiddenLayer = neuronsOnHiddenLayer

	# print('Preprocessing dataset selected...')
	
	if ARGS.dataset == 'wine':
		data, target, old_target = preprocess_dataset(ARGS.dataset)
		data_train, data_test, target_train, target_test, old_target_test = split_dataset(np.array(data), np.array(target), old_target)
	else:
		data, target, old_target = preprocess_dataset(ARGS.dataset)
		data_train, data_test, target_train, target_test = split_dataset(np.array(data), np.array(target))

	# print('Splitting into training and testing...')
	

	# print('Getting mini-batches')
	mini_batches  = getMiniBatches(data_train, target_train, ARGS.sizeMiniBatch)
	size_x = len(mini_batches[0][0][0])

	if ARGS.dataset == 'wine':
		size_y = 3
	elif ARGS.dataset == 'tracks':
		size_y = 2

	MLPClassifier = MLPClass(ARGS.lowWeight, ARGS.highWeight, ARGS.neuronsOnHiddenLayer, size_x, size_y)
	# print('len(mini_batches)')
	# print((mini_batches[12]))	
	w_middle = []
	w_output = []
	wb_middle= []
	wb_output= []
	
	for epoch in range(ARGS.maxNumberOfIterations):		
		nb = 0
		for mb in mini_batches:
			print('Numero Batch: ', nb)
			if len(mb[0]) != 0:	# Checks that mini batch has data
				x_mini = mb[0]
				y_mini = mb[1]
				
				w_middle, w_output, wb_middle, wb_output = MLPClassifier.fit_by_mini_batch(x_mini, y_mini, ARGS.learningRate, ARGS.alphaMomentum)
			nb = nb + 1
		errorGlobal = 0
		
		y_pred = []
		for i in range(len(data_test)):													
			y_layer, y_net, error = MLPClassifier.forward(data_test[i], target_test[i], w_middle, w_output, wb_middle, wb_output)			
			# print('y_net: ', y_net)
			result = np.where(y_net == np.amax(y_net))
			# print('result[0]: ', result[0][0])
			if result[0][0] == 0:
				strResult = 'Bad'
			elif result[0][0] == 1:
				strResult = 'Mid'
			else:
				strResult = 'Good'
			y_pred.append(strResult)
			# print('target_test[i]: ', target_test[i])
			# y_net = numpy.where(condition[, x, y])
			errorGlobal = errorGlobal + error

		errorGlobal = errorGlobal / len(data_test)
		print('old_target_test')
		print(np.array(old_target_test))
		print('y_pred')
		print(np.array(y_pred))
		print(MultiClassConfusionMatrix(np.array(old_target_test),np.array(y_pred)))
		
		print('errorGlobal on epoch: ', errorGlobal, epoch)