import numpy as np
import sys
import math
import argparse

# from sklearn.linear_model import KFold

global ARGS

def parse_arguments():
	parser = argparse.ArgumentParser()
	parser.add_argument('exercise', type=str, help='Describes the number of exercise (exercise1, exercise2, exercise3). This changes the training and label sets')
	parser.add_argument('lowWeight', type=float, default = -1, help='Describes the low weight for initialization of parameters.')
	parser.add_argument('highWeight', type=float, default = 1, help='Describes the high weight for initialization of parameters.')
	parser.add_argument('neuronsOnHiddenLayer', type=str,  help='Describes the number of neurons on the hidden layers.')
	parser.add_argument('maxNumberOfIterations', type=int, default=500, help='Describes the max number of iterations')	
	parser.add_argument('tol1', type=float, default = 0.01, help='Describes the tolerance for example on every element from training set.')
	parser.add_argument('tol2', type=float, default = 0.05, help='Describes the tolerance for mean squared error.')
	parser.add_argument('learningRate', type=float, default = 0.3, help='Describes the value for learning Rate')
	parser.add_argument('alphaMomentum', type=float, default = 0.9, help='Describes the value for alpha on momentum.')
	ARGStemp = parser.parse_args()
	return ARGStemp

# function to create a list containing mini-batches 
def getMiniBatches(x, y, size): 
	mini_batches  = [] 
	data          = np.hstack((x, y)) 
	np.random.shuffle(data) # Random data

	n_minibatches = data.shape[0] // size 
	i             = 0
  
	for i in range(n_minibatches + 1):
		mb     = data[i * size:(i + 1)*size, :] 
		X_mini = mb[:, :-1] 
		Y_mini = mb[:, -1].reshape((-1, 1)) 
		mini_batches.append((X_mini, Y_mini)) 

	if data.shape[0] % size != 0: # If there needs to create one adittional mini batch
		mb     = data[i * size:data.shape[0]] 
		X_mini = mb[:, :-1] 
		Y_mini = mb[:, -1].reshape((-1, 1)) 		
		mini_batches.append((X_mini, Y_mini))

	return mini_batches 


def sigmoid(x):
	out = []
	for i in range(len(x)):
		aux = 1 / (1 + math.exp(-x[i]))
		# if aux > 0.5:
		# 	aux = 1
		# else:
		# 	aux = 0
		out.append(aux)
	return np.array(out)

def forward(x, y, w_middle, w_output, wb_middle, wb_output):
	# Middle Layer
	y_layer = []
	input = np.array(x)
	for w_neurons, w_bias in zip(w_middle, wb_middle):
		input_with_bias = np.append(input, 1)		
		w_neurons       = np.array(w_neurons)		
		w_bias          = np.array(w_bias)		
		w_current       = np.concatenate((w_neurons, w_bias.T), axis = 1)		
		output          = sigmoid(np.dot(w_current, input_with_bias))
		y_layer.append(output)		
		input           = output

			
	# Output Layer	
	y_layer_bias = np.append(input, 1)  # bias initialization	
	w_current    = np.concatenate((w_output, wb_output.T), axis = 1)
	y_net        = sigmoid(np.dot(w_current, y_layer_bias))
	
	# Calculating error
	error        = (np.sum((y - y_net)**2))/2
	
	return y_layer, y_net, error

def proccess_mini_batch(x_mini, y_mini, w_middle, w_output, wb_middle, wb_output):
	global ARGS

	print('w_middle ANTES')
	print(w_middle)
	print('wb_middle ANTES')
	print(wb_middle)
	print('w_output ANTES')
	print(w_output)	
	print('wb_output ANTES')
	print(wb_output)


	delta_w_output_old  = np.zeros_like(w_output)
	delta_o_old         = np.zeros_like(wb_output[0])	
	delta_w_current_old = []
	delta_o_hidden_old  = []

	batch_delta_w_output         = []
	batch_delta_o                = []
	batch_delta_w_middle_current = []
	batch_delta_o_hidden         = []
	for w, wb in zip(w_middle, wb_middle):
		delta_w_current_old.append(np.zeros_like(w))
		delta_o_hidden_old.append(np.zeros_like(wb[0]))

	for i in range(len(x_mini)):
		print('ENTRO ELEMENTO')
		y_layer, y_net, error = forward(x_mini[i], y_mini[i], w_middle, w_output, wb_middle, wb_output)
		delta_o            = np.array(-(y_global[i] - y_net) * y_net * (np.ones(len(y_net)) - y_net)) + (ARGS.alphaMomentum * delta_o_old)
		delta_o_old        = delta_o
		idx2               = len(y_layer) - 1
		delta_w_output     = np.array((delta_o[np.newaxis]).T * y_layer[idx2]) + (ARGS.alphaMomentum * delta_w_output_old)
		delta_w_output_old = delta_w_output
		old_w_output       = w_output

		batch_delta_w_output.append(delta_w_output)
		batch_delta_o.append(delta_o)

		# w_output           = w_output - (ARGS.learningRate * delta_w_output)
		# wb_output          = wb_output - (ARGS.learningRate * delta_o)

		# Calculate delta w middle
		for idx in range(len(y_layer) - 2, 0, -1):

			if idx == 0:
				input     = x_mini[i]
				layer     = y_layer[idx]
				idx_layer = idx
			else:						
				layer     = y_layer[idx + 1]
				idx_layer = idx + 1
				input     = y_layer[idx]
						
			temp                           = np.ones(len(layer)) - layer				
			delta_o_hidden                 = (np.dot(delta_o, old_w_output) * layer * temp) + (ARGS.alphaMomentum * delta_o_hidden_old[idx_layer])
			delta_o_hidden_old[idx_layer]  = delta_o_hidden
			
			delta_w_middle_current         = np.array(delta_o_hidden[np.newaxis].T * input) + (ARGS.alphaMomentum * delta_w_current_old[idx_layer])
			delta_w_current_old[idx_layer] = delta_w_middle_current

			old_w_output             = w_middle[idx+1]
			print('delta_w_middle_current INSIDE')
			print(delta_w_middle_current)
			print('idx +1: ')
			print(idx + 1)
			print('batch_delta_w_middle_current')
			print(batch_delta_w_middle_current)

			batch_delta_w_middle_current.insert(idx+1, delta_w_middle_current)
			batch_delta_o_hidden.insert(idx+1, delta_o_hidden)
			# w_middle[idx+1]          = w_middle[idx+1] - (ARGS.learningRate * delta_w_middle_current)
			# wb_middle[idx+1]         = wb_middle[idx+1] - (ARGS.learningRate * delta_o_hidden)


	batch_delta_w_output         = np.array(batch_delta_w_output)
	batch_delta_o                = np.array(batch_delta_o)
	batch_delta_w_middle_current = np.array(batch_delta_w_middle_current)
	batch_delta_o_hidden         = np.array(batch_delta_o_hidden)

	geral_delta_w_output         = np.sum(batch_delta_w_output, axis = 0)
	geral_delta_o                = np.sum(batch_delta_o, axis = 0)
	geral_delta_w_middle_current = np.zeros_like(batch_delta_w_middle_current[0])
	geral_delta_o_hidden         = np.zeros_like(batch_delta_o_hidden[0])

	# update weights	
	for current_w_middle, current_o_hidden in zip(batch_delta_w_middle_current, batch_delta_o_hidden):
		geral_delta_o_hidden         = geral_delta_o_hidden + current_o_hidden
		geral_delta_w_middle_current = geral_delta_w_middle_current + current_w_middle

	geral_delta_w_middle_current = np.array(geral_delta_w_middle_current)
	geral_delta_o_hidden         = np.array(geral_delta_o_hidden)

	print('batch_delta_w_middle_current')
	print(batch_delta_w_middle_current)
	print(len(batch_delta_w_middle_current))
	print('geral')
	print(geral_delta_w_middle_current)

	# print('batch_delta_o_hidden')
	# print(batch_delta_o_hidden)
	# print(len(batch_delta_o_hidden))
	# print('geral')
	# print(geral_delta_o_hidden)

	len_batch = len(x_mini)
	w_output  = w_output - (ARGS.learningRate * geral_delta_w_output)
	wb_output = wb_output - (ARGS.learningRate * geral_delta_o)
	idx       = 0
	for current_w_middle, current_o_hidden in zip(geral_delta_w_middle_current, geral_delta_o_hidden):
		print('w_middle[idx]')
		print(w_middle[idx])
		aux = ((ARGS.learningRate / len_batch)* current_w_middle)
		print('aux')
		print(aux)
		print('current_w_middle')
		print(current_w_middle)
		w_middle[idx]  = w_middle[idx] - ((ARGS.learningRate / len_batch)* current_w_middle)
		wb_middle[idx] = wb_middle[idx] - ((ARGS.learningRate / len_batch) * current_o_hidden)
		++idx

	print('w_middle')
	print(w_middle)
	print('wb_middle')
	print(wb_middle)
	print('w_output')
	print(w_output)	
	print('wb_output')
	print(wb_output)

	return w_middle, w_output, wb_middle, wb_output

	


def backpropagation(x_global, y_global, w_middle, w_output, wb_middle, wb_output):
	global ARGS
	achieved = False
	delta_w_output_old = np.zeros_like(w_output)
	delta_o_old        = np.zeros_like(wb_output[0])	
	delta_w_current_old = []
	delta_o_hidden_old  = []

	for w, wb in zip(w_middle, wb_middle):
		delta_w_current_old.append(np.zeros_like(w)) 
		delta_o_hidden_old.append(np.zeros_like(wb[0]))

	print('delta_o_hidden_old')
	print(delta_o_hidden_old)

	print('delta_w_current_old')
	print(delta_w_current_old)

	for i in range(len(w_middle)):
		print('i: ', i)
		print(w_middle[i].shape)


	for ni in range(ARGS.maxNumberOfIterations):
		y_net_best = []
		for i in range(len(x_global)):
			error = 1
			while error >= ARGS.tol1:
				
				y_layer, y_net, error = forward(x_global[i], y_global[i], w_middle, w_output, wb_middle, wb_output)
								
				# Calculate delta w output				
				delta_o            = np.array(-(y_global[i] - y_net) * y_net * (np.ones(len(y_net)) - y_net)) + (ARGS.alphaMomentum * delta_o_old)
				delta_o_old        = delta_o
				idx2               = len(y_layer) - 1
				delta_w_output     = np.array((delta_o[np.newaxis]).T * y_layer[idx2]) + (ARGS.alphaMomentum * delta_w_output_old)
				delta_w_output_old = delta_w_output
				old_w_output       = w_output
				w_output           = w_output - (ARGS.learningRate * delta_w_output)				
				wb_output          = wb_output - (ARGS.learningRate * delta_o)
				print('old_w_output EMPIEZA')
				print(old_w_output.shape)
				
				# print('len(y_layer)')
				# print(y_layer)
				# Calculate delta w middle
				for idx in range(len(y_layer) - 1, 0, -1):
									
					if idx == 0:
						# print('aqui idx = 0')
						input     = x_global[i]
						layer     = y_layer[idx+1]
						# layer     = y_layer[idx]
						idx_layer = idx
					else:					
						# print('aqui else')	
						layer     = y_layer[idx]						
						idx_layer = idx
						input     = y_layer[idx-1]					

					temp                           = np.ones(len(layer)) - layer									
					print('================================================')
					print('idx: ', idx)	
					print('delta_o')
					print(delta_o.shape)

					print('old_w_output')
					print(old_w_output.shape)

					print('layer')
					print(layer.shape)

					print('temp')
					print(temp.shape)

					print('delta_o_hidden_old')
					print('idx_layer: ', idx_layer)
					print(delta_o_hidden_old[idx_layer].shape)

					print('================================================')



					def backward(self, X, y, output):
				        #backward propogate through the network
				        self.output_error = y - output # error in output
				        self.output_delta = self.output_error * self.sigmoid(output, deriv=True)
				        
				        self.z2_error = self.output_delta.dot(self.W2.T) #z2 error: how much our hidden layer weights contribute to output error
				        self.z2_delta = self.z2_error * self.sigmoid(self.z2, deriv=True) #applying derivative of sigmoid to z2 error
				        
				        self.W1 += X.T.dot(self.z2_delta) # adjusting first set (input -> hidden) weights
				        self.W2 += self.z2.T.dot(self.output_delta) # adjusting second set (hidden -> output) weights
								
					
					# print('old_w_output')
					# print(old_w_output)
					delta_o_hidden                 = (np.dot(delta_o, old_w_output) * layer * temp) + (ARGS.alphaMomentum * delta_o_hidden_old[idx_layer])
					delta_o_hidden_old[idx_layer]  = delta_o_hidden

					# print('delta_o_hidden')
					# print(delta_o_hidden.shape)
					
					delta_w_middle_current         = np.array(delta_o_hidden[np.newaxis].T * input) + (ARGS.alphaMomentum * delta_w_current_old[idx_layer])
					delta_w_current_old[idx_layer] = delta_w_middle_current

					old_w_output             = w_middle[idx]	# ACHF - Creo que no se actualiza los pesos old_w_output porque siempre corresponden a los pesos de la ultima camada
					# print('old_w_output')
					# print(old_w_output.shape)
					w_middle[idx]          = w_middle[idx] - (ARGS.learningRate * delta_w_middle_current)
					wb_middle[idx]         = wb_middle[idx] - (ARGS.learningRate * delta_o_hidden)

					# old_w_output             = w_middle[idx]					
					# w_middle[idx]          = w_middle[idx] - (ARGS.learningRate * delta_w_middle_current)
					# wb_middle[idx]         = wb_middle[idx] - (ARGS.learningRate * delta_o_hidden)


									
			y_net_best.append(y_net)
							
		mse = np.sum(np.subtract(np.array(y_global), np.array(y_net_best)) ** 2)		
		
		if mse < ARGS.tol2:
			achieved = True
			print('MSE was achieved: ', mse)
			print(np.array(y_net_best))			
			break
	
	if achieved == False:
		print('MSE was not achieved: ', mse)
		print(np.array(y_net_best))

	return w_middle, wb_middle, w_output, wb_output




if __name__ == '__main__':

	x_global = []
	y_global = []
	global ARGS
	ARGS = parse_arguments()
	if ARGS.exercise =='exercise1':		
		x_global = np.array([[1, 1], [0, 0], [0, 1], [1, 0]])
		y_global = np.array([[0], [0], [1], [1]])		
	elif ARGS.exercise =='exercise2':
		# Size of the autoencoder
		size = 4
		x_global = np.identity(size)
		y_global = np.identity(size)

	elif ARGS.exercise =='exercise3':
		# Size of the autoencoder
		size = 15
		x_global = np.identity(size)
		y_global = np.identity(size)
		
	else:
		sys.exit('The number of exercise is invalid.')


	neuronsOnHiddenLayer = [int(strDim) for strDim in ARGS.neuronsOnHiddenLayer[1:-1].split(',')]
	ARGS.neuronsOnHiddenLayer = neuronsOnHiddenLayer
	print('neuronsOnHiddenLayer')
	print(neuronsOnHiddenLayer)

	#Initialization of weights
	w_middle  = []
	wb_middle = []
	before    = len(x_global[0])
	for neurons in neuronsOnHiddenLayer:
		w_temp  = np.random.uniform(low = ARGS.lowWeight, high = ARGS.highWeight, size = [neurons, before])
		wb_temp = np.random.uniform(low = ARGS.lowWeight, high = ARGS.highWeight, size = [1, neurons])
		w_middle.append(w_temp)
		wb_middle.append(wb_temp)
		before  = neurons 		# At the end, before is going to have the last number of neurons of the last layer

	# Weight initialization last layer  
	w_output        = np.random.uniform(low = ARGS.lowWeight, high = ARGS.highWeight, size = [len(y_global[0]), before])
	wb_output       = np.random.uniform(low = ARGS.lowWeight, high = ARGS.highWeight, size = [1, len(y_global[0])])

	print('w_middle')
	print(w_middle)

	print('wb_middle')
	print(wb_middle)

	
	print('=============== INPUT ===============')
	print('Exercise: ', ARGS.exercise)
	print('Low Weight: ', ARGS.lowWeight)
	print('High Weight: ', ARGS.highWeight)
	print('Neurons on hidden layer: ', ARGS.neuronsOnHiddenLayer)
	print('Max number of iterations: ', ARGS.maxNumberOfIterations)
	print('Tolerance 1(for example): ', ARGS.tol1)
	print('Tolerance 2(for MSE): ', ARGS.tol2)
	print('Learning Rate: ', ARGS.learningRate)
	print('Alpha momentum: ', ARGS.alphaMomentum)
	
	backpropagation(x_global, y_global, w_middle, w_output, wb_middle, wb_output)
	print('empece proccess_mini_batch')
	# proccess_mini_batch(x_global, x_global, w_middle, w_output, wb_middle, wb_output)