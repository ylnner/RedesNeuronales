import numpy as np
import sys
import math
import argparse

global ARGS

def parse_arguments():
	parser = argparse.ArgumentParser()
	parser.add_argument('exercise', type=str, help='Describes the number of exercise (exercise1, exercise2, exercise3). This changes the training and label sets')
	parser.add_argument('lowWeight', type=float, default = -1, help='Describes the low weight for initialization of parameters.')
	parser.add_argument('highWeight', type=float, default = 1, help='Describes the high weight for initialization of parameters.')
	parser.add_argument('numberNeuronsHidden', type=int, default = 3, help='Describes the number of neurons on the hidden layer.')
	parser.add_argument('maxNumberOfIterations', type=int, default=500, help='Describes the max number of iterations')	
	parser.add_argument('tol1', type=float, default = 0.01, help='Describes the tolerance for example on every element from training set.')
	parser.add_argument('tol2', type=float, default = 0.05, help='Describes the tolerance for mean squared error.')
	parser.add_argument('learningRate', type=float, default = 0.3, help='Describe the learningRate')
	ARGStemp = parser.parse_args()
	return ARGStemp

def sigmoid(x):
	out = []
	for i in range(len(x)):
		out.append(1 / (1 + math.exp(-x[i])))
	return np.array(out)

def forward(x, y, w_middle, w_output, wb_middle, wb_output):
	# Middle Layer
	y_layer = []
	input = np.array(x)
	for w_neurons, w_bias in zip(w_middle, wb_middle):
		input_with_bias = np.append(input, 1)
		print('input_with_bias')
		print(input_with_bias)

		w_neurons       = np.array(w_neurons)
		print('w_neurons')
		print(w_neurons)

		w_bias          = np.array(w_bias)
		print('w_bias')
		print(w_bias)

		w_current       = np.concatenate((w_neurons, w_bias.T), axis = 1)
		print('w_current')
		print(w_current)

		output          = sigmoid(np.dot(w_current, input_with_bias))
		y_layer.append(output)
		print('output')
		print(output)

		input           = output

			
	# Output Layer
	#y_layer_bias= np.append(y_layer, 1)  # bias initialization	
	y_layer_bias= np.append(input, 1)  # bias initialization	
	w_current = np.concatenate((w_output, wb_output.T), axis = 1)
	y_net 		= sigmoid(np.dot(w_current, y_layer_bias))

	# Calculating error
	error 		= (np.sum((y - y_net)**2))/2

	print('y_layer')
	print(np.array(y_layer))

	return y_layer, y_net, error


def backpropagation(x_global, y_global, w_middle, w_output, wb_middle, wb_output):
	global ARGS

	y_layer, y_net, error = forward(x_global[0], y_global[0], w_middle, w_output, wb_middle, wb_output)

	achieved = False
	for ni in range(ARGS.maxNumberOfIterations):
		y_net_best = []
		for i in range(len(x_global)):
			error = 1
			while error >= ARGS.tol1:				
				y_layer, y_net, error = forward(x_global[i], y_global[i], w_middle, w_output, wb_middle, wb_output)
				# Calculate delta w output
				delta_o        = -(y_global[i] - y_net) * y_net * (np.ones(len(y_net)) - y_net)  	# [1x1]    # vector [1 x 8]
				delta_w_output = np.array((delta_o[np.newaxis]).T * y_layer)						# [1x2]    # matrix [8 x 3]
				old_w_output   = w_output
				w_output       = w_output - (ARGS.learningRate * delta_w_output)					# [1x2]    # matrix [8 x 3]
				wb_output      = wb_output - (ARGS.learningRate * delta_o)							# [1x1]    # matrix [1 x 8]

				
				# Calculate delta w input	
				temp           = np.ones(len(y_layer)) - y_layer				
				delta_o_hidden = (np.dot(delta_o , old_w_output) * y_layer * temp)				
				delta_w_input  = np.array(delta_o_hidden[np.newaxis].T * x_global[i])				# matrix [3 x 8]					
				w_input        = w_input - (ARGS.learningRate * delta_w_input)							# matrix [3 x 8]
				wb_input       = wb_input - (ARGS.learningRate * delta_o_hidden)						# matrix [1 x 3]

				# Calculate delta w middle
				for idx in range(len(y_layer), 0, -1):
					layer           = y_layer[idx]
					temp            = np.ones(len(layer)) - layer
					delta_o_hidden  = (np.dot(delta_o, old_w_output) * layer * temp)
					delta_w_current = np.array(delta_o_hidden[np.newaxis].T * x_global[i])
					w_middle[idx]   = w_middle[idx] - (ARGS.learningRate * delta_w_current)
					wb_middle[idx]  = wb_middle[idx] - (ARGS.learningRate * delta_o_hidden)

					delta_o 		= -(y_global[i] - layer) * layer * (np.ones(len(layer)) - layer)
					old_w_output    = w_middle[idx]
			y_net_best.append(y_net)
			print('alcance')
					
		mse = np.sum(np.subtract(np.array(y_global), np.array(y_net_best)) ** 2)		
		
		if mse < ARGS.tol2:
			achieved = True
			print('MSE was achieved: ', mse)
			print(np.array(y_net_best))
			print('w_input')
			print(w_input)
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
		size = 8
		x_global = np.identity(size)
		y_global = np.identity(size)

	elif ARGS.exercise =='exercise3':
		# Size of the autoencoder
		size = 15
		x_global = np.identity(size)
		y_global = np.identity(size)
		
	else:
		sys.exit('The number of exercise is invalid.')



	neurons_on_layers = [8, 3, 1]

	#Initialization of weights
	w_middle  = []
	wb_middle = []
	before    = len(x_global[0])
	for neurons in neurons_on_layers:
		w_temp  = np.random.uniform(low = ARGS.lowWeight, high = ARGS.highWeight, size = [neurons, before])
		wb_temp = np.random.uniform(low = ARGS.lowWeight, high = ARGS.highWeight, size = [1, neurons])
		w_middle.append(w_temp)
		wb_middle.append(wb_temp)
		before  = neurons 		# At the end, before is going to have the last number of neurons of the last layer

	# Weight initialization last layer  
	w_output        = np.random.uniform(low = ARGS.lowWeight, high = ARGS.highWeight, size = [len(y_global[0]), before])
	wb_output       = np.random.uniform(low = ARGS.lowWeight, high = ARGS.highWeight, size = [1, len(y_global[0])])


	# # Weight initialization first layer  
	# w_input         = np.random.uniform(low = ARGS.lowWeight, high = ARGS.highWeight, size = [ARGS.numberNeuronsHidden, len(x_global[0])])
	# wb_input        = np.random.uniform(low = ARGS.lowWeight, high = ARGS.highWeight, size =  [1, ARGS.numberNeuronsHidden])
		
	# # Weight initialization last layer  
	# w_output        = np.random.uniform(low = ARGS.lowWeight, high = ARGS.highWeight, size = [len(y_global[0]), ARGS.numberNeuronsHidden])
	# wb_output       = np.random.uniform(low = ARGS.lowWeight, high = ARGS.highWeight, size = [1, len(y_global[0])])


	print('=============== INPUT ===============')
	print('Exercise: ', ARGS.exercise)
	print('Low Weight: ', ARGS.lowWeight)
	print('High Weight: ', ARGS.highWeight)
	print('Number of neurons on hidden layer: ', ARGS.numberNeuronsHidden)
	print('Max number of iterations: ', ARGS.maxNumberOfIterations)
	print('Tolerance 1(for example): ', ARGS.tol1)
	print('Tolerance 2(for MSE): ', ARGS.tol2)
	print('Learning Rate: ', ARGS.learningRate)
	
	backpropagation(x_global, y_global, w_middle, w_output, wb_middle, wb_output)