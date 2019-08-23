import numpy as np
import math

x_global = np.array([[1, 0, 0, 0, 0, 0, 0, 0],
			  [0, 1, 0, 0, 0, 0, 0, 0],
			  [0, 0, 1, 0, 0, 0, 0, 0],
			  [0, 0, 0, 1, 0, 0, 0, 0],
			  [0, 0, 0, 0, 1, 0, 0, 0],
			  [0, 0, 0, 0, 0, 1, 0, 0],
			  [0, 0, 0, 0, 0, 0, 1, 0],
			  [0, 0, 0, 0, 0, 0, 0, 1]])	  
#x_global = np.array([[1, 1], [0, 0], [0, 1], [1, 0]])
			
y_global = np.array([[1, 0, 0, 0, 0, 0, 0, 0],
			  [0, 1, 0, 0, 0, 0, 0, 0],
			  [0, 0, 1, 0, 0, 0, 0, 0],
			  [0, 0, 0, 1, 0, 0, 0, 0],
			  [0, 0, 0, 0, 1, 0, 0, 0],
			  [0, 0, 0, 0, 0, 1, 0, 0],
			  [0, 0, 0, 0, 0, 0, 1, 0],
			  [0, 0, 0, 0, 0, 0, 0, 1]])
#y_global = [[0], [0], [1], [1]]

def sigmoid(x):
	out = []
	for i in range(len(x)):
		out.append(1 / (1 + math.exp(-x[i])))
	return np.array(out)

def forward(x, y, w_input, w_output, wb_input, wb_output):

	# Hidden Layer
	x 			= np.append(np.array(x), 1)   # bias initialization for every example
	#w_current 	= np.hstack((w_input, np.atleast_2d(wb_input).T))	
	w_current = np.concatenate((w_input, wb_input.T), axis = 1)
	y_layer   	= sigmoid(np.dot(w_current, x))

	# Output Layer
	y_layer_bias= np.append(y_layer, 1)  # bias initialization
	# w_current 	= np.hstack((w_output, np.atleast_2d(wb_output).T))
	w_current = np.concatenate((w_output, wb_output.T), axis = 1)

	y_net 		= sigmoid(np.dot(w_current, y_layer_bias))

	# Calculating error
	error 		= (np.sum((y - y_net)**2))/2

	return y_layer, y_net, error

def backpropagation(x_global, y_global, w_input, w_output, wb_input, wb_output, learning_rate, number_iterations):
	tol1  = 0.01
	tol2  = 0.05
	
	for ni in range(number_iterations):
		y_net_best = []
		for i in range(len(x_global)):
			error = 1
			while error > tol1:				
				y_layer, y_net, error = forward(x_global[i], y_global[i], w_input, w_output, wb_input, wb_output)												
				# Calculate delta w output
				delta_o        = -(y_global[i] - y_net) * y_net * (np.ones(len(y_net)) - y_net)  	# vector [1 x 8]
				delta_w_output = np.array((delta_o[np.newaxis]).T * y_layer)						# matrix [8 x 3]
				old_w_output   = w_output
				w_output       = w_output - (learning_rate * delta_w_output)						# matrix [8 x 3]
				wb_output      = wb_output - (learning_rate * delta_o)								# matrix [1 x 8]

				
				# Calculate delta w input	
				temp           = np.ones(len(y_layer)) - y_layer
				# print('temp')
				# print(temp)
				# print('delta_o')
				# print(delta_o)
				# print('w_output')
				# print(w_output)
				# print('np.dot(delta_o , w_output)')
				# print(np.dot(delta_o , w_output))


				delta_o_hidden = (np.dot(delta_o , old_w_output) * y_layer * temp)
				# print('delta_o_hidden')
				# print(delta_o_hidden)
				# print('[np.newaxis].T')
				# print(delta_o_hidden[np.newaxis].T)
				# print('x_global[i]')
				# print(x_global[i])				
				delta_w_input  = np.array(delta_o_hidden[np.newaxis].T * x_global[i])							# matrix [3 x 8]	
				# print('delta_w_input')
				# print(delta_w_input)
				w_input        = w_input - (learning_rate * delta_w_input)							# matrix [3 x 8]
				wb_input       = wb_input - (learning_rate * delta_o_hidden)						# matrix [1 x 3]	


				# print('w_input')
				# print(w_input)

				# print('wb_input')
				# print(wb_input)

			y_net_best.append(y_net)
			
					
		mse = np.sum(np.subtract(np.array(y_global), np.array(y_net_best)) ** 2)
		print('y_net_best')
		print(y_net_best)
		print('mse: ', mse)

		if mse < tol2:
			print('MSE was achieved')
			print(mse)
			break
				
	return w_input, wb_input, w_output, wb_output




if __name__ == '__main__':
	nneurons_hidden = 3
	learning_rate   = 0.4
	low_weight      = -1
	high_weight     = 1
	# Weight initialization first layer  
	w_input         = np.random.uniform(low = low_weight, high = high_weight, size = [nneurons_hidden, len(x_global)])
	wb_input        = np.random.uniform(low = low_weight, high = high_weight, size =  [1, nneurons_hidden])
	# w_input        = np.array([[0.4, 0.5], [0.8,0.8]])
	# wb_input		= np.array([[-0.6, -0.2]])
	
	# Weight initialization last layer  
	w_output        = np.random.uniform(low = low_weight, high = high_weight, size = [len(y_global), nneurons_hidden])
	wb_output       = np.random.uniform(low = low_weight, high = high_weight, size = [1, len(y_global)])
	# w_output     = np.array([[-0.4, 0.9]])
	# wb_output 	 = np.array([[-0.3]])

	
	number_iterations = 3000
	backpropagation(x_global, y_global, w_input, w_output, wb_input, wb_output, learning_rate, number_iterations)		