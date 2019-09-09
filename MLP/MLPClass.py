import numpy as np
import math

class MLPClass:
	def __init__(self, lowWeight, highWeight, neuronsOnHiddenLayer, size_x, size_y):
		self.lowWeight             = lowWeight
		self.highWeight            = highWeight
		self.neuronsOnHiddenLayer  = neuronsOnHiddenLayer		

		#Initialization of weights
		w_middle  = []
		wb_middle = []
		# before    = len(x_global[0])
		before    = size_x
		for neurons in neuronsOnHiddenLayer:
			w_temp  = np.random.uniform(low = lowWeight, high = highWeight, size = [neurons, before])
			wb_temp = np.random.uniform(low = lowWeight, high = highWeight, size = [1, neurons])
			w_middle.append(w_temp)
			wb_middle.append(wb_temp)
			before  = neurons 		# At the end, before is going to have the last number of neurons of the last layer

		# Weight initialization last layer  		
		w_output        = np.random.uniform(low = lowWeight, high = highWeight, size = [size_y, before])
		wb_output       = np.random.uniform(low = lowWeight, high = highWeight, size = [1, size_y])

		self.w_middle  = w_middle
		self.wb_middle = wb_middle
		self.w_output  = w_output
		self.wb_output = wb_output

	def sigmoid(self, x):		
		
		return (1 /(1+ np.exp(-x)))
		# if x < 0:
		# 	return 1 - 1/(1 + np.exp(x))
		# else:
		# 	return 1 /(1+ np.exp(-x))


		# print('x: ', x)
		# out = []
		# for i in range(len(x)):
		# 	aux = 1 / (1 + math.exp(-x[i]))			
		# 	out.append(aux)
		# return np.array(out)

		# def sigmoid(gamma):
  # if gamma < 0:
  #   return 1 - 1/(1 + math.exp(gamma))
  # else:
  #   return 1/(1 + math.exp(-gamma))

	def forward(self, x, y, w_middle, w_output, wb_middle, wb_output):
		# Middle Layer
		y_layer = []
		input = np.array(x)
		idx = 0
		for w_neurons, w_bias in zip(w_middle, wb_middle):			
			input_with_bias = np.append(input, 1)		
			w_neurons       = np.array(w_neurons)		
			w_bias          = np.array(w_bias)		
			w_current       = np.concatenate((w_neurons, w_bias.T), axis = 1)			
			output          = self.sigmoid(np.dot(w_current, input_with_bias))
			y_layer.append(output)		
			input           = output
			idx = idx + 1

				
		# Output Layer	
		y_layer_bias = np.append(input, 1)  # bias initialization	
		w_current    = np.concatenate((w_output, wb_output.T), axis = 1)		
		y_net 		 = self.sigmoid(np.dot(w_current, y_layer_bias))

		# Calculating error
		error        = (np.sum((y - y_net)**2))/len(y_net)
		
		return y_layer, y_net, error

	
	def fit_by_mini_batch(self, x_mini, y_mini, learningRate, alphaMomentum):
		self.learningRate          = learningRate
		self.alphaMomentum         = alphaMomentum		

		w_middle  = self.w_middle
		wb_middle = self.wb_middle
		w_output  = self.w_output
		wb_output = self.wb_output

		delta_w_output_old  = np.zeros_like(w_output)
		delta_o_old         = np.zeros_like(wb_output[0])	
		delta_w_current_old = []
		delta_o_hidden_old  = []

		batch_delta_w_output         = []
		batch_delta_o                = []
		batch_delta_w_middle_current = [None] * (len(w_middle))	
		batch_delta_o_hidden         = [None] * (len(w_middle))

		for w, wb in zip(w_middle, wb_middle):
			delta_w_current_old.append(np.zeros_like(w))
			delta_o_hidden_old.append(np.zeros_like(wb[0]))

		for i in range(len(x_mini)):			
			y_layer, y_net, error = self.forward(x_mini[i], y_mini[i], w_middle, w_output, wb_middle, wb_output)
			delta_o            = np.array(-(y_mini[i] - y_net) * y_net * (np.ones(len(y_net)) - y_net)) + (self.alphaMomentum * delta_o_old)
			delta_o_old        = delta_o
			idx2               = len(y_layer) - 1
			delta_w_output     = np.array((delta_o[np.newaxis]).T * y_layer[idx2]) + (self.alphaMomentum * delta_w_output_old)
			delta_w_output_old = delta_w_output
			old_w_output       = w_output

			batch_delta_w_output.append(delta_w_output)			
			batch_delta_o.append(delta_o)		
			
			for idx in range(len(y_layer) - 1, -1, -1):				
				if idx == 0:					
					input     = x_mini[i]					
					if len(y_layer) == 1:	# Only one element						
						layer = y_layer
					else:							
						layer     = y_layer[idx]								
					idx_layer = idx

				else:										
					layer     = y_layer[idx]						
					idx_layer = idx 
					input     = y_layer[idx-1]					

				temp                     = np.ones(len(layer)) - layer
				delta_o_hidden           = (np.dot(delta_o, old_w_output) * layer * temp) + (self.alphaMomentum * delta_o_hidden_old[idx])
				delta_o_hidden_old[idx]  = delta_o_hidden
							
				if len(y_layer) == 1:
					delta_w_middle_current         = (delta_o_hidden.T * input) + (self.alphaMomentum * delta_w_current_old[idx])
				else:
					delta_w_middle_current         = (delta_o_hidden[np.newaxis].T * input) + (self.alphaMomentum * delta_w_current_old[idx])


				delta_w_current_old[idx]          = delta_w_middle_current
				old_w_output                      = w_middle[idx]
				try:
					batch_delta_w_middle_current[idx] = batch_delta_w_middle_current[idx] + np.array(delta_w_middle_current)					
				except: 
					batch_delta_w_middle_current[idx] = np.array(delta_w_middle_current)
				batch_delta_o_hidden[idx]         = np.array(delta_o_hidden)				
				delta_o                           = delta_o_hidden

		
		batch_delta_w_output         = np.array(batch_delta_w_output)
		batch_delta_o                = np.array(batch_delta_o)
		geral_delta_w_output         = np.sum(batch_delta_w_output, axis = 0)				
		geral_delta_o                = np.sum(batch_delta_o, axis = 0)		
		geral_delta_w_middle_current = batch_delta_w_middle_current
		geral_delta_o_hidden         = batch_delta_o_hidden

		
		len_batch = len(x_mini)		
		w_output  = w_output - ((self.learningRate/ len_batch) * geral_delta_w_output)
		wb_output = wb_output - ((self.learningRate/ len_batch) * geral_delta_o)
		idx       = 0
		for current_w_middle, current_o_hidden in zip(geral_delta_w_middle_current, geral_delta_o_hidden):				
			w_middle[idx]  = w_middle[idx] - ((self.learningRate / len_batch)* current_w_middle)
			wb_middle[idx] = wb_middle[idx] - ((self.learningRate / len_batch) * current_o_hidden)
			idx = idx +1

		self.w_output = w_output
		self.w_middle = w_middle
		self.wb_middle = wb_middle
		self.wb_output = wb_output
		
		return w_middle, w_output, wb_middle, wb_output	