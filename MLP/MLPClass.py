class MLPClass
	def __init__(self, lowWeight, highWeight, neuronsOnHiddenLayer, x_global, y_global):
		self.lowWeight             = lowWeight
		self.highWeight            = highWeight
		self.neuronsOnHiddenLayer  = neuronsOnHiddenLayer

		#Initialization of weights
		w_middle  = []
		wb_middle = []
		before    = len(x_global[0])
		for neurons in neuronsOnHiddenLayer:
			w_temp  = np.random.uniform(low = lowWeight, high = highWeight, size = [neurons, before])
			wb_temp = np.random.uniform(low = lowWeight, high = highWeight, size = [1, neurons])
			w_middle.append(w_temp)
			wb_middle.append(wb_temp)
			before  = neurons 		# At the end, before is going to have the last number of neurons of the last layer

		# Weight initialization last layer  
		w_output        = np.random.uniform(low = lowWeight, high = highWeight, size = [len(y_global[0]), before])
		wb_output       = np.random.uniform(low = lowWeight, high = highWeight, size = [1, len(y_global[0])])

		self.w_middle  = w_middle
		self.wb_middle = wb_middle
		self.w_output  = w_output
		self.wb_output = wb_output


	
	def fit(self, x_global, y_global, maxNumberOfIterations, learningRate, alphaMomentum, tol1, tol2)
		self.maxNumberOfIterations = maxNumberOfIterations
		self.learningRate          = learningRate
		self.alphaMomentum         = alphaMomentum
		self.tol1                  = tol1
		self.tol2                  = tol2

		w_middle  = self.w_middle
		wb_middle = self.wb_middle
		w_output  = self.w_output
		wb_output = self.wb_output

		achieved = False
		delta_w_output_old = np.zeros_like(w_output)
		delta_o_old        = np.zeros_like(wb_output[0])
		print('init delta_w_output_old')
		print(delta_w_output_old)

		delta_w_current_old = []
		delta_o_hidden_old  = []

		for w, wb in zip(w_middle, wb_middle):
			delta_w_current_old.append(np.zeros_like(w)) 
			delta_o_hidden_old.append(np.zeros_like(wb[0]))


		for ni in range(maxNumberOfIterations):
			y_net_best = []
			for i in range(len(x_global)):
				error = 1
				while error >= tol1:
					
					y_layer, y_net, error = forward(x_global[i], y_global[i], w_middle, w_output, wb_middle, wb_output)
									
					# Calculate delta w output
					print('delta_o_old')
					print(delta_o_old)
					delta_o            = np.array(-(y_global[i] - y_net) * y_net * (np.ones(len(y_net)) - y_net)) + (alphaMomentum * delta_o_old)										
					delta_o_old        = delta_o
					idx2               = len(y_layer) - 1				
					delta_w_output     = np.array((delta_o[np.newaxis]).T * y_layer[idx2]) + (alphaMomentum * delta_w_output_old)
					delta_w_output_old = delta_w_output
					old_w_output       = w_output				
					w_output           = w_output - (learningRate * delta_w_output)				
					wb_output          = wb_output - (learningRate * delta_o)
					
					
					# Calculate delta w middle
					for idx in range(len(y_layer) - 2, 0, -1):					
						if idx == 0:
							input     = x_global[i]
							layer     = y_layer[idx]
							idx_layer = idx
						else:						
							layer     = y_layer[idx + 1]
							idx_layer = idx + 1
							input     = y_layer[idx]
									
						temp                           = np.ones(len(layer)) - layer				
						delta_o_hidden                 = (np.dot(delta_o, old_w_output) * layer * temp) + (alphaMomentum * delta_o_hidden_old[idx_layer])
						delta_o_hidden_old[idx_layer]  = delta_o_hidden
						
						delta_w_middle_current         = np.array(delta_o_hidden[np.newaxis].T * input) + (alphaMomentum * delta_w_current_old[idx_layer])
						delta_w_current_old[idx_layer] = delta_w_middle_current

						old_w_output             = w_middle[idx+1]					
						w_middle[idx+1]          = w_middle[idx+1] - (learningRate * delta_w_middle_current)
						wb_middle[idx+1]         = wb_middle[idx+1] - (learningRate * delta_o_hidden)
										
				y_net_best.append(y_net)
								
			mse = np.sum(np.subtract(np.array(y_global), np.array(y_net_best)) ** 2)		
			
			if mse < tol2:
				achieved = True
				print('MSE was achieved: ', mse)
				print(np.array(y_net_best))			
				break
		
		if achieved == False:
			print('MSE was not achieved: ', mse)
			print(np.array(y_net_best))

		return w_middle, wb_middle, w_output, wb_output