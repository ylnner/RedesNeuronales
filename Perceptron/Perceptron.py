import numpy as np

# x = np.array([[+1, -1, -1, +1, -1, -1, -1, -1, +1, -1, -1, -1, +1, +1, +1, -1, +1, -1, -1, -1, +1, +1, -1, -1, -1 ,+1],
# 		 	  [+1, -1, -1, +1, -1, +1, -1, -1, +1, -1, -1, -1, +1, +1, +1, -1, +1, -1, -1, -1, +1, +1, -1, -1, -1, +1],
# 		 	  [+1, -1, -1, +1, -1, -1, -1, -1, -1, -1, -1, -1, +1, +1, +1, -1, +1, -1, -1, -1, +1, +1, -1, +1, -1, +1],
# 		 	  [+1, -1, -1, +1, -1, -1, -1, -1, +1, -1, -1, -1, +1, +1, -1, -1, +1, -1, -1, -1, +1, -1, -1, +1, -1, +1],
# 		 	  [+1, -1, -1, +1, -1, -1, -1, +1, +1, +1, -1, -1, +1, +1, -1, -1, +1, -1, -1, -1, +1, +1, -1, -1, -1, +1], 
# 		 	  [+1, -1, -1, +1, -1, -1, -1, +1, +1, +1, -1, -1, +1, +1, +1, -1, +1, -1, +1, -1, +1, +1, -1, -1, -1, +1]])
# w = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])


x = np.array([[+1, +1, -1, -1, -1, +1, +1, -1, -1, -1, +1, -1, +1, +1, +1, -1, -1, -1, +1, -1, -1, -1, -1, +1, -1, -1],
			 [+1, +1, +1, -1, -1, +1, +1, -1, -1, -1, +1, -1, +1, +1, +1, -1, -1, -1, +1, -1, -1, -1, -1, +1, -1, -1],
			 [+1, +1, -1, -1, -1, +1, +1, -1, -1, -1, +1, -1, +1, +1, +1, -1, -1, +1, +1, +1, -1, -1, -1, +1, -1, -1],
			 [+1, +1, -1, +1, -1, +1, +1, -1, -1, -1, +1, -1, +1, +1, +1, -1, -1, -1, +1, -1, -1, +1, -1, +1, -1, +1],
			 [+1, +1, -1, +1, +1, +1, +1, -1, -1, -1, +1, -1, +1, +1, +1, -1, -1, -1, +1, -1, -1, -1, -1, +1, -1, -1],
			 [+1, +1, -1, -1, -1, +1, +1, -1, -1, -1, +1, -1, +1, +1, +1, -1, -1, +1, +1, -1, -1, -1, -1, +1, -1, +1]])


w = np.random.random_sample(26)


print('len(x): ', len(x))
print('len(x[0]): ', len(x[0]))
print('len(w): ', len(w))
print('w: ', w)

# output = [-1, -1, -1, -1, -1, -1]
output = [1, 1, 1, 1, 1, 1]


number_train = len(x)

isError = True
i = 0

number_iter = 10
learning_rate = 0.1

for i in range(number_iter):
	# isOK = True	
	for j in range(len(x)):

		ynet = np.sum(x[j] * w)
		# print('x[j] * w: ', x[j] * w)
		if ynet >= 0:
			ynet = 1
		else:
			ynet = -1

		print('ynet: ', ynet)
		print('output[j]: ', output[j])
		if ynet != output[j]:
			isOK    = False
			error   = output[j] - ynet
			delta_w = learning_rate * x[j] * error
			w       = w + delta_w
			# w = w + (learning_rate * (output[j] - ynet) * x[j])
			print('new_w: ', w)
			print('j: ', j)
	
	# if isOK == True:
	# 	break

print('final_w: ', w)