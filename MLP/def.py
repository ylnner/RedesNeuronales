import numpy as np 
import math

x = np.array([[1, 0, 0, 0, 0, 0, 0, 0],
              [0, 1, 0, 0, 0, 0, 0, 0],
              [0, 0, 1, 0, 0, 0, 0, 0],
              [0, 0, 0, 1, 0, 0, 0, 0],
              [0, 0, 0, 0, 1, 0, 0, 0],
              [0, 0, 0, 0, 0, 1, 0, 0],
              [0, 0, 0, 0, 0, 0, 1, 0],
              [0, 0, 0, 0, 0, 0, 0, 1]])
            
y = np.array([[1, 0, 0, 0, 0, 0, 0, 0],
              [0, 1, 0, 0, 0, 0, 0, 0],
              [0, 0, 1, 0, 0, 0, 0, 0],
              [0, 0, 0, 1, 0, 0, 0, 0],
              [0, 0, 0, 0, 1, 0, 0, 0],
              [0, 0, 0, 0, 0, 1, 0, 0],
              [0, 0, 0, 0, 0, 0, 1, 0],
              [0, 0, 0, 0, 0, 0, 0, 1]])

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

def MLP(x, y, number_layers, number_neurons, w_input, w_output, w):
  y_temp = x[0]
  for i in range(number_layers):    
    if i == 0:  
      w_current              = w_input
      number_neurons_current = number_neurons
    elif i == number_layers - 1:      
      w_current              = w_output
      number_neurons_current = len(y)
    else:
      number_neurons_current = number_neurons
    

    new_y = []
    number_neurons_current
    for j in range(number_neurons_current):
      temp = np.sum(y_temp * w_current[j])      
      new_y.append(sigmoid(temp))

    y_temp = np.array(new_y)
    
    if i == number_layers -1:      
      error = (np.sum((y[0] - y_temp)**2))/2


  return error, y_temp
      
   

def backpropagation(x, y, number_layers, number_neurons, low_weight, high_weight, learning_rate = 0.2):
  print('llegue')
  error      = 1
  tolerancia = 0.5
  
  # Weight initialization first layer  
  w_input  = np.random.uniform(low = low_weight, high = high_weight, size = len(x))
  
  # Weight initialization last layer  
  w_output = np.random.uniform(low = low_weight, high = high_weight, size = len(y))

  # Weight initialization for middles layers
  w = []
  # We do not include the first and last layer, they are variable by the size of the input
  for i in range(1, number_layers - 2 ):  
    w_temp = np.random.uniform(low = low_weight, high = high_weight, size = [number_layers, number_neurons])
    w.append(w_temp)


  error, ynet = MLP(x, y, number_layers, number_neurons, w_input, w_output, w)
  print('error: ', error)  
  # Calculate delta w input    

  # Calculate delta w output
  print('w_old')
  print(w_output)
  delta_w_output = np.around((-1 * (y[0] - ynet) * ynet * (1 - ynet) * w_output), 10)
  w_output = w_output + (learning_rate * delta_w_output)
  print('delta_w_output: ')
  print(delta_w_output)
  print('w_new')
  print(w_output)


  print('y[0]')
  print(y[0])
  print('ynet')
  print(ynet)
  print('w_output')
  print(w_output)

  # while error > tolerancia:
  #   w_input, w_output, w, error, ynet = MLP(x, y, number_layers, number_neurons, w_input, w_output, w)

  #   # Calculate delta w input    

  #   # Calculate delta w output
  #   delta_w_output = -(y - ynet) * ynet * (1 - ynet) * w_output
  #   print('delta_w_output: ')
  #   print(delta_w_output)


  #   # Calculate delta w

  



number_layers  = 2
number_neurons = 3
low_weight     = -0.1
high_weight    = 0.1
backpropagation(x, y, number_layers, number_neurons, low_weight, high_weight)

