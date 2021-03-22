import tensorflow as tf
import numpy as np 

from tensorflow import keras
from tensorflow.keras import layers

from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Activation, Dense 

model = Sequential() 
layer_1 = Dense(16, input_shape = (8,)) 

model.add(layer_1)

print(layer_1.input_shape) #(None, 8)
print(layer_1.output_shape) #(None, 16)
#print(layer_1.get_weights())

print('\nInput.................')
print(layer_1.input)
print('\nOutput................')
print(layer_1.output)

input = [ [1, 2], [3, 4] ]
kernel = [ [0.5, 0.75], [0.25, 0.5] ] 
result = np.dot(input, kernel)
print('\nResult.......................')
print(result)