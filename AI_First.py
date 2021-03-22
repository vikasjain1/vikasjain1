
# https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/
# first neural network with keras tutorial

import tensorflow
import csv
import pandas
import matplotlib.pyplot as plt
import numpy as np

from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

print('\n1. ************************************************* Load the dataset')

dataset = pandas.read_csv('../Data/diabetes_data.csv')
print(dataset.head(5))

print('\n ********************** Split into input (X) and output (y) variables')
X = dataset.iloc[:,0:8]
train_x = np.asarray(X)
Y = dataset.iloc[:,8]
train_y = np.asarray(Y)

print('\n2. ********************************************DEFINE THE Keras MODEL')
# https://keras.io/api/layers/core_layers/ 
# Core Layers => Input object, Dense Layer, Acgtivaton Layer, Embedding Layer, Masking Layer, Lambda Layer'

print('\n Employing Sequential model; 3 Dense layers; 2 ReLU (i/p) and 1 Signoid (o/p) functions')      
model = Sequential()

# first hidden layer has 12 nodes and uses the relu activation function.
model.add(Dense(12, input_dim=8, activation='relu'))

# second hidden layer has 8 nodes and uses the relu activation function.
model.add(Dense(8, activation='relu'))

#o utput layer has one node and uses the sigmoid activation function.
model.add(Dense(1, activation='sigmoid'))

print('\n3. ****************************************** COMPILE THE Keras MODEL')
# Training a Network means finding the best set of weights to map inputs to outputs in our dataset
# => (a) Specify a Loss Function    -> to evaluate a set of weights
# => (b) Use an Optimizer           -> to search through different weights for the network
# (a) -> Binary Classification's Cross Entropy 
# (b) -> Gradient Descent's Adam version (optimization algorithm)

print('\n Employing Cross Entropy (Loss); Gradient Descent\'s Adam (Optimizer)')
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print('n4. ****************************** FIT/TRAIN THE Keras MODEL on dataset')

print('\n Employing Epochs=150; Batch=10 => Update wts after each batch of 10 samples/rows/ of data\n')
history = model.fit(X, Y, epochs=50, batch_size=10)

# list all data in history
print('\nHistory Keys %s' %(history.history.keys()))

print('\nPlot accuracy history')
plt.plot(history.history['accuracy'])
#plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'test'], loc='upper left')
plt.show()

print('\nPlot loss history')
plt.plot(history.history['loss'])
#plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'test'], loc='upper left')
plt.show()

print('\n5. ***************************************** EVALUATE THE Keras MODEL')

# Fit/TRAINing gives us an idea of how well we have modeled the dataset (e.g. train accuracy), 
# but no idea of how well the algorithm might perform on new data


# evaluate() returns a list with two values. 
# (1) - the loss of the model on the dataset 
# (2) - the accuracy of the model on the dataset. 
# We are only interested in reporting the accuracy, so may ignore the loss value.
loss, accuracy = model.evaluate(X, Y)
print('Loss: %.2f' % (loss*100))
print('Accuracy: %.2f' % (accuracy*100))

print('\n6. ********************** Make probability predictions with the model')
# Output layer is using Sigmoid activation function
# so the predictions will be a probability in the range between 0 and 1
predictions = model.predict(X)
rounded = [round(x[0]) for x in predictions]
print(rounded)

# Summarize the first 5 cases
for i in range(5):
	print('%s => %d (expected %d)' % (train_x[i], predictions[i], train_y[i]))
