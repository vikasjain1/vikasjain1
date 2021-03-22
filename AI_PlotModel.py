import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import models
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import plot_model

from tensorflow.keras import layers
from tensorflow.keras.layers import Dense

def get_model():
    model = models.Sequential()
    model.add(layers.Conv2D(16,(3,3),activation='relu',input_shape=(135,240,3),padding = 'same'))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Conv2D(32,(3,3),activation='relu',padding = 'same'))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Conv2D(64,(3,3),activation='relu',padding = 'same'))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Conv2D(64,(3,3),activation='relu',padding = 'same'))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Conv2D(128,(3,3),activation='relu',padding = 'same'))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(128,activation="relu"))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(27,activation="softmax"))

    return model

model = get_model()
print(model.summary())
plot_model(model, to_file='model.png') 