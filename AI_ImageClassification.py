#https://keras.io/examples/vision/image_classification_from_scratch/

#https://colab.research.google.com/github/keras-team/keras-io/blob/master/examples/vision/ipynb/image_classification_from_scratch.ipynb#scrollTo=bbXOxR9gftQd


import tensorflow as tf
from tensorflow import keras

from tensorflow.keras import layers
import matplotlib.pyplot as plt

import AI_Utils as au

print("\n0.\tTensorFlow version...")
print(tf.version.VERSION)

print("\n1.\tDelete corrupt files...")
au.deleteCorruptImages()

print("\n2.\tGenerate a DataSet...")
image_size = (180, 180)
batch_size = 32

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "Pets",
    validation_split=0.2,
    subset="training",
    seed=1337,
    image_size=image_size,
    batch_size=batch_size,
)


val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "Pets",
    validation_split=0.2,
    subset="validation",
    seed=1337,
    image_size=image_size,
    batch_size=batch_size,
)

print("\n3.\tVisualize the data...")
#Here are first 9 images in the training dataset. 
#As you can see, label 1 is "dog" and label 0 is "cat".
plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(int(labels[i]))
        plt.axis("off")
       
print("\n4.\tUsing image data augmentation...")
layer_random_flip= layers.experimental.preprocessing.RandomFlip(mode="horizontal")
layer_random_rotation = layers.experimental.preprocessing.RandomRotation(0.2)

model_seq = keras.Sequential([layer_random_flip, layer_random_flip,])

print("\n5.\tVisualize the augmented data...")
plt.figure(figsize=(10, 10))
for images, _ in train_ds.take(1):
    for i in range(9):
        augmented_images = model_seq(images)
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(augmented_images[0].numpy().astype("uint8"))
        plt.axis("off")

print("\n6.\tStandardizing the data (OPT#1 - Make it part of the model)...")

model_seq.add(layers.Dense(units=4,input_shape=(1,2,3,))) #hidden layer 1 with input      
model_seq.add(layers.Dense(units=4)) #hidden layer 2
model_seq.add(layers.Dense(units=1)) #output layer  

inputs = keras.Input(shape=(1,2,3,))
x = model_seq(inputs)
x = layers.experimental.preprocessing.Rescaling(1./255)(x)

print("\n7.\tMake model...")
#model = au.make_model(input_shape=image_size + (3,), num_classes=2)

print("\n8.\tPlot model...")
#tf.keras.utils.plot_model(model)
 
tf.keras.utils.plot_model(
    model_seq,
    to_file='model.png',
    show_shapes=False,
    show_layer_names=True,
    rankdir='TB'
)


