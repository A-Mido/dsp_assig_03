from load import Load_Data as LD
import tensorflow as tf
from tensorflow import keras
import numpy as np
import time

# taking activation function type
AF = input("\nchoose activation function [relu - sigmoid]: ")
while(1): 
    if AF == 'relu':
        break
    elif AF == 'sigmoid':
        break
    else:
        AF = input('error choose one [relu or sigmoid]: ')

# taking number of filters in first convolution stage
NF = int(input("\nchoose number of filters in first conv layer [6 or 8]: "))
while(1): 
    if NF == 6:
        break
    elif NF == 8:
        break
    else:
        AF = int(input('error choose [6 or 8]: '))

# training times
TT = int(input("\nenter number of training times: "))

while(1): 
    if TT > 0:
        break
    else:
        TT = int(input('error enter number of training times: '))

# loading trainig data
train_images = LD.train_images_norm
train_classes = LD.train_lables

# loading testing data
test_images = LD.test_images_norm
test_classes = LD.test_lables

# expanding dimensions for further operations
train_images = tf.expand_dims(train_images, axis=-1)
test_images = tf.expand_dims(test_images, axis=-1)


lenet = keras.models.Sequential([
    # first conv layer with num of filters and activation function
    keras.layers.Conv2D(NF, kernel_size=5, strides=1,  activation=AF, input_shape=train_images[0].shape, padding='same'),
    # first sub sampling layer with strides and frequency
    keras.layers.AveragePooling2D(strides=2),
    # second conv layer with num of filters and activation function
    keras.layers.Conv2D(16, kernel_size=5, strides=1, activation=AF, padding='valid'),
    # second sub sampling layer with strides and frequency
    keras.layers.AveragePooling2D(strides=2),
     # flattening data
    keras.layers.Flatten(),
    #  120 full connections layer
    keras.layers.Dense(120, activation=AF),
    #  84 full connections layer
    keras.layers.Dense(84, activation=AF),
    # output layer with 10 output neurons
    keras.layers.Dense(10, activation='softmax') 
])

# compiling model
lenet.compile(optimizer='adam', loss=keras.losses.sparse_categorical_crossentropy, metrics=['accuracy'])

# training data
start_training_time = time.time()
lenet.fit(train_images, train_classes, epochs=TT)
print(f"Training time = {round(time.time() - start_training_time, 2)} seconds\n")

# testing model
start_testing_time = time.time()
acc = lenet.evaluate(test_images, test_classes)
print(f"Testing time = {round(time.time() - start_testing_time, 3)} seconds\n")


print(f"Accuracy of testing data = {round(acc[1],3)} %")