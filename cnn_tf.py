'''Trains a simple convnet on the MNIST dataset.
Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''
from __future__ import print_function
import keras
import pandas as pd
import numpy as np 
import tensorflow as tf
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, MaxPooling3D, AveragePooling2D
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator

batch_size = 128
num_classes = 10
epochs = 1

mnist_test=pd.read_csv("x_test.csv")
mnist_test=np.array(mnist_test)
mnist_test=mnist_test[:,1:]
mnist_test=mnist_test / 255.0
mnist_test=tf.reshape(mnist_test,[-1,28,28,1])
print(mnist_test.shape)
# input image dimensions
img_rows, img_cols = 28, 28

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

shift=0.2
datagen1 = ImageDataGenerator(width_shift_range=shift, height_shift_range=shift)
datagen2 = ImageDataGenerator(rotation_range=20)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
#x_train=np.concatenate((x_train,x_test),0)
dummy1=x_train
dummy2=x_train
datagen1.fit(dummy1)
datagen2.fit(dummy2)
x_train=np.concatenate((x_train,dummy1),0)
x_train=np.concatenate((x_train,dummy2),0)

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
y_train=np.concatenate((y_train,y_train,y_train),0)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(AveragePooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
data=model.predict_on_batch(mnist_test)
data=np.array(data)
for i in range(data.shape[0]):
  print(np.argmax(data[i]))
