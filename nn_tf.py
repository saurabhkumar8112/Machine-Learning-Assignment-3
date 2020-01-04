import tensorflow as tf
import pandas as pd 
import numpy as np 

mnist = tf.keras.datasets.mnist
mnist_test=pd.read_csv("x_test.csv")
mnist_test=np.array(mnist_test)
mnist_test=mnist_test[:,1:]
mnist_test=mnist_test / 255.0
mnist_test=tf.reshape(mnist_test,[-1,28,28,1])
print(mnist_test.shape)

(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
print(x_train.shape)
print(type(x_train))
x_train=np.concatenate((x_train,x_test),0)
y_train=np.concatenate((y_train,y_test),0)
print(x_train.shape)
print(type(x_train))


model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(55, activation=tf.nn.relu),
  tf.keras.layers.Dense(55, activation=tf.nn.relu),
  tf.keras.layers.Dense(555, activation=tf.nn.relu),
  tf.keras.layers.Dense(50, activation=tf.nn.relu),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=10)
data=model.predict_on_batch(mnist_test)
data=np.array(data)
for i in range(data.shape[0]):
	print(np.argmax(data[i]))
