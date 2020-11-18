# Clasificador para Rpi Fully-Connected

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(10, activation='sigmoid', input_shape=(784,)))
model.add(Dense(10, activation='softmax'))

model.load_weights('fully_connected.h5')

#%% Cargamos los datos y realizamos las transformaciones

from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_test = x_test.reshape(10000, 784)
x_test = x_test.astype('float32')
x_test /= 255
print(x_test.shape)

#%% Realizamos predicciones

import numpy as np
import matplotlib.pyplot as plt

plt.imshow(x_test[12].reshape(28,28), cmap=plt.cm.binary)
dato_prueba = x_test[12].reshape(1, -1)
print(dato_prueba.shape)
print(y_test[12])

print(np.argmax(model.predict(dato_prueba)))


