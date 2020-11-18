# Clasificador para Rpi basado en CNN

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten

model = Sequential()
model.add(Conv2D(32, (5, 5), activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (5, 5), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(10, activation='softmax'))
model.summary()

model.load_weights('CNN.h5')

#%% Cargamos los datos y realizamos las transformaciones

from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_test = x_test / 255
x_test = x_test.reshape(10000, 28, 28, 1)
x_test = x_test.astype('float32')

#%% Predicción

import numpy as np
import matplotlib.pyplot as plt


plt.imshow(x_test[12].reshape(28,28), cmap=plt.cm.binary)
print(y_test[12])

predicciones = model.predict(x_test[0:20]) # predicciones
print("Predicción:", np.argmax(predicciones[12]))
plt.show()


