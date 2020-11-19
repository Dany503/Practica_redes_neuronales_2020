# Clasificador para Rpi basado en CNN

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
import numpy as np
from PIL import Image

model = Sequential()
model.add(Conv2D(32, (5, 5), activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (5, 5), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(10, activation='softmax'))
model.summary()

model.load_weights('CNN.h5')

#%% Predecimos una foto

img = Image.open('nueve.png').convert('L')
img_array = np.array(img)
img_array = 255-img_array
img_array = img_array[np.newaxis, :,:, np.newaxis]

print("PREDICCIÓN:", np.argmax(model.predict(img_array)))
