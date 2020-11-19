# Clasificador para Rpi Fully-Connected

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from PIL import Image

model = Sequential()
model.add(Dense(10, activation='sigmoid', input_shape=(784,)))
model.add(Dense(10, activation='softmax'))

model.load_weights('fully_connected.h5')

#%% Predecimos una foto

img = Image.open('nueve.png').convert('L')
img_array = np.array(img)
img_array = 255-img_array
img_array = img_array.reshape(1, -1)

print("PREDICCIÓN:", np.argmax(model.predict(img_array)))
