#importamos las librerías
from keras.datasets import mnist
import matplotlib.pyplot as plt

# obtenemos los datos para train y test 
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.ndim) # dimensión 3
print(x_train.shape) # forma de los datos (60000,28,28)
print(len(y_train)) # cantidad de datos de entrenamiento
print(len(y_test)) # cantidad de datos de testeo

#%% visualizamos los datos, mostramos la matriz de uno de los
# datos

plt.imshow(x_train[2], cmap=plt.cm.binary)
print(y_train[2]) 

#%% Escalamos los datos a 0 y 1

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

x_train /= 255 # dividimos por el mÃ¡ximo
x_test /= 255

#%% tranformamos los datos para una red densa

x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)

print(x_train.shape)
print(x_test.shape)

#%% pasamos las etiquetas a forma vectorial

from tensorflow.keras.utils import to_categorical

y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

#%% Definimos la red neuronal

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(10, activation='sigmoid', input_shape=(784,)))
model.add(Dense(10, activation='softmax'))

model.summary()

#%% Entrenamiento

batch_size = 32
num_classes = 10
epochs=20

model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])

history = model.fit(x_train, y_train,
          validation_split=0.33,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1
          )

#%% Evaluación

test_loss, test_acc = model.evaluate(x_test, y_test)

print('Test loss:', test_loss)
print('Test accuracy:', test_acc)

#%% Predicción

import numpy as np

plt.imshow(x_test[12].reshape(28,28), cmap=plt.cm.binary)
dato_prueba = x_test[12].reshape(1, -1)
print(dato_prueba.shape)
print(y_test[12])

print(np.argmax(model.predict(dato_prueba))) # predicción


#%% Almacenamos el modelo

model.save_weights("fully_connected.h5")

#%% visualizaciÃ³n del entrenamiento

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Accuracy del modelo')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.grid(True)
plt.legend(['train', 'test'], loc='upper left')
plt.ylim([0.3, 1])


plt.figure(2)
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('loss del modelo')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.grid(True)
plt.legend(['train', 'test'], loc='upper left')

#%% matriz de confunsiÃ³n

import numpy as np
import itertools

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('Observación')
    plt.xlabel('Predicción')
    
from sklearn.metrics import confusion_matrix

# Predecimos las clases para los datos de test
Y_pred = model.predict(x_test)
# Convertimos las predicciones en one hot encoding
Y_pred_classes = np.argmax(Y_pred, axis = 1) 
# Convertimos los datos de test en one hot encoding
Y_true = np.argmax(y_test, axis = 1) 
# Computamos la matriz de confusiÃ³n
confusion_mtx = confusion_matrix(Y_true, Y_pred_classes) 
# Mostramos los resultados
plot_confusion_matrix(confusion_mtx, classes = range(10))