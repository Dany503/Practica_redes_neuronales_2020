# Cargamos los datos

from keras.datasets import mnist
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train  = x_train / 255 # escalamos
x_test = x_test / 255

x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)

y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

#%% Definimos la red CNN

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

#%% Entrenamiento

batch_size = 32
num_classes = 10
epochs=10

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
print(y_test[12])

predicciones = model.predict(x_test) # predicciones
print("Predicción:", np.argmax(predicciones[12]))

#%% Almacenamos el modelo

model.save_weights("CNN.h5")

#%% visualización del entrenamiento

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

