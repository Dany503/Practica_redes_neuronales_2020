# -*- coding: utf-8 -*-

"""

Implementación de CNN clasificadora de señales de tráfico en RPI3.

Desarrollado por Eduardo Moscosio Navarro

Grado de Ingeniería Electrónica, Robótica y Mecatrónica (ETSI)

"""

# Cargamos librerías a usar
import keras

import cv2
import numpy as np
import copy

# Funciones de preprocesado:
def image_preproc(img, coef = None, width = None, height = None, inter = cv2.INTER_AREA):
    dim = (width,height)
    # RGB to Gray image conversion
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # resize the image
    img_prep = cv2.resize(gray, dim, interpolation = inter)
    # rescale the image
    img_prep.astype('float32') # Convierte a float32
    img_prep = img_prep/coef # Escalado
    # return the resized image
    return img_prep

# Funciones de resultados:
def signal_type(prediction):
  # Vector con los nombres de las señales:
  signal = np.array(["Velocidad máxima 20 Km/h", "Velocidad máxima 30 Km/h", "Velocidad máxima 50 Km/h", "Velocidad máxima 60 Km/h", "Velocidad máxima 70 Km/h",
            "Velocidad máxima 80 Km/h", "Fin de limitación de velocidad máxima 80 Km/h", "Velocidad máxima 100 Km/h", "Velocidad máxima 120 Km/h", "Adelantamiento prohibido",
            "Adelantamiento prohibido para camiones", "Intersección con prioridad", "Calzada con prioridad", "Ceda el paso", "STOP", "Circulación prohibida en ambos sentidos",
            "Prohibición de acceso a vehículos destinados a transporte de mercancías", "Entrada prohibida", "Otros peligros", "Curva peligrosa hacia la izquierda",
            "Curva peligrosa hacia la derecha", "Curvas peligrosas hacia la izquierda", "Perfil irregular", "Pavimento deslizante", "Estrechamiento de calzada por la derecha",
            "Obras", "Proximidad de semáforo", "Peatones", "Niños", "Ciclistas", "Pavimento deslizante por hielo o nieve", "Paso de animales en libertad", "Fin de prohibiciones",
            "Sentido obligatorio derecha", "Sentido obligatorio izquierda", "Sentido obligatorio recto", "Recto y derecha únicas direcciones permitidas",
            "Recto e izquierda únicas direcciones permitidas", "Paso obligatorio derecha", "Paso obligatorio izquierda", "Intersección de sentido giratorio-obligatorio",
            "Fin de prohibición de adelantamiento", "Fin de prohibición de adelantamiento para camiones"])
  # Se asocia el número obtenido en la predicción con el nombre de la señal:
  if len(prediction) > 0:
    for k in range(0,len(prediction)):
      if prediction[k] < 10:
        print(str(prediction[k]) + "   ==>  " + str(signal[prediction[k]]))
      else:
        print(str(prediction[k]) + "  ==>  " + str(signal[prediction[k]]))
  else:
    print(str(prediction) + "  ==>  " + str(signal[prediciton]))
  

# Carga el modelo con los pesos de la red:
from keras import layers
from keras import models

# cargar json y crear el modelo
# Nombre del archivo:
nombre = "/home/pi/Implementacion_Red_Neuronal/red_neuronal" # PONER DIRECCIÓN EN DISCO DE RED NEURONAL ENTRE ""

json_file = open(nombre + ".json", 'r')
model_json = json_file.read()
json_file.close()
model = models.model_from_json(model_json)

# cargar pesos al nuevo modelo
model.load_weights(nombre + ".h5")
print("Cargado modelo desde disco.")
model.summary() # Para ver como es la red


# Prueba con una imagen:
from time import time
from picamera import PiCamera

# Define la camara:
camera = PiCamera()
camera.resolution = (640,480)
camera.rotation = 180
camera.start_preview(fullscreen=False, window=(30,30,640,480)) 

print('Prueba de cámara (PULSAR INTRO)')
okay = input() # SE PULSA INTRO, ES ÚNICAMENTE PARA INICIAR LA CÁMARA POR PRIMERA VEZ
# Le llega una imagen:
output = np.empty((480, 640, 3), dtype=np.uint8)
camera.capture(output, 'rgb')
print('Prueba realizada!')

elapsed = []
history_pred = []
pred_act = []

while(1):
    # Espera a que llegue una imagen:
    print('Esperando imagen (PULSAR INTRO CUANDO SE DESEE CAPTURAR IMAGEN)')
    okay = input() ## SE PULSA INTRO, ES ÚNICAMENTE PARA INICIAR LA CÁMARA PsOR PRIMERA VEZ
    # Le llega una imagen:
    output = np.empty((480, 640, 3), dtype=np.uint8)
    camera.capture(output, 'rgb')
    print('Imagen capturada')
    signal = copy.copy(output)
    
    # Empieza el proceso:
    start_time = time() # Tiempo de ejecución comienza

    # Preprocesado:
    ancho = 64
    alto = 64
    signal_prep = image_preproc(signal, coef = 255, width = ancho, height = alto)
    test = signal_prep.reshape([-1,ancho, alto,1])
    
    # Clasificación:
    predictions = model.predict(test, batch_size=1, verbose=0) # Obtiene los 43 porcentajes para la imagen
    pred_max = np.argmax(predictions, axis=-1) # Se queda con la que tiene mayor porcentaje

    # Termina el proceso:
    elapsed_time = time() - start_time # Tiempo de ejecución termina
    print("Tiempo empleado: %.10f seconds." % elapsed_time) # Imprime el tiempo que ha tardado

    # Imprime la clase predicha y la imagen original:
    print("Las predicciones son: ", predictions)
    print("La señal predicha es de la clase: ")
    pred_act.append(pred_max)
    signal_type(pred_act)
    pred_act = []
    
    print('¿Clasificacion correcta? (y/n)')
    select1 = input()
    if select1=="y":
        elapsed.append(elapsed_time)
        numero_imagenes = len(elapsed)
        history_pred.append(pred_max)

    
    print('¿Hacer otra clasificacion? (y/n)')
    select = input()
    if select=="n":
        break

# Imprime la clase predicha y la imagen original:
try:
    print("Numero de imagenes clasificadas: ", numero_imagenes)
    print("Tiempo medio que tarda en segundos: ", sum(elapsed)/numero_imagenes)
    print("Historial de predicciones:")
    signal_type(history_pred)
    camera.stop_preview()
    camera.close()
except:
    print("niguna imagen se ha clasificado correctamente")
# Cierra la vista previa y libera la camara:
    camera.stop_preview()
    camera.close()
