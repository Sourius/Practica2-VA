# -*- coding: utf-8 -*-
import cv2
import numpy as np
from matplotlib import pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import os

def entradasToMap(file_path):
  f = open(file_path, mode='r', encoding='utf-8')
  map_entradas = {}
  # leer lineas
  for line in f:
    # obtener tipo
    ficheros = line.strip().split(";")
    tipo = ficheros[0]
    map_entradas[tipo] = ficheros[1:]
  f.close()
  return map_entradas
  
# devuelve las rutas todas las imagenes de entrenamiento que hay dentro del directorio y los subdirectorios
# devuelve los valores de classificacion de todas las imagenes de entrenamiento 
# recibe la ruta del directorio
def getSignalImagePaths(dir_path):
  img_signal_paths = []
  img_signal_types = []
  aux = entradasToMap("entradas.txt")
  # https://stackoverflow.com/questions/2909975/python-list-directory-subdirectory-and-files
  for path, subdirs, files in os.walk(dir_path):
    for name in files:
      subdir = os.path.basename(path)
      if any(subdir in value for value in aux.values()):
        img_signal_paths.append(os.path.join(path,name))
        img_signal_types.append(1)
      else:
        img_signal_paths.append(os.path.join(path,name))
        img_signal_types.append(4)
  return img_signal_paths, img_signal_types

# muestra imagen con titulo
def printImage(title, img):
  plt.title(title)
  plt.imshow(img)
  plt.show()

# devuelve la imagen equalizada recortada
# recibe la ruta de la imagen
def getProcessedImage(img_path):
  crop_dim = (30,30)
  original = cv2.imread(img_path,0)
  img_eq = cv2.equalizeHist(original)
  return cv2.resize(img_eq, (30,30))

# calcula y devuelve el vector de caracteristicas hog de la imagen
# recibe la imagen equalizada recortada en 30x30
def applyHOG(img):
  winSize = (32,32)
  blockSize = (16,16)
  blockStride = (8,8)
  cellSize = (4,4)
  nbins = 9
  hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins)
  return hog.compute(img)

# reduccion de la dimension mediante LDA
# devuelve el analizador LDA
def trainLDA(data, answers):
  lda = LinearDiscriminantAnalysis()
  lda.fit(data, answers)
  return lda

# devuelve la classificacion de la imagen
# recibe analizador lda
# recibe la ruta de la imagen
def predictImage(lda, img_path):
  img = getProcessedImage(img_path)
  hog_value_of_img = applyHOG(img).flatten()
  hog_value_of_img = np.array(hog_value_of_img) 
  # nan en hog values
  hog_value_of_img = np.nan_to_num(hog_value_of_img)
  return lda.predict([hog_value_of_img])[0]

# devuelve la classificacion de las imagenes ubicadas dentro del directorio
# recibe el analizador lda
# recibe la ruta del directorio donde estan las imagenes a classificar
def predictImageList(lda, img_dir):
  predictions = []
  for img_path in img_dir:
    predictions.append(predictImage(lda, img_path))
  return predictions

# muestra las imagenes y su classificacion
# recibe las rutas de las imagenes
# recibe los valores de las predicciones
def showImagePredictions(img_paths, predictions):
  for i in range(len(img_paths)):
    printImage(predictions[i], getProcessedImage(img_paths[i]))

# entrenamiento
img_paths, img_correct_types = getSignalImagePaths('/content/drive/MyDrive/train_recortadas_reducido')
hog_values = []
for img_path in img_paths:
  hog_value_of_img = applyHOG(getProcessedImage(img_path)).flatten()
  hog_values.append(hog_value_of_img)
hog_values = np.array(hog_values) # a veces hay nan en hog_values
hog_values = np.nan_to_num(hog_values)

lda = trainLDA(hog_values, img_correct_types)
lda_values = lda.transform(hog_values)
plt.hist(lda_values)
plt.show()

#lda = trainLDA(lda_values, img_correct_types)
#print(lda.predict(lda_values))

# classificacion
preds = predictImageList(lda, img_paths)
print()
showImagePredictions(img_paths, preds)