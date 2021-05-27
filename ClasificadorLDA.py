from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import cv2
import Constants
import numpy as np

# devuelve la precision global de los valores predecidos
# recibe los valores correctos y los valores predecidos
def getPrecision(correct_vals, pred_vals):
    pred_vals = np.array(pred_vals)
    correct_vals = np.array(correct_vals)
    matches = pred_vals == correct_vals
    return round((matches.sum() / len(matches)) * 100 , 2)

class ClasificadorLDA:
    #constructor
    def __init__(self):
        self.lda = LinearDiscriminantAnalysis() # lda
        win_size = (Constants.HOG_WIN_SIZE, Constants.HOG_WIN_SIZE) # tamaño de la imagen
        block_size = (Constants.HOG_BLOCK_SIZE, Constants.HOG_BLOCK_SIZE) # tamaño del bloque
        block_stride = (Constants.HOG_BLOCK_STRIDE, Constants.HOG_BLOCK_STRIDE) # tamaño de desplazamiento entre los bloques
        cell_size = (Constants.HOG_CELL_SIZE, Constants.HOG_CELL_SIZE) # tamaño de las celdas
        self.hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, Constants.HOG_N_BINS) # descriptor hog

    # devuelve el vector de caracteristicas de la imagen
    # recibe la imagen redimensionada
    def applyHOG(self, img):
        hog_value = self.hog.compute(img).flatten()
        hog_value = np.nan_to_num(np.array(hog_value))
        return hog_value

    # devuelve los vectores de caracteristicas de varias imagenes
    # recibe imgs, una array con las imagenes
    def applyHOGAll(self, imgs):
        hog_values = []
        for img in imgs:
            hog_value = self.applyHOG(img)
            hog_values.append(hog_value)
        return np.array(hog_values)

    # entrena el clasificador con las imagenes de entenamiento
    # recibe las imagenes y sus valores de clasificacion
    def train(self, imgs, answers):
        self.lda.fit(self.applyHOGAll(imgs), answers)

    # devuelve la prediccion
    # TODO: investigar porque falla solo si da tiempo
    def predict(self, img):
        hog_value = self.applyHOG(img)
        return self.lda.predict(hog_value)

    # devuelve las predicciones y la precision global de las predicciones
    # recibe las imagenes y sus valores de clasificacion
    def predictAll(self, imgs, answers):
        hog_values = self.applyHOGAll(imgs)
        predictions = self.lda.predict(hog_values)
        return predictions, getPrecision(answers, predictions)

    # devuelve los valores reducidos de las imagenes
    # recibe las imagenes y sus valores de clasificacion
    def transform(self, imgs, answers):
        return self.lda.transform(imgs, answers)