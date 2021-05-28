import numpy as np
from skimage.feature import local_binary_pattern

# devuelve la precision global de los valores predecidos
# recibe los valores correctos y los valores predecidos
def getStats(correct_vals, pred_vals):
    pred_vals = np.array(pred_vals)
    correct_vals = np.array(correct_vals)
    matches = pred_vals == correct_vals
    # TODO: ampliar stats: precison, tpr, fpr, recall, f1score, matriz
    return round((matches.sum() / len(matches)) * 100, 2)

class Clasificador:
    def __init__(self, clasificador, reductor, descriptor):
        self.clasificador = clasificador
        self.reductor = reductor
        self.descriptor = descriptor
        self.reductor = reductor

    # devuelve el vector de caracteristicas lbp de la imagen
    # recibe la imagen redimensionada
    def _getLBPEigenVectors(self, img):
        # TODO: averiguar segundo parametro
        return local_binary_pattern(img, 8, 4) # LBP

    # devuelve el vector de caracteristicas hog de la imagen
    # recibe la imagen redimensionada
    def _getHOGEigenVectors(self, img):
        return self.descriptor.compute(img) #HOG

    # devuelve el vector de caracteristicas de la imagen
    # recibe la imagen redimensionada
    def getEigenVectors(self, img):
        pass

    # devuelve los vectores de caracteristicas de varias imagenes
    # recibe imgs, una array con las imagenes redimensionadas
    def getEigenValuesAll(self, imgs):
        eigen_vectors_list = []
        for img in imgs:
            eigen_vectors = self.getEigenVectors(img)
            eigen_vectors = np.nan_to_num(np.array(eigen_vectors).flatten())
            eigen_vectors_list.append(eigen_vectors)
        return np.array(eigen_vectors_list)

    # devuelve los valores reducidos de vector de caracteristicas
    # recibe las lista de los vectores de caracteristicasy los valores de clasificación
    def _reduceValues(self, eigen_vectors_list, answers):
        if answers is not None:
            return self.reductor.fit_transform(eigen_vectors_list, answers)
        else:
            return self.reductor.transform(eigen_vectors_list)

    # entrena el clasificador con las imagenes de entenamiento
    # recibe la lista de vectores de caracteristicas y sus valores de clasificacion
    def train(self, eigen_vectors, answers):
        pass

    # entrena el clasificador con las imagenes de entenamiento
    # recibe los valores reducidos y los valores de clasificación
    def _train(self, reduced_values, answers):
        self.clasificador.fit(reduced_values, answers) 

    # devuelve las prediccion de una imagen
    # recibe la imagen
    def predict(self, data):
        preds = self.predictAll([data])
        return preds[0]

    # devuelve las predicciones de las imagenes
    # recibe las imagenes
    def predictAll(self, imgs):
        pass

	# devuelve las predicciones de las imagenes
    # recibe los valores reducidos de las imagenes
    def _predictAll(self, reduced_values):
        return self.clasificador.predict(reduced_values)
