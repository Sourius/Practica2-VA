import numpy as np


# devuelve la precision global de los valores predecidos
# recibe los valores correctos y los valores predecidos
def getStats(correct_vals, pred_vals):
    pred_vals = np.array(pred_vals)
    correct_vals = np.array(correct_vals)
    matches = pred_vals == correct_vals
    # TODO: ampliar stats: precison, tpr, fpr, recall, f1score, matriz
    return round((matches.sum() / len(matches)) * 100, 2)


class Clasificador:
    def __init__(self, clasificador, descriptor, reductor):
        self.clasificador = clasificador
        self.descriptor = descriptor
        self.reductor = reductor

    # devuelve el vector de caracteristicas de la imagen
    # recibe la imagen redimensionada
    def getEigenVectors(self, img):
        pass

    # devuelve los vectores de caracteristicas de varias imagenes
    # recibe imgs, una array con las imagenes
    def getEigenValuesAll(self, imgs):
        pass

    # entrena el clasificador con las imagenes de entenamiento
    # recibe las imagenes y sus valores de clasificacion
    def train(self, data_list, answers):
        pass

    # devuelve las prediccion de la imagen
    # recibe los valores reducidos de las imagenes
    def predict(self, data):
        pass

    # devuelve las predicciones y la precision global de las predicciones
    # recibe las imagenes y sus valores de clasificacion
    def predictAll(self, data_list):
        pass

class ClasificadorLDA(Clasificador):
    def __init__(self, clasificador, descriptor, reductor):
        Clasificador.__init__(self, clasificador, descriptor, reductor)
    
    # entrena el clasificador con las imagenes de entenamiento
    # recibe los valores reducidos de las imagenes y sus valores de clasificacion
    def train(self, data_list, answers):
        eigen_values_list = self.getEigenValuesAll(data_list)
        self.clasificador.fit(eigen_values_list, answers)

	# devuelve las prediccion de la imagen
    # recibe los valores reducidos de las imagenes
    def predict(self, data):
        preds = self.predictAll([data])
        return preds[0]

    # devuelve las predicciones de las imagenes
    # recibe los valores reducidos de las imagenes
    def predictAll(self, data_list):
        eigen_values_list = self.getEigenValuesAll(data_list)
        predictions = self.clasificador.predict(eigen_values_list)
        return predictions
