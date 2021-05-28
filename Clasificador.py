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
    def __init__(self, clasificador, reductor, descriptor):
        self.clasificador = clasificador
        self.reductor = reductor
        self.descriptor = descriptor
        self.reductor = reductor

    # devuelve el vector de caracteristicas de la imagen
    # recibe la imagen redimensionada
    def getEigenVectors(self, img):
        pass

    # devuelve los vectores de caracteristicas de varias imagenes
    # recibe imgs, una array con las imagenes
    def getEigenValuesAll(self, imgs):
        eigen_vectors_list = []
        for img in imgs:
            eigen_vectors = self.getEigenVectors(img)
            eigen_vectors_list.append(eigen_vectors)
        return np.array(eigen_vectors_list)

    # devuelve los valores reducidos de vector de caracteristicas
    # recibe las lista de los vectores de caracteristicas y los valores de clasificación
    def _reduceValues(self, eigen_vectors_list, answers):
        if answers is not None:
            return self.reductor.fit_transform(eigen_vectors_list, answers)
        else:
            return self.reductor.transform(eigen_vectors_list)

    # entrena el clasificador con las imagenes de entenamiento
    # recibe las imagenes y sus valores de clasificacion
    def train(self, data_list, answers):
        pass

    # devuelve las prediccion de la imagen
    # recibe los valores reducidos de las imagenes
    def predict(self, data):
        preds = self.predictAll([data])
        return preds[0]

    # devuelve las predicciones de las imagenes
    # recibe las imagenes
    def predictAll(self, data_list):
        pass
