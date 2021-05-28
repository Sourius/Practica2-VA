import numpy as np

# devuelve la precision global de los valores predecidos
# recibe los valores correctos y los valores predecidos
def getPrecision(correct_vals, pred_vals):
    pred_vals = np.array(pred_vals)
    correct_vals = np.array(correct_vals)
    matches = pred_vals == correct_vals
    return round((matches.sum() / len(matches)) * 100 , 2)

class Clasificador:
    def __init__(self, clasificador, descriptor, reductor):
        self.clasificador = clasificador
        self.descriptor = descriptor
        self.reductor = reductor

    # devuelve el vector de caracteristicas de la imagen
    # recibe la imagen redimensionada
    def getEigenVectors(self, img):
        eigen_vectors = self.descriptor.compute(img).flatten()
        eigen_vectors = np.nan_to_num(np.array(eigen_vectors))
        return eigen_vectors

    # devuelve los vectores de caracteristicas de varias imagenes
    # recibe imgs, una array con las imagenes
    def getEigenValuesAll(self, imgs):
        eigen_vectors_list = []
        for img in imgs:
            eigen_vectors = self.getEigenVectors(img)
            eigen_vectors_list.append(eigen_vectors)
        return np.array(eigen_vectors_list)

    # entrena el clasificador con las imagenes de entenamiento
    # recibe las imagenes y sus valores de clasificacion
    def train(self, imgs, answers):
        eigen_values_list = self.getEigenValuesAll(imgs)
        if self.reductor is not None:
            data = self.reductor.transform(eigen_values_list, answers)
            self.clasificador.fit(data, answers)
        else:
            self.clasificador.fit(eigen_values_list, answers)

    def predict(self, img):
        eigen_vectors = self.getEigenVectors(img)
        return self.clasificador.predict(eigen_vectors)

    # devuelve las predicciones y la precision global de las predicciones
    # recibe las imagenes y sus valores de clasificacion
    def predictAll(self, imgs, answers):
        eigen_values_list = self.getEigenValuesAll(imgs)
        predictions = self.clasificador.predict(eigen_values_list)
        return predictions, getPrecision(answers, predictions)

    # devuelve los valores reducidos de las imagenes
    # recibe las imagenes
    def reduce(self, imgs):
        if self.reductor is None:
            return self.clasificador.transform(imgs)
        else:
            return self.reductor.transform(imgs)
