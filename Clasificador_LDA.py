from Clasificador import Clasificador

class Clasificador_LDA(Clasificador):
    def __init__(self, clasificador, reductor, descriptor):
        super().__init__(clasificador, reductor, descriptor)
    
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