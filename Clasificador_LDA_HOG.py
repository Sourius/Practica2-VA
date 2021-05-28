from Clasificador import Clasificador
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import cv2
import Constants


class ClasificadorLDA(Clasificador):
    # constructor
    def __init__(self):
        lda = LinearDiscriminantAnalysis()  # lda
        # utilizar vector de caracteristicas hog
        win_size = (Constants.HOG_WIN_SIZE, Constants.HOG_WIN_SIZE)  # tamaño de la imagen
        block_size = (Constants.HOG_BLOCK_SIZE, Constants.HOG_BLOCK_SIZE)  # tamaño del bloque
        block_stride = (
            Constants.HOG_BLOCK_STRIDE, Constants.HOG_BLOCK_STRIDE)  # tamaño de desplazamiento entre los bloques
        cell_size = (Constants.HOG_CELL_SIZE, Constants.HOG_CELL_SIZE)  # tamaño de las celdas
        hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, Constants.HOG_N_BINS)  # descriptor hog
        # crear clasificador
        Clasificador.__init__(self, lda, hog)

    # entrena el clasificador con las imagenes de entenamiento
    # recibe las imagenes y sus valores de clasificacion
    def train(self, data_list, answers):
        eigen_values_list = self.getEigenValuesAll(data_list)
        self.clasificador.fit(eigen_values_list, answers)

    def predict(self, data):
        preds = self.predictAll([data])
        return preds[0]

    # devuelve las predicciones y la precision global de las predicciones
    # recibe las imagenes y sus valores de clasificacion
    def predictAll(self, data_list):
        eigen_values_list = self.getEigenValuesAll(data_list)
        predictions = self.clasificador.predict(eigen_values_list)
        return predictions
