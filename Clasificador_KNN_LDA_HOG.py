import cv2
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from Clasificador import Clasificador
import Constants

class Clasificador_KNN_LDA_HOG(Clasificador):
    def __init__(self):
        knn = KNeighborsClassifier(n_neighbors=Constants.N_NEIGHBOURS)

        # reduccion de dimensionalidad con lda
        lda = LinearDiscriminantAnalysis()

        # vectores de caracteristicas hog
        win_size = (Constants.HOG_WIN_SIZE, Constants.HOG_WIN_SIZE)  # tamaño de la imagen
        block_size = (Constants.HOG_BLOCK_SIZE, Constants.HOG_BLOCK_SIZE)  # tamaño del bloque
        block_stride = (Constants.HOG_BLOCK_STRIDE, Constants.HOG_BLOCK_STRIDE)  # tamaño de desplazamiento entre los bloques
        cell_size = (Constants.HOG_CELL_SIZE, Constants.HOG_CELL_SIZE)  # tamaño de las celdas
        hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, Constants.HOG_N_BINS)  

        # crear clasificador KNN_LDA_HOG
        super().__init__(knn, lda, hog)

    # devuelve el vector de caracteristicas de la imagen
    # recibe la imagen redimensionada
    def getEigenVectors(self, img):
        return self._getHOGEigenVectors(img)

    def train(self, data_list, answers):
        eigen_vectors = self.getEigenValuesAll(data_list)
        reduced_values = self._reduceValues(eigen_vectors, answers)
        self._train(reduced_values, answers) 

    def predictAll(self, data_list):
        eigen_vectors = self.getEigenValuesAll(data_list)
        reduced_values = self._reduceValues(eigen_vectors, None)
        return self._predictAll(reduced_values)
