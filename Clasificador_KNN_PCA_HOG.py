import cv2
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from Clasificador import Clasificador
import Constants

class Clasificador_KNN_PCA_HOG(Clasificador):
    def __init__(self, n):
        # clasificador KNN
        knn = KNeighborsClassifier(n_neighbors=Constants.N_NEIGHBOURS)
        
        # reduccon de dimensionalidad con PCA
        pca = PCA(n, svd_solver='full')
        
        # utilizar vector de caracteristicas hog
        win_size = (Constants.HOG_WIN_SIZE, Constants.HOG_WIN_SIZE)  # tamaño de la imagen
        block_size = (Constants.HOG_BLOCK_SIZE, Constants.HOG_BLOCK_SIZE)  # tamaño del bloque
        block_stride = (Constants.HOG_BLOCK_STRIDE, Constants.HOG_BLOCK_STRIDE)  # tamaño de desplazamiento entre los bloques
        cell_size = (Constants.HOG_CELL_SIZE, Constants.HOG_CELL_SIZE)  # tamaño de las celdas
        hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, Constants.HOG_N_BINS)  # descriptor hog
        
        super().__init__(knn, pca, hog)

    # devuelve los vectores de caracteristicas de varias imagenes
    # recibe imgs, una array con las imagenes
    def getEigenVectors(self, img):
        return self._getHOGEigenVectors(img)

    # entrena el clasificador con las imagenes de entenamiento
    # recibe los valores reducidos de las imagenes y sus valores de clasificacion
    def train(self, data_list, answers):
        eigen_vectors_list = self.getEigenValuesAll(data_list)
        reduced_values = self._reduceValues(eigen_vectors_list, answers)
        self._train(reduced_values, answers)

    # devuelve las predicciones de las imagenes
    # recibe los valores reducidos de las imagenes
    def predictAll(self, data_list):
        eigen_vectors_list = self.getEigenValuesAll(data_list)
        reduced_values = self._reduceValues(eigen_vectors_list, None)
        return self._predictAll(reduced_values)