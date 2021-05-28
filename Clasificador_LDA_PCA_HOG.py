import cv2
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from Clasificador import Clasificador
import Constants

class Clasificador_LDA_PCA_HOG(Clasificador):
    def __init__(self, n):
        # clasificador lda
        lda = LinearDiscriminantAnalysis()
        
        # reduccion de dimensionalidad con pca
        pca = PCA(n_components=n, svd_solver='full')
        
        # vectores de caracteristicas hog
        win_size = (Constants.HOG_WIN_SIZE, Constants.HOG_WIN_SIZE)  # tamaño de la imagen
        block_size = (Constants.HOG_BLOCK_SIZE, Constants.HOG_BLOCK_SIZE)  # tamaño del bloque
        block_stride = (Constants.HOG_BLOCK_STRIDE, Constants.HOG_BLOCK_STRIDE)  # tamaño de desplazamiento entre los bloques
        cell_size = (Constants.HOG_CELL_SIZE, Constants.HOG_CELL_SIZE)  # tamaño de las celdas
        hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, Constants.HOG_N_BINS)  
        
        # clasificador LDA_PCA_HOG
        super().__init__(lda, pca, hog)

    # devuelve el vector de caracteristicas hog de la imagen
    # recibe la imagen redimensionada
    def getEigenVectors(self, img):
        return self._getHOGEigenVectors(img)

    def train(self, data_list, answers):
        eigen_vectors_list = self.getEigenValuesAll(data_list)
        reduced_values = self._reduceValues(eigen_vectors_list, answers)
        return self._train(reduced_values, answers)

    def predictAll(self, data_list):
        eigen_vectors_list = self.getEigenValuesAll(data_list)
        reduced_values = self._reduceValues(eigen_vectors_list, None)
        return self._predictAll(reduced_values)
