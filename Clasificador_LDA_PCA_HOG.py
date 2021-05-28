from sklearn.decomposition import PCA
from Clasificador import Clasificador
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import Constants
import numpy as np
import cv2

class Clasificador_LDA_PCA_HOG(Clasificador):
    def __init__(self, n):
        lda = LinearDiscriminantAnalysis()
        pca = PCA(n_components=n, svd_solver='full')
        win_size = (Constants.HOG_WIN_SIZE, Constants.HOG_WIN_SIZE)  # tamaño de la imagen
        block_size = (Constants.HOG_BLOCK_SIZE, Constants.HOG_BLOCK_SIZE)  # tamaño del bloque
        block_stride = (Constants.HOG_BLOCK_STRIDE, Constants.HOG_BLOCK_STRIDE)  # tamaño de desplazamiento entre los bloques
        cell_size = (Constants.HOG_CELL_SIZE, Constants.HOG_CELL_SIZE)  # tamaño de las celdas
        hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, Constants.HOG_N_BINS)  # descriptor hog
        super().__init__(lda, pca, hog)

    # devuelve el vector de caracteristicas de la imagen
    # recibe la imagen redimensionada
    def getEigenVectors(self, img):
        eigen_vectors = self.descriptor.compute(img).flatten()
        eigen_vectors = np.nan_to_num(np.array(eigen_vectors))
        return eigen_vectors

    def train(self, data_list, answers):
        eigen_values_list = self.getEigenValuesAll(data_list)
        pca_values_list = self.reductor.fit_transform(eigen_values_list, answers)
        self.clasificador.fit(pca_values_list, answers)

    def predictAll(self, data_list):
        eigen_values_list = self.getEigenValuesAll(data_list)
        pca_values_list = self.reductor.fit_transform(eigen_values_list)
        predicts = self.clasificador.predict(pca_values_list)
        return predicts
