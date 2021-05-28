from Clasificador import Clasificador
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import Constants
from skimage.feature import local_binary_pattern
import cv2
from sklearn.neighbors import KNeighborsClassifier
import numpy as np


class Clasificador_KNN_LDA_LBP(Clasificador):
    def __init__(self, k):
        self.clasificador = KNeighborsClassifier(n_neighbors=k)
        self.reductor = LinearDiscriminantAnalysis()
        self.descriptor = None

    # devuelve el vector de caracteristicas de la imagen
    # recibe la imagen redimensionada
    def getEigenVectors(self, img):
        eigen_vectors = local_binary_pattern(img, 8, 4).flatten()
        eigen_vectors = np.nan_to_num(np.array(eigen_vectors))
        return eigen_vectors

    def train(self, data_list, answers):
        eigen_vectors = self.getEigenValuesAll(data_list)
        lda_values_list = self.reduce(eigen_vectors, answers)
        self.clasificador.fit(lda_values_list, answers)

    def predictAll(self, data_list):
        eigen_vectors = self.getEigenValuesAll(data_list)
        lda_values_list = self.reductor.transform(eigen_vectors)
        return self.clasificador.predict(lda_values_list)
