from Clasificador import Clasificador
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from skimage.feature import local_binary_pattern
from sklearn.neighbors import KNeighborsClassifier
import numpy as np


class Clasificador_KNN_LDA_LBP(Clasificador):
    def __init__(self, k):
        knn = KNeighborsClassifier(n_neighbors=k)
        lda = LinearDiscriminantAnalysis()
        # se utiliza directamente el algoritmo, descriptor = None
        super().__init__(knn, lda, None)

    # devuelve el vector de caracteristicas de la imagen
    # recibe la imagen redimensionada
    def getEigenVectors(self, img):
        eigen_vectors = local_binary_pattern(img, 8, 4).flatten()
        eigen_vectors = np.nan_to_num(np.array(eigen_vectors))
        return eigen_vectors

    def train(self, data_list, answers):
        eigen_vectors = self.getEigenValuesAll(data_list)
        self.reductor.fit(eigen_vectors, answers)
        reduced_values = self.reductor.transform(eigen_vectors)
        self.clasificador.fit(reduced_values, answers)   

    def predictAll(self, data_list):
        eigen_vectors = self.getEigenValuesAll(data_list)
        reduced_values = self.reductor.transform(eigen_vectors)
        return self.clasificador.predict(reduced_values)
