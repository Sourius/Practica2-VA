from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from Clasificador import Clasificador
import Constants

class Clasificador_KNN_PCA_LBP(Clasificador):
    def __init__(self, n):
        # clasificador KNN
        knn = KNeighborsClassifier(n_neighbors=Constants.N_NEIGHBOURS)
        
        # reduccon de dimensionalidad con PCA
        pca = PCA(n, svd_solver='full')

        # utilizar vector de caracteristicas lbp
		# clasificador KNN_PCA_LBP
        super().__init__(knn, pca, None)

    # devuelve los vectores de caracteristicas de varias imagenes
    # recibe imgs, una array con las imagenes
    def getEigenVectors(self, img):
        return self._getLBPEigenVectors(img)
    
    # entrena el clasificador con las imagenes de entenamiento
    # recibe los valores reducidos de las imagenes y sus valores de clasificacion
    def train(self, data_list, answers):
        eigen_vectors_list = self.getEigenValuesAll(data_list)
        reduced_values = self._reduceValues(eigen_vectors_list, answers)
        self._train(reduced_values, answers)

    # devuelve las predicciones de las imagenes
    # recibe los valores reducidos de las imagenes
    def predictAll(self, data_list):
        eigen_values_list = self.getEigenValuesAll(data_list)
        reduced_values = self._reduceValues(eigen_values_list, None)
        return self._predictAll(reduced_values)
