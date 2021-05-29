from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from Clasificador import Clasificador
import Constants

class Clasificador_KNN_LDA_LBP(Clasificador):
    def __init__(self):
        knn = KNeighborsClassifier(n_neighbors=Constants.N_NEIGHBOURS)
        lda = LinearDiscriminantAnalysis()
        # se utiliza directamente el algoritmo, descriptor = None
        super().__init__(knn, lda, None)

    # devuelve el vector de caracteristicas de la imagen
    # recibe la imagen redimensionada
    def getEigenVectors(self, img):
        return self._getLBPEigenVectors(img)

    def train(self, imgs, answers):
        eigen_vectors_list = self.getEigenValuesAll(imgs)
        reduced_values = self._reduceValues(eigen_vectors_list, answers)
        self._train(reduced_values, answers) 

    def predictAll(self, imgs):
        eigen_vectors_list = self.getEigenValuesAll(imgs)
        reduced_values = self._reduceValues(eigen_vectors_list, None)
        return self._predictAll(reduced_values)
