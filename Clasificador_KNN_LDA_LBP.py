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

    def getEigenVectors(self, img):
        return self._getLBPEigenVectors(img)

    def train(self, data_list, answers):
        super().train(data_list, answers)

    def predictAll(self, data_list):
        return super().predictAll(data_list)
