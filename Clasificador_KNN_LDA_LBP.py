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

    def train(self, imgs, answers):
        self._train(imgs, answers)

    def predictAll(self, imgs):
        return self._predictAll(imgs)
