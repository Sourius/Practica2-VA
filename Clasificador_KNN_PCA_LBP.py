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

    def getEigenVectors(self, img):
        return self._getLBPEigenVectors(img)

    def train(self, imgs, answers):
        self._train(imgs, answers)

    def predictAll(self, imgs):
        return self._predictAll(imgs)
