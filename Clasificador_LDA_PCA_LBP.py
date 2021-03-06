from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from Clasificador import Clasificador


class Clasificador_LDA_PCA_LBP(Clasificador):
    def __init__(self, n):
        # clasificador lda
        lda = LinearDiscriminantAnalysis()

        # reduccion de dimensionalidad con pca
        pca = PCA(n_components=n, svd_solver='full')

        # vectores de caracteristicas lbp
        # clasificador LDA_PCA_LBP
        super().__init__(lda, pca, None)

    def getEigenVectors(self, img):
        return self._getLBPEigenVectors(img)

    def train(self, imgs, answers):
        self._train(imgs, answers)

    def predictAll(self, imgs):
        return self._predictAll(imgs)
