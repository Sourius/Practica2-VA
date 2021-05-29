from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from Clasificador import Clasificador


class Clasificador_LDA_LDA_LBP(Clasificador):
    def __init__(self):
        lda = LinearDiscriminantAnalysis()  # lda
        # crear clasificador
        super().__init__(lda, None, None)

    def getEigenVectors(self, img):
        return self._getLBPEigenVectors(img)

    def train(self, imgs, answers):
        self._lda_lda_train(imgs, answers)

    def predictAll(self, imgs):
        return self._lda_lda_predictAll(imgs)
