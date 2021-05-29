from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from Clasificador import Clasificador


class Clasificador_LDA_LDA_LBP(Clasificador):
    def __init__(self):
        lda = LinearDiscriminantAnalysis()  # lda
        # crear clasificador
        super().__init__(lda, None, None)

    def getEigenVectors(self, img):
        return self._getLBPEigenVectors(img)

    def train(self, data_list, answers):
        super().lda_train(data_list, answers)

    def predictAll(self, data_list):
        return super().lda_predictAll(data_list)
