from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from Clasificador import Clasificador

class Clasificador_LDA_LDA_LBP(Clasificador):
    # constructor
    def __init__(self):
        lda = LinearDiscriminantAnalysis()  # lda
        # crear clasificador
        super().__init__(lda, None, None)

    # devuelve el vector de caracteristicas de la imagen
    # recibe la imagen redimensionada
    def getEigenVectors(self, img):
        return self._getLBPEigenVectors(img)

    def train(self, imgs, answers):
        eigen_vectors_list = self.getEigenValuesAll(imgs)
        return self._train(eigen_vectors_list, answers)

    def predictAll(self, imgs):
        eigen_vectors_list = self.getEigenValuesAll(imgs)
        return self._predictAll(eigen_vectors_list)
    