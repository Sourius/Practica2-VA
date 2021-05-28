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

    # devuelve el vector de caracteristicas hog de la imagen
    # recibe la imagen redimensionada
    def getEigenVectors(self, img):
        return self._getLBPEigenVectors(img)

    def train(self, data_list, answers):
        eigen_vectors_list = self.getEigenValuesAll(data_list)
        reduced_values = self._reduceValues(eigen_vectors_list, answers)
        return self._train(reduced_values, answers)

    def predictAll(self, data_list):
        eigen_vectors_list = self.getEigenValuesAll(data_list)
        reduced_values = self._reduceValues(eigen_vectors_list, None)
        return self._predictAll(reduced_values)

