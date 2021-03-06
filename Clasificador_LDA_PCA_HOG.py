import cv2
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from Clasificador import Clasificador
import Constants


class Clasificador_LDA_PCA_HOG(Clasificador):
    def __init__(self, n):
        # clasificador lda
        lda = LinearDiscriminantAnalysis()

        # reduccion de dimensionalidad con pca
        pca = PCA(n_components=n, svd_solver='full')

        # vectores de caracteristicas hog
        win_size = (Constants.HOG_WIN_SIZE, Constants.HOG_WIN_SIZE)  # tamaño de la imagen
        block_size = (Constants.HOG_BLOCK_SIZE, Constants.HOG_BLOCK_SIZE)  # tamaño del bloque
        block_stride = (
        Constants.HOG_BLOCK_STRIDE, Constants.HOG_BLOCK_STRIDE)  # tamaño de desplazamiento entre los bloques
        cell_size = (Constants.HOG_CELL_SIZE, Constants.HOG_CELL_SIZE)  # tamaño de las celdas
        hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, Constants.HOG_N_BINS)

        # clasificador LDA_PCA_HOG
        super().__init__(lda, pca, hog)

    def getEigenVectors(self, img):
        return self._getHOGEigenVectors(img)

    def train(self, imgs, answers):
        self._train(imgs, answers)

    def predictAll(self, imgs):
        return self._predictAll(imgs)
