import cv2
from Clasificador import Clasificador
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import Constants


class Clasificador_LDA_LDA_HOG(Clasificador):
    def __init__(self):
        lda = LinearDiscriminantAnalysis()  # lda

        # utilizar vector de caracteristicas hog
        win_size = (Constants.HOG_WIN_SIZE, Constants.HOG_WIN_SIZE)  # tamaño de la imagen
        block_size = (Constants.HOG_BLOCK_SIZE, Constants.HOG_BLOCK_SIZE)  # tamaño del bloque
        block_stride = (
        Constants.HOG_BLOCK_STRIDE, Constants.HOG_BLOCK_STRIDE)  # tamaño de desplazamiento entre los bloques
        cell_size = (Constants.HOG_CELL_SIZE, Constants.HOG_CELL_SIZE)  # tamaño de las celdas
        hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, Constants.HOG_N_BINS)  # descriptor hog

        # crear clasificador
        super().__init__(lda, None, hog)

    def getEigenVectors(self, img):
        return self._getHOGEigenVectors(img)

    def train(self, imgs, answers):
        self._lda_lda_train(imgs, answers)

    def predictAll(self, imgs):
        return self._lda_lda_predictAll(imgs)

