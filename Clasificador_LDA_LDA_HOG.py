from Clasificador_LDA import Clasificador_LDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import cv2
import Constants
import numpy as np


class Clasificador_LDA_LDA_HOG(Clasificador_LDA):
    # constructor
    def __init__(self):
        lda = LinearDiscriminantAnalysis()  # lda
        # utilizar vector de caracteristicas hog
        win_size = (Constants.HOG_WIN_SIZE, Constants.HOG_WIN_SIZE)  # tamaño de la imagen
        block_size = (Constants.HOG_BLOCK_SIZE, Constants.HOG_BLOCK_SIZE)  # tamaño del bloque
        block_stride = (Constants.HOG_BLOCK_STRIDE, Constants.HOG_BLOCK_STRIDE)  # tamaño de desplazamiento entre los bloques
        cell_size = (Constants.HOG_CELL_SIZE, Constants.HOG_CELL_SIZE)  # tamaño de las celdas
        hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, Constants.HOG_N_BINS)  # descriptor hog
        # crear clasificador
        super().__init__(lda, None, hog)

	# devuelve el vector de caracteristicas de la imagen
    # recibe la imagen redimensionada
    def getEigenVectors(self, img):
        eigen_vectors = self.descriptor.compute(img).flatten()
        eigen_vectors = np.nan_to_num(np.array(eigen_vectors))
        return eigen_vectors
		
