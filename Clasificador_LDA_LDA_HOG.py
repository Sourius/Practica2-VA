import cv2
from Clasificador import Clasificador
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import Constants


class Clasificador_LDA_LDA_HOG(Clasificador):
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
        return self._getHOGEigenVectors(img)

    def train(self, data_list, answers):
        eigen_vectors_list = self.getEigenValuesAll(data_list)
        return self._train(eigen_vectors_list, answers)      

    def predictAll(self, data_list):		
        eigen_vectors_list = self.getEigenValuesAll(data_list)
        return self._predictAll(eigen_vectors_list)