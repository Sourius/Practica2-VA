from Clasificador import Clasificador
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import cv2
import Constants

class ClasificadorLDA(Clasificador):
    #constructor
    def __init__(self):
        lda = LinearDiscriminantAnalysis() # lda
        
        win_size = (Constants.HOG_WIN_SIZE, Constants.HOG_WIN_SIZE) # tamaño de la imagen
        block_size = (Constants.HOG_BLOCK_SIZE, Constants.HOG_BLOCK_SIZE) # tamaño del bloque
        block_stride = (Constants.HOG_BLOCK_STRIDE, Constants.HOG_BLOCK_STRIDE) # tamaño de desplazamiento entre los bloques
        cell_size = (Constants.HOG_CELL_SIZE, Constants.HOG_CELL_SIZE) # tamaño de las celdas
        hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, Constants.HOG_N_BINS) # descriptor hog
        
        Clasificador.__init__(self, lda, hog, None)