from skimage.feature import local_binary_pattern
from Clasificador_LDA import Clasificador_LDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import numpy as np

class Clasificador_LDA_LDA_LBP(Clasificador_LDA):
    # constructor
    def __init__(self):
        lda = LinearDiscriminantAnalysis()  # lda
        # crear clasificador
        super().__init__(lda, None, None)

	# devuelve el vector de caracteristicas de la imagen
    # recibe la imagen redimensionada
    def getEigenVectors(self, img):
		# TODO: averiguar segundo parametro
        eigen_vectors = local_binary_pattern(img, 8, 4)
        eigen_vectors = np.nan_to_num(np.array(eigen_vectors).flatten())
        return eigen_vectors

