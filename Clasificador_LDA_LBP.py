from skimage.feature import local_binary_pattern
from Clasificador import ClasificadorLDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import numpy as np

class ClasificadorLDALBP(ClasificadorLDA):
    # constructor
    def __init__(self):
        lda = LinearDiscriminantAnalysis()  # lda
        # crear clasificador
        ClasificadorLDA.__init__(self, lda, None, None)

	# devuelve el vector de caracteristicas de la imagen
    # recibe la imagen redimensionada
    def getEigenVectors(self, img):
		# TODO: averiguar segundo parametro
        eigen_vectors = local_binary_pattern(img, 8, 4)
        eigen_vectors = np.nan_to_num(np.array(eigen_vectors).flatten())
        return eigen_vectors

    # devuelve los vectores de caracteristicas de varias imagenes
    # recibe imgs, una array con las imagenes
    def getEigenValuesAll(self, imgs):
        eigen_vectors_list = []
        for img in imgs:
            eigen_vectors = self.getEigenVectors(img)
            eigen_vectors_list.append(eigen_vectors)
        return np.array(eigen_vectors_list)
