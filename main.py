from Clasificador_LDA_LDA_HOG import Clasificador_LDA_LDA_HOG
from Clasificador_LDA_LDA_LBP import Clasificador_LDA_LDA_LBP
from Clasificador_LDA_PCA_HOG import Clasificador_LDA_PCA_HOG
from Clasificador_KNN_PCA_HOG import Clasificador_KNN_PCA_HOG
from Clasificador_KNN_LDA_LBP import Clasificador_KNN_LDA_LBP
from Clasificador_KNN_LDA_HOG import Clasificador_KNN_LDA_HOG

import argparse
import ImageUtils
import Clasificador

if __name__ == '__main__':
    # cargar imagenes de entrenamiento y test
    train_images, train_answers = ImageUtils.getImages('train_recortadas', True)
    test_images, test_answers = ImageUtils.getImages('test_reconocimiento', False)
    
    parser = argparse.ArgumentParser(description='Trains and executes a given classifier over a set of testing images')
    parser.add_argument('--clasificador', default="clasificador_lda_lda_hog", help='Classifier string name')

    # obtener los argumentos
    args = parser.parse_args()

    clasificador = args.clasificador.lower()
    print(clasificador.upper())

    if(clasificador == "clasificador_lda_lda_hog"):
        clf = Clasificador_LDA_LDA_HOG() # clasificador LDA_LDA_HOG
        
    if(clasificador == "clasificador_lda_lda_lbp"):
        clf = Clasificador_LDA_LDA_LBP() # clasificador LDA_LDA_LBP

    if(clasificador == "clasificador_lda_pca_hog"):
        n = min(len(train_images), len(test_images)) 
        clf = Clasificador_LDA_PCA_HOG(n)# clasificador LDA_PCA_HOG

    if(clasificador == "clasificador_knn_lda_hog"):
        clf = Clasificador_KNN_LDA_HOG()# clasificador KNN_LDA_HOG
    
    if(clasificador == "clasificador_knn_lda_lbp"):
        clf = Clasificador_KNN_LDA_LBP()# clasificador KNN_LDA_LBP
    
    if(clasificador == "clasificador_knn_pca_hog"):
        clf = Clasificador_KNN_PCA_HOG()# clasificador KNN_PCA_HOG
    
    clf.train(train_images, train_answers)# entrenamiento
    predicts = clf.predictAll(test_images)# predección
    precision = Clasificador.getStats(test_answers, predicts)
    print("Precision: " + str(precision) + " %")
