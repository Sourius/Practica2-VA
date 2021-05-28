from Clasificador_LDA_LDA_HOG import Clasificador_LDA_LDA_HOG
from Clasificador_LDA_LDA_LBP import Clasificador_LDA_LDA_LBP
from Clasificador_LDA_PCA_HOG import Clasificador_LDA_PCA_HOG
from Clasificador_KNN_PCA_HOG import Clasificador_KNN_PCA_HOG
from Clasificador_KNN_LDA_LBP import Clasificador_KNN_LDA_LBP
<<<<<<< HEAD
=======
from Clasificador_KNN_LDA_HOG import Clasificador_KNN_LDA_HOG
>>>>>>> parent of d3014aa (more classifiers + update main)

import ImageUtils
import Clasificador

if __name__ == '__main__':
    # cargar imagenes de entrenamiento y test
    train_images, train_answers = ImageUtils.getImages('train_recortadas', True)
    test_images, test_answers = ImageUtils.getImages('test_reconocimiento', False)
<<<<<<< HEAD
    
    lda_hog = Clasificador_LDA_LDA_HOG() # clasificador LDA
    # entrenar
    lda_hog.train(train_images, train_answers)
    # predecir
    predicts_lda_hog = lda_hog.predictAll(test_images)
    precision_lda_hog = Clasificador.getStats(test_answers, predicts_lda_hog)
    print("Precision LDA LDA HOG: " + str(precision_lda_hog) + " %")
    
    lda_lda_lbp = Clasificador_LDA_LDA_LBP()
    # train
    lda_lda_lbp.train(train_images, train_answers)
    predicts_lda_lda_lbp = lda_lda_lbp.predictAll(test_images)
    precision_lda_lda_lbp = Clasificador.getStats(test_answers, predicts_lda_lda_lbp)
    print("Precision LDA LDA LBP: "+str(precision_lda_lda_lbp) + "%")
    
    knn_pca = Clasificador_KNN_PCA_HOG()
    knn_pca.train(train_images, train_answers)
    predicts_knn_pca = knn_pca.predictAll(test_images)
    precision_knn_pca = Clasificador.getStats(test_answers, predicts_knn_pca)
    print("Precision KNN PCA HOG: "+str(precision_knn_pca) + " %")
    
    knn_lda = Clasificador_KNN_LDA_LBP()
    knn_lda.train(train_images, train_answers)
    predicts_knn_lda = knn_lda.predictAll(test_images)
    precision_knn_lda = Clasificador.getStats(test_answers, predicts_knn_lda)
    print("Precision KNN LDA LBP: "+str(precision_knn_lda) + " %")
    
    n = min(len(train_images), len(test_images))
    lda_pca_hog = Clasificador_LDA_PCA_HOG(n)
    lda_pca_hog.train(train_images, train_answers)
    predicts_lda_pca_hog = lda_pca_hog.predictAll(test_images)
    precision_lda_pca_hog = Clasificador.getStats(test_answers, predicts_lda_pca_hog)
    print("Precision KNN LDA LBP: "+str(precision_lda_pca_hog) + " %")
=======
    
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
>>>>>>> parent of d3014aa (more classifiers + update main)
