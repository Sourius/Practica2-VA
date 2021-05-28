from Clasificador_LDA_LDA_LBP import Clasificador_LDA_LDA_LBP
from Clasificador_LDA_LDA_HOG import Clasificador_LDA_LDA_HOG
from Clasificador_KNN_PCA_HOG import Clasificador_KNN_PCA_HOG
from Clasificador_KNN_LDA_LBP import Clasificador_KNN_LDA_LBP
from Clasificador_LDA_PCA_HOG import Clasificador_LDA_PCA_HOG
import ImageUtils
import Clasificador

if __name__ == '__main__':
    # cargar imagenes de entrenamiento y test
    train_images, train_answers = ImageUtils.getImages('train_recortadas', True)
    test_images, test_answers = ImageUtils.getImages('test_reconocimiento', False)
    
    lda_hog = Clasificador_LDA_LDA_HOG() # clasificador LDA
    # entrenar
    lda_hog.train(train_images, train_answers)
    # predecir
    predicts_lda_hog = lda_hog.predictAll(test_images)
    precision_lda_hog = Clasificador.getStats(test_answers, predicts_lda_hog)
    print("Precision LDA LDA HOG: " + str(precision_lda_hog) + " %")

    lda_lbp = Clasificador_LDA_LDA_LBP()
    # train
    lda_lbp.train(train_images, train_answers)
    predicts_lda_lbp = lda_lbp.predictAll(test_images)
    precision_lda_lbp = Clasificador.getStats(test_answers, predicts_lda_lbp)
    print("Precision LDA LDA LBP: "+str(precision_lda_lbp) + "%")

    knn_pca = Clasificador_KNN_PCA_HOG()
    knn_pca.train(train_images, train_answers)
    predicts_knn_pca = knn_pca.predictAll(test_images)
    precision_knn_pca = Clasificador.getStats(test_answers, predicts_knn_pca)
    print("Precision KNN PCA HOG: "+str(precision_knn_pca) + "%")

    knn_lda = Clasificador_KNN_LDA_LBP(5)
    knn_lda.train(train_images, train_answers)
    predicts_knn_lda = knn_lda.predictAll(test_images)
    precision_knn_lda = Clasificador.getStats(test_answers, predicts_knn_lda)
    print("Precision KNN LDA LBP: "+str(precision_knn_lda) + "%")

    n = min(len(train_images), len(test_images))
    lda_pca_hog = Clasificador_LDA_PCA_HOG(n)
    lda_pca_hog.train(train_images, train_answers)
    predicts_lda_pca_hog = lda_pca_hog.predictAll(test_images)
    precision_lda_pca_hog = Clasificador.getStats(test_answers, predicts_lda_pca_hog)
    print("Precision KNN LDA LBP: "+str(precision_lda_pca_hog) + "%")