from Clasificador_LDA_LDA_HOG import Clasificador_LDA_LDA_HOG
from Clasificador_LDA_LDA_LBP import Clasificador_LDA_LDA_LBP
from Clasificador_LDA_PCA_HOG import Clasificador_LDA_PCA_HOG
from Clasificador_LDA_PCA_LBP import Clasificador_LDA_PCA_LBP
from Clasificador_KNN_LDA_LBP import Clasificador_KNN_LDA_LBP
from Clasificador_KNN_LDA_HOG import Clasificador_KNN_LDA_HOG
from Clasificador_KNN_PCA_HOG import Clasificador_KNN_PCA_HOG
from Clasificador_KNN_PCA_LBP import Clasificador_KNN_PCA_LBP

import argparse
import ImageUtils
import Clasificador

if __name__ == '__main__':
    parser = argparse.ArgumentParser( description='Entrena sober train y ejecuta el clasificador sobre imgs de test')
    parser.add_argument('--train_path', type=str, default="train_recortadas", help='Path al directorio de imgs de train')
    parser.add_argument('--test_path', type=str, default="test_reconocimiento", help='Path al directorio de imgs de test')
    parser.add_argument('--classifier', type=str, default="lda_lda_hog", help='String con el nombre del clasificador')

    # obtener los argumentos
    args = parser.parse_args()

    # cargar y procesar imagenes de entrenamiento y test
    train_images, train_answers = ImageUtils.getImages(args.train_path, True)
    test_images, test_answers = ImageUtils.getImages(args.test_path, False)

    #Tratamiento de los datos
    # Crear el clasificador 
    classifier_name = args.classifier.lower()
    print(classifier_name.upper())

    if classifier_name == "lda_lda_hog":
        clf = Clasificador_LDA_LDA_HOG() # clasificador LDA_LDA_HOG
    elif classifier_name == "lda_lda_lbp":
        clf = Clasificador_LDA_LDA_LBP() # clasificador LDA_LDA_LBP
    elif classifier_name == "lda_pca_hog":
        n = min(len(train_images), len(test_images)) 
        clf = Clasificador_LDA_PCA_HOG(n)# clasificador LDA_PCA_HOG
    elif classifier_name == "lda_pca_lbp":
        n = min(len(train_images), len(test_images)) 
        clf = Clasificador_LDA_PCA_LBP(n)# clasificador LDA_PCA_LBP
    elif classifier_name == "knn_lda_hog":
        clf = Clasificador_KNN_LDA_HOG()# clasificador KNN_LDA_HOG
    elif classifier_name == "knn_lda_lbp":
        clf = Clasificador_KNN_LDA_LBP()# clasificador KNN_LDA_LBP
    elif classifier_name == "knn_pca_hog":
        n = min(len(train_images), len(test_images)) 
        clf = Clasificador_KNN_PCA_HOG(n)# clasificador KNN_PCA_HOG
    elif  classifier_name == "knn_pca_lbp":
        n = min(len(train_images), len(test_images)) 
        clf = Clasificador_KNN_PCA_LBP(n)# clasificador KNN_PCA_LBP
    else:
        raise ValueError('Tipo de clasificador incorrecto')        
    
    # Entrenar el clasificador
    clf.train(train_images, train_answers)# entrenamiento

    predicts = clf.predictAll(test_images)# predección
    precision = Clasificador.getStats(test_answers, predicts)
    print("Precision: " + str(precision) + " %")

    # Guardar los resultados en ficheros de texto
