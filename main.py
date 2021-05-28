from Clasificador_LDA_HOG import ClasificadorLDA
from Clasificador_LDA_PCA_HOG import ClasificadorLDA_PCA_HOG
from Clasificador_KNN_LDA_LBP import Clasificador_KNN_LDA_LBP
import ImageUtils
import Clasificador

if __name__ == '__main__':
    # cargar imagenes de entrenamiento y test
    train_images, train_answers = ImageUtils.getImages('train_recortadas', True)
    test_images, test_answers = ImageUtils.getImages('test_reconocimiento', False)
    #clf = ClasificadorLdaPcaHog(min(len(train_answers), len(test_images)))  # clasificador LDA
    clf = Clasificador_KNN_LDA_LBP(5)

    # entrenar
    clf.train(train_images, train_answers)
    
    # predecir con LDA
    predicts = clf.predictAll(test_images)
    print("Precision: " + str(Clasificador.getStats(test_answers, predicts)) + " %")


