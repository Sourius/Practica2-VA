from Clasificador_LDA_HOG import ClasificadorLDA
import ImageUtils
import Clasificador

if __name__ == '__main__':
    lda = ClasificadorLDA() # clasificador LDA
    # cargar imagenes de entrenamiento y test
    train_images, train_answers = ImageUtils.getImages('train_recortadas', True)
    test_images, test_answers = ImageUtils.getImages('test_reconocimiento', False)

    # entrenar
    lda.train(train_images, train_answers)
    
    # predecir con LDA
    predicts = lda.predictAll(test_images)
    print("Precision: " + str(Clasificador.getStats(test_answers, predicts)) + " %")

    #print(str(clf.predict(test_images[0])))

