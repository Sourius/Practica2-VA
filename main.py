from ClasificadorLDA import ClasificadorLDA
import ImageUtils

if __name__ == '__main__':
    lda = ClasificadorLDA() # clasificador LDA
    # cargar imagenes de entrenamiento y test
    train_images, train_answers = ImageUtils.getImages('train_recortadas', True)
    test_images, test_answers = ImageUtils.getImages('test_reconocimiento', False)

    # entrenar
    lda.train(train_images, train_answers)
    
    # predecir con LDA
    _, precision_con_lda = lda.predictAll(test_images, test_answers)
    print("Precision: " + str(precision_con_lda) + " %")

    # TODO: investigar porque falla --> solo si da tiempo
    #print(str(clf.predict(test_images[0])))

