from ClasificadorLDA import ClasificadorLDA
import ImageUtils

if __name__ == '__main__':
    clf = ClasificadorLDA() # clasificador LDA
    # cargar imagenes de entrenamiento y test
    train_images, train_answers = ImageUtils.getImages('train_recortadas', True)
    test_images, test_answers = ImageUtils.getImages('test_reconocimiento', False)

    # entrenar
    clf.train(train_images, train_answers)
    
    # predecir con LDA
    _, precision = clf.predictAll(test_images, test_answers)
    precision_con_lda = round(precision * 100,2)
    print("Precision: " + str(precision_con_lda) + " %")

    # TODO: investigar porque falla --> solo si da tiempo
    #print(str(clf.predict(test_images[0])))

