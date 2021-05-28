from Clasificador_LDA_LBP import ClasificadorLDALBP
from Clasificador_LDA_HOG import ClasificadorLDAHOG
import ImageUtils
import Clasificador

if __name__ == '__main__':
    lda_hog = ClasificadorLDAHOG() # clasificador LDA
    # cargar imagenes de entrenamiento y test
    train_images, train_answers = ImageUtils.getImages('train_recortadas', True)
    test_images, test_answers = ImageUtils.getImages('test_reconocimiento', False)

    # entrenar
    lda_hog.train(train_images, train_answers)
    
    # predecir con LDA
    predicts_lda_hog = lda_hog.predictAll(test_images)
    precision_lda_hog = Clasificador.getStats(test_answers, predicts_lda_hog)
    print("Precision LDA LDA HOG: " + str(precision_lda_hog) + " %")

    lda_lbp = ClasificadorLDALBP()
    lda_lbp.train(train_images, train_answers)
    predicts_lda_lbp = lda_lbp.predictAll(test_images)
    precision_lda_lbp = Clasificador.getStats(test_answers, predicts_lda_lbp)
    print("Precision LDA LDA LBP: "+str(precision_lda_lbp) + "%")


