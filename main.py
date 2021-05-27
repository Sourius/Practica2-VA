from Clasificador import Clasificador
import ImageUtils

if __name__ == '__main__':
    clf = Clasificador()
    train_images, train_answers = ImageUtils.getImages('/Users/christianbenavides/OneDrive - Universidad Rey Juan Carlos/Cuarto curso/Material/VA/Practica/Practica 2/train_recortadas',True)
    test_images, test_answers = ImageUtils.getImages('/Users/christianbenavides/OneDrive - Universidad Rey Juan Carlos/Cuarto curso/Material/VA/Practica/Practica 2/test_reconocimiento', False)
    clf.train(train_images, train_answers)
    _, precision = clf.predict(test_images, test_answers)
    print("Precision: " + str(round(precision * 100,2)) + " %")


