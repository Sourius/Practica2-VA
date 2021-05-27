from Clasificador import Clasificador
import ImageUtils

if __name__ == '__main__':
    clf = Clasificador()
    train_images, train_answers = ImageUtils.getImages('train_recortadas', True)
    test_images, test_answers = ImageUtils.getImages('test_reconocimiento', False)
    clf.train(train_images, train_answers)
    _, precision = clf.predictAll(test_images, test_answers)
    print("Precision: " + str(round(precision * 100,2)) + " %")

    #print(str(clf.predict(test_images[0])))

