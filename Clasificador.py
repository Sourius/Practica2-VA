import numpy as np
from skimage.feature import local_binary_pattern
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import Constants
import os


def getStats(correct_vals, pred_vals):
    tags = [i for i in range(43)]
    m = confusion_matrix(y_true=correct_vals, y_pred=pred_vals, labels=tags, normalize='true')
    report = pd.DataFrame.from_dict(classification_report(y_true=correct_vals, y_pred=pred_vals, output_dict=True, zero_division=0)).transpose().drop(['accuracy', 'macro avg', 'weighted avg'], axis=0).drop(['support'], axis=1)
    clf_report = {'precision': report['precision'].mean(),
             'recall': report['recall'].mean(),
             'f1-score': report['f1-score'].mean(),
             'confusion-matrix': m
            }
    return clf_report


class Clasificador:
    def __init__(self, clasificador, reductor, descriptor):
        self.clasificador = clasificador
        self.reductor = reductor
        self.descriptor = descriptor
        self.reductor = reductor

    def generateResults(self, paths, correct_vals, pred_vals):
        report = getStats(correct_vals, pred_vals)
        output_file = open(Constants.RESULTS_FILE, "w")
        for i in range(len(paths)):
            pred = str(pred_vals[i]) if pred_vals[i] > 9 else "0" + str(pred_vals[i])
            output_file.write(paths[i] + "; " + str(pred) + "\n")
        output_file.close()

        if not os.path.exists(Constants.RESULTS_DIR):
            os.mkdir(Constants.RESULTS_DIR)

        class_name = str(type(self).__name__)
        file_path = os.path.join(Constants.RESULTS_DIR, class_name + "_stats.txt")
        stats_file = open(file_path, "w")
        stats_file.write(class_name + "\n")
        stats_file.write("Precision: " + str(round(report['precision'], 4)) + "\n")
        stats_file.write("Recall: " + str(round(report['precision'], 4)) + "\n")
        stats_file.write("F1 score: " + str(round(report['f1-score'], 4)) + "\n")
        stats_file.close()

        df = pd.DataFrame(report['confusion-matrix'])
        plt.figure(figsize=(20, 20))
        sn.heatmap(df, annot=True)
        img_path = os.path.join(Constants.RESULTS_DIR, class_name + "_matriz_confusion")
        plt.savefig(img_path)

    # devuelve el vector de caracteristicas lbp de la imagen
    # recibe la imagen redimensionada
    def _getLBPEigenVectors(self, img):
        # TODO: averiguar segundo parametro
        return local_binary_pattern(img, 8, 4)  # LBP

    # devuelve el vector de caracteristicas hog de la imagen
    # recibe la imagen redimensionada
    def _getHOGEigenVectors(self, img):
        return self.descriptor.compute(img)  # HOG

    # devuelve el vector de caracteristicas de la imagen
    # recibe la imagen redimensionada
    def getEigenVectors(self, img):
        pass

    # devuelve los vectores de caracteristicas de varias imagenes
    # recibe imgs, una array con las imagenes redimensionadas
    def getEigenValuesAll(self, imgs):
        eigen_vectors_list = []
        for img in imgs:
            eigen_vectors = self.getEigenVectors(img)
            eigen_vectors = np.nan_to_num(np.array(eigen_vectors).flatten())
            eigen_vectors_list.append(eigen_vectors)
        return np.array(eigen_vectors_list)

    # devuelve los valores reducidos de vector de caracteristicas
    # recibe las lista de los vectores de caracteristicasy los valores de clasificación
    def reduceValues(self, eigen_vectors_list, answers):
        if answers is not None:
            return self.reductor.fit_transform(eigen_vectors_list, answers)
        else:
            return self.reductor.transform(eigen_vectors_list)

    # entrena el clasificador con las imagenes de entenamiento
    # recibe la lista de vectores de caracteristicas y sus valores de clasificacion
    def train(self, data, answers):
        eigen_vectors = self.getEigenValuesAll(data)
        reduced_values = self.reduceValues(eigen_vectors, answers)
        self.clasificador.fit(reduced_values, answers)

    def lda_train(self, data, answers):
        eigen_vectors = self.getEigenValuesAll(data)
        self.clasificador.fit(eigen_vectors, answers)

    # devuelve las prediccion de una imagen
    # recibe la imagen
    def predict(self, data):
        preds = self.predictAll([data])
        return preds[0]

    # devuelve las predicciones de las imagenes
    # recibe las imagenes
    def predictAll(self, data):
        eigen_vectors = self.getEigenValuesAll(data)
        reduced_values = self.reduceValues(eigen_vectors, None)
        return self.clasificador.predict(reduced_values)

    def lda_predictAll(self, data):
        eigen_vectors = self.getEigenValuesAll(data)
        return self.clasificador.predict(eigen_vectors)
