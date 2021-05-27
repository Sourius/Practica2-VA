from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import cv2
import Constants
import numpy as np


def getPrecision(correct_vals, pred_vals):
    pred_vals = np.array(pred_vals)
    correct_vals = np.array(correct_vals)
    matches = pred_vals == correct_vals
    return matches.sum() / len(matches)


class Clasificador:
    def __init__(self):
        self.lda = LinearDiscriminantAnalysis()
        self.hog = cv2.HOGDescriptor((Constants.HOG_WIN_SIZE, Constants.HOG_WIN_SIZE),
                                     (Constants.HOG_BLOCK_SIZE, Constants.HOG_BLOCK_SIZE),
                                     (Constants.HOG_BLOCK_STRIDE, Constants.HOG_BLOCK_STRIDE),
                                     (Constants.HOG_CELL_SIZE, Constants.HOG_CELL_SIZE), Constants.HOG_N_BINS)

    def train(self, data, answers):
        self.lda.fit(self.applyHOGAll(data), answers)

    def predict(self, data):
        return self.lda.predict(self.applyHOG(data))

    def applyHOG(self, img):
        return self.hog.compute(img)

    def applyHOGAll(self, imgs):
        hog_values = []
        for img in imgs:
            hog_values.append(self.applyHOG(img).flatten())
        return hog_values

    def predictAll(self, imgs, answers):
        predictions = []
        for img in imgs:
            predictions.append(self.predict(img))
        return predictions, getPrecision(answers, predictions)
