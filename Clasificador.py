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
        win_size = (Constants.HOG_WIN_SIZE, Constants.HOG_WIN_SIZE)
        block_size = (Constants.HOG_BLOCK_SIZE, Constants.HOG_BLOCK_SIZE)
        block_stride = (Constants.HOG_BLOCK_STRIDE, Constants.HOG_BLOCK_STRIDE)
        cell_size = (Constants.HOG_CELL_SIZE, Constants.HOG_CELL_SIZE)
        self.hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, Constants.HOG_N_BINS)

    def applyHOG(self, img):
        hog_value = self.hog.compute(img).flatten()
        hog_value = np.nan_to_num(np.array(hog_value))
        return hog_value

    def applyHOGAll(self, imgs):
        hog_values = []
        for img in imgs:
            hog_value = self.applyHOG(img)
            hog_values.append(hog_value)
        return np.array(hog_values)

    def train(self, imgs, answers):
        self.lda.fit(self.applyHOGAll(imgs), answers)

    # TODO: investigar porque falla
    def predict(self, img):
        hog_value = self.applyHOG(img)
        return self.lda.predict(hog_value)

    def predictAll(self, imgs, answers):
        hog_values = self.applyHOGAll(imgs)
        predictions = self.lda.predict(hog_values)

        return predictions, getPrecision(answers, predictions)
