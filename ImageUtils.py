import cv2
import Constants
import os
import matplotlib.pyplot as plt


# devuelve la imagen equalizada recortada
# recibe la ruta de la imagen
def getProcessedImage(img_path):
    crop_dim = (Constants.DIM_X, Constants.DIM_Y)
    original = cv2.imread(img_path, 0)
    img_eq = cv2.equalizeHist(original)
    return cv2.resize(img_eq, crop_dim)


# muestra imagen con titulo
# recibe el titulo y la imagen
def printImage(title, img):
    plt.title(title)
    plt.imshow(img)
    plt.show()


# devuelve las imagenes que hay dentro del directorio y los subdirectorios
# devuelve valores de clasificación de las imagenes, 43 (0 a 42) clases
# recibe la ruta del directorio y un valor boolean que indica si es ruta de entrenamiento o no
def getImages(dir_path, istrain):
    imgs = []
    imgs_types = []
    for path, _, files in os.walk(dir_path):
        for name in files:
            if name.endswith(".ppm"):
                img_path = os.path.join(path, name)
                imgs.append(getProcessedImage(img_path))
                if istrain:
                    imgs_types.append(int(os.path.basename(path)))
                else:
                    imgs_types.append(int(name.strip().split('-')[0]))
    return imgs, imgs_types


def cvtGRAY2BGR(img_list):
    return [cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) for img in img_list]
