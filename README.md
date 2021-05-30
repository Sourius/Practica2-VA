# Clasificación de señales

## Introducción
Se ha implementado un clasificador supervisado de señales de tráfico con 43 tipos de señales distintas.

A partir de un directorio con imágenes de entrenamiento, se procesan las imágenes para entrenar el clasificador (entrenamiento). A partir de un directorio de prueba, se procesan las imágenes de las señales para reconocer de qué tipo son las señales (prueba, clasificación o test).

Con la clasificación de las imágenes de test se generan las salidas.

## Funcionamiento

El programa recibe 3 argumentos opcionales: train_path (directorio donde se ubican las imágenes de entrenamiento), test_path (directorio donde se encuentran las imágenes de prueba) y classifier (el nombre del clasificador).

El programa está formado principalmente por dos fases: entrenamiento y pruebas.
- Entrenamiento: A partir de un directorio con imágenes de entrenamiento, se procesan las imágenes para entrenar el clasificador. Cada imagen de la señal recortada está dentro de un subdirectorio que indica el tipo.
- Pruebas o testing: A partir de un directorio de prueba, se procesan las imágenes para clasificarlas en los 43 tipos de señales. Cada nombre del fichero viene precedido por el tipo a la cual pertenece.

Para clasificar las imágenes de test se crea el clasificador con el clasificador de sklearn correspondiente, el objeto para aplicar el algoritmo de reducción (reductor) y el objeto para calcular los vectores de características (descriptor). 

Con las imágenes de entrenamiento se calculan sus vectores de características, se obtienen los valores reducidos con fit_transform si es necesario y se entrena el clasificador con fit. 

Con las imágenes de test se calculan sus vectores de características, se obtienen los valores reducidos con transform si es necesario y se clasifican con predict.

El programa genera como resultado:
- resultado.txt: se guardan las imágenes de test con la predicción obtenida con el clasificador indicado
- directorio resultado: 
  - se guardan las estadísticas (precision, recall y f1 score) en un fichero png con el nombre que el detector + stats.txt.
  - se guarda la imagen con la matriz de confusión normalizada con el nombre del detector + matriz_confusion

### Ejecución del programa
Para poner en marcha el clasificador, se debe ejecutar el programa principal (main.py) con los 3 argumentos: train_path (directorio donde se ubican las imágenes de entrenamiento, por defecto: train_recortadas), test_path (directorio donde se encuentran las imágenes de prueba, por defecto: test_reconocimiento), classifier (nombre del clasificador). 

Los nombres de los clasificadores implementados son los siguientes:
- LDA_LDA_HOG
- LDA_LDA_LBP
- LDA_PCA_HOG
- LDA_PCA_LBP
- KNN_LDA_HOG
- KNN_LDA_LBP
- KNN_PCA_HOG
- KNN_PCA_LBP

Las primeras 3 letras indican el tipo de clasificador de sklearn utilizado, las segundas 3 letras indican el algoritmo de reducción de dimensionalidad utilizada y las tres últimas letras indican los tipos de vectores de características utilizados.

### Ejemplo de ejecución
- Ejecución con los parámetros:<br>
![image](https://user-images.githubusercontent.com/47939220/120105148-0039c180-c158-11eb-8d60-3897b44461d8.png)

- Fragmento de fichero txt de predicciones:<br>
![image](https://user-images.githubusercontent.com/47939220/120105166-13e52800-c158-11eb-910e-f691044a5fc6.png)

- Resultado de fichero de texto con las estadísticas <br>
![image](https://user-images.githubusercontent.com/47939220/120105184-295a5200-c158-11eb-80f1-ee9862817b43.png)

- Fragmento de la matriz de confusión <br>
![image](https://user-images.githubusercontent.com/47939220/120105197-3aa35e80-c158-11eb-8ca5-3d60fa198ac2.png)

## Implementación
Se ha implementado un clasificador LDA con reducción de dimensionalidad con LDA y el vector de características HOG como clasificador base. Para las alternativas se han utilizado clasificadores LDA y KNN, algoritmos de reducción LDA y PCA y vectores de características HOG y LBP. 

**Cada alternativa es una combinación posible de estos clasificadores, algoritmos de reducción de dimensionalidad y vectores de características.**

Para cada clasificador se ha creado una clase en su propio fichero .py.

Los ficheros .py utilizados para implementar los clasificadores son los siguientes:
- ImageUtils: proporciona metodos utiles para el tratamiento de las imágenes
- Clasificador: clase padre de cada clasificador implementado. Contiene los métodos comunes y los atributos en los que se guardan el clasificador de sklearn, el algoritmo de reducción de dimensionalidad y el vector de características
- Constants: un fichero donde se guardan las constantes de programa
- main: fichero principal o ejecutable

### Métodos de OpenCV utilizados
- imread: utilizado en ImageUtils para cargar las imágenes de entrenamiento y test
- equalizeHist: utilizado en ImageUtils para ampliar el rango dinámico 
- resize: utilizado en ImageUtils para redimensionar el tamaño de las imágenes a 30,30
- cvtColor: utilizado en ImageUtils para obtener la imagen con otros tipos de representación (openCV utiliza BGR por defecto)
- HogDescriptor: se utiliza para crear el objeto que calcula el vector de caracteristicas HoG. Para crear el objeto se usan los parametros de constants.

### Métodos de sklearn utilizados
- KNeighborsClassifier: utilizado para crear un clasificador knn
- LinearDiscriminantAnalysis: utilizado para crear el objeto que se utiliza para la reducción de dimensionalidad y clasificación con LDA.
- PCA: utilizado para crear el objeto que se utiliza para la reducción de domensionalidad con LDA. Para crear el objeto se utilizan los parámetros de Constants.
- confussion_matrix: utilizado para calcular la matriz de confusión dadas las predicciones y respuestas.
- classification_report: utilizado para obtener las estadísticas (precision, recall, f1 score)


Para calcular los vectores de características se utiliza:
- local_binary_pattern: para obtener el vector de características LBP de las imágenes con los parámetros especificados en Constants: 8 vecinos y radio 4.
- compute: para obtener el vector de características HoG

Para la reducción de dimensionalidad con los algoritmos de reducción:
- fit_transform: se utiliza para entrenar el LDA o PCA y obtener los valores reducidos de los vectores de características de las imágenes de entrenamiento.
- fit: se utiliza para obtener los valores reducidos de los vectores de características de las imágenes de test

De los clasificadores KNN y LDA, se utiliza: 
- fit: entrena con los vectores de características en caso de LDA con reducción con LDA y con los valores reducidos de los vectores de características en el resto.
- predict: se obtienen las predicciones de una lista de vectores de características en caso de predicción con LDA y reducción con LDA y de los valores reducidos en el resto.

## Estadísticas de los clasificadores

![image](https://user-images.githubusercontent.com/47939220/120105337-ce752a80-c158-11eb-98f1-453db9d6b1d4.png)

![image](https://user-images.githubusercontent.com/47939220/120105357-dd5bdd00-c158-11eb-9742-94d3947da43b.png)

A partir de los resultados de los clasificadores, se puede apreciar que el mejor clasificador ha sido el que usa LDA como clasificador, PCA como reducción y como vector de características HoG, con un F1 score de 0.9312.

Con los datos de entrenamiento y test, la precisión y el recall coinciden porque los falsos negativos son los mismos que los falsos positivos ya que al no clasificarse de un tipo, forzosamente se clasifican en otro.


