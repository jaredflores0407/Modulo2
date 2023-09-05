# Modulo2

### CONSTRUCCIÓN DE RED NEURONAL ###
"def entrenar_red_neuronal(X_train, y_train, X_test, y_test, hidden_size=5, learning_rate=0.000001, epochs=1100):"
La función entrenar_red_neuronal() toma los siguientes argumentos:

   - X_train: El conjunto de datos de entrenamiento. Es una matriz de características de entrada, donde cada fila representa un dato.
   - y_train: Las etiquetas de entrenamiento. Es un vector de objetivos de salida, donde cada elemento representa la etiqueta para el dato correspondiente en X_train.
   - X_test: El conjunto de datos de prueba. Es una matriz de características de entrada, donde cada fila representa un dato.
   - y_test: Las etiquetas de prueba. Es un vector de objetivos de salida, donde cada elemento representa la etiqueta para el dato correspondiente en X_test.
   - hidden_size: El número de neuronas en la capa oculta.
   - learning_rate: La tasa de aprendizaje para el algoritmo de descenso del gradiente.
   - epochs: El número de épocas de entrenamiento.

    
La función primero inicializa los pesos y sesgos de la red neuronal. Los pesos se inicializan a valores aleatorios y los sesgos se inicializan a cero.


Después entra en el bucle de entrenamiento. En dónde por cada época, la función realiza los siguientes pasos:

    - Realiza la propagación hacia adelante en donde las salidas de la red neuronal se comparan con los valores de destino conocidos para calcular la pérdida. (salidas de la capa oculta, Activación sigmoidea) (capa de salida, Activación lineal)
    - Calcula la pérdida entre las salidas predichas y las etiquetas de destino reales.
    - Realiza la retropropagación para calcular las gradientes de la pérdida con respecto a los pesos y sesgos.
    - Actualiza los pesos y sesgos utilizando el algoritmo de descenso del gradiente.

La pérdida se calcula como la media de la diferencia al cuadrado entre las salidas de la red neuronal y los valores de destino, se imprime el valor de la pérdida para cada época.

Finalmente, se define una función para realizar predicciones, en las que toma los datos de entrada y devuelve las predicciones hechas utilizando el modelo.


### SEPARACIÓN DE DATOS (TRAIN-TEST) Y EVALUACIÓN DEL MODELO CON DISTINTAS MÉTRICAS ###

En esta parte del código, se genera un ciclo en el que se van eligiendo distintos tamaños para el train y el test, así como un muestra de diferente tamaño para realizar predicciones.

    - Se estandarizan los datos de entrada para el entrenamiento de la red neuronal
    - Se pasan los argumentos correspondientes para la red neuronal, incluyendo un test size diferente con cada iteración
    - Se elige una muestra de tamaño diferente para realizar predicciones y se estadarizan las entradas
    - Se utilizan distintas métricas para la evalución de lo resultados obtenidos con los valores reales de la muestra previamente establecida 
    - Métricas utilizadas (RMSE, R-squared, accuracy_score, precision_score, recall_score, f1_score, confusion matrix)




