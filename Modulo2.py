#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import csv
from sklearn.metrics import mean_squared_error, r2_score


# In[2]:


### CONSTRUCCIÓN DE RED NEURONAL ###

def train_neural_network(X_train, y_train, X_test, y_test, hidden_size=5, learning_rate=0.000001, epochs=1500):
    # Inicialización de pesos y sesgos
    input_size = X_train.shape[1]
    output_size = y_train.shape[1]
    weights_input_hidden = np.random.rand(input_size, hidden_size)
    bias_hidden = np.zeros((1, hidden_size))
    weights_hidden_output = np.random.rand(hidden_size, output_size)
    bias_output = np.zeros((1, output_size))

    # Training loop
    for epoch in range(epochs):
        
        #propagación hacia adelante de la red neuronal. Calculan las salidas de la 
        #capa oculta y la capa de salida, utilizando los pesos y sesgos de la red.
        
        # Forward propagation
        hidden_input = np.dot(X_train, weights_input_hidden) + bias_hidden
        hidden_output = 1 / (1 + np.exp(-hidden_input))  # Activación sigmoide para capa oculta
        final_input = np.dot(hidden_output, weights_hidden_output) + bias_output
        predicted_output = final_input  # Activación lineal para la capa externa

        # Función de pérdida
        loss = 0.5 * np.mean((predicted_output - y_train) ** 2)      #pérdida entre las salidas de la red neuronal y los valores de destino conocidos.

        # Backpropagation
        dloss_predicted = predicted_output - y_train
        dloss_hidden = np.dot(dloss_predicted, weights_hidden_output.T)
        dloss_hidden_output = dloss_predicted
        dloss_hidden_input = dloss_hidden * hidden_output * (1 - hidden_output)

        # Actualizar los pesos y los sesgos
        weights_hidden_output -= learning_rate * np.dot(hidden_output.T, dloss_hidden_output)
        bias_output -= learning_rate * np.sum(dloss_hidden_output, axis=0, keepdims=True)
        weights_input_hidden -= learning_rate * np.dot(X_train.T, dloss_hidden_input)
        bias_hidden -= learning_rate * np.sum(dloss_hidden_input, axis=0, keepdims=True)

        if epoch % 100 == 0:
            print(f'Epoch {epoch}, Loss: {loss}')

    # Evaluación del modelo con el test
    hidden_input_test = np.dot(X_test, weights_input_hidden) + bias_hidden
    hidden_output_test = 1 / (1 + np.exp(-hidden_input_test))
    final_input_test = np.dot(hidden_output_test, weights_hidden_output) + bias_output
    predicted_output_test = final_input_test
    test_loss = 0.5 * np.mean((predicted_output_test - y_test) ** 2)

    print(f'Test Loss: {test_loss}')

    # Funcion de predicción
    def predict(X_new):
        hidden_input = np.dot(X_new, weights_input_hidden) + bias_hidden
        hidden_output = 1 / (1 + np.exp(-hidden_input))
        final_input = np.dot(hidden_output, weights_hidden_output) + bias_output
        predicted_output = final_input
        return np.round(predicted_output).astype(int)

    return predict


# In[3]:


train= pd.read_csv('diabetes2.csv')
len(train)
#train= pd.read_csv('train.csv')

#test= pd.read_csv('test.csv')
#test.drop(columns='id', inplace=True)


# In[4]:


X= train.drop('Outcome', axis=1)
y= train['Outcome'].values.reshape(-1,1)

#X= train.drop('price_range', axis=1)
#y= train['price_range'].values.reshape(-1,1)


# In[9]:


### Separación de los datos y metricas utilizadas para la evaluación del modelo ###

splits= [.15,.2,.25,.35,.45]
sample= [25,50,75,125,150]
    
for i,j in zip(splits,sample):
    # Dividir el df entre train y test
    scaler = StandardScaler()
    X_normalized = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=i, random_state=42)

    
    predictor = train_neural_network(X_train, y_train, X_test, y_test)

    print()

    # Hacer predicciones con el modelo
    test= train.sample(n=j, random_state=42)
    X_new= test.drop('Outcome', axis=1)

    scaler = StandardScaler()
    X_new_norm = scaler.fit_transform(X_new)
    y_real= test['Outcome']

    predictions = predictor(X_new_norm)

    # Evaluacion del modelo con diferentes métricas (e.g., RMSE, R-squared)
    rmse = np.sqrt(mean_squared_error(y_real, predictions))
    r_squared = r2_score(y_real, predictions)
    print(f"RMSE: {rmse:.4f}")
    print(f"R-squared: {r_squared:.4f}")

    print("Predictions:", predictions)

    print()

    from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score


    # Calcula accuracy
    accuracy = accuracy_score(y_real, predictions)

    # Calcula precision
    precision = precision_score(y_real, predictions, average='weighted')

    # Calcula recall
    recall = recall_score(y_real, predictions, average='weighted')

    # Calcula F1 score
    f1 = f1_score(y_real, predictions, average='weighted')

    print()

    # Imprimir resultados
    print(f'Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1:.4f}')

    # Reporte detallado de los resultados
    class_report = classification_report(y_real, predictions)
    print('\nClassification Report:\n', class_report)

    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    import matplotlib.pyplot as plt

    # Calcula matriz de confusión
    conf_matrix = confusion_matrix(y_real, predictions)

    # Plot the confusion matrix as a heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()
    
    print('RESULTADOS CON UN TEST SIZE DE: ', i)
    print()
    print()
    print()


# In[ ]:





# In[ ]:




