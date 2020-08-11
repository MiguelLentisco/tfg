#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Imports
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM, RepeatVector, TimeDistributed, Dense
from keras.regularizers import l2
from keras.models import load_model
from keras.optimizers import Adam
from scipy.stats import gaussian_kde
from keras.utils.vis_utils import plot_model

class LSTM_AD:
    """
        Clase que implementa un detector de anomalías usando
        un modelo autoencoder con capas LSTM.
        
        Attributes
        ----------
        model : keras.Sequential
            Autoencoder LSTM
        n_neur : int
            Número de neuronas base para las capas
        alpha : float
            Parámetro de regularización L2
        lr : float
            Learning rate
        epochs : int
            Número de épocas de entrenamiento
        mode : int
            Si incluir espacio de codificación (1) o no (2)
        hist : keras.Historial
            Historial de entrenamiento
        kernel : scipy.gaussian_kde
            Distribución de errores estimada
    """
    def __init__(self, n_neur = 32, alpha = 0, lr = 0.001, epochs = 300,
                 mode = 2):
        """
            Constructor de la clase
            
            Parameters
            ----------
            n_neur : int
                Número de neuronas base para las capas
            alpha : float
                Parámetro de regularización L2
            lr : float
                Learning rate
            epochs : int
                Número de épocas de entrenamiento
            mode : int
                Si incluir espacio de codificación (1) o no (2)
        """
        self.model = None
        self.n_neur = n_neur
        self.alpha = alpha
        self.lr = lr
        self.epochs = epochs
        self.hist = None
        self.kernel = None
        self.mode = mode
        
    def create_model(self, X):
        """
            Crea la arquitectura del autoencoder LSTM con los atributos
            de la clase.
            
            Parameters
            ----------
            X : np.numpy
                Series temporales
        """
        # Con espacio de codificación
        if self.mode == 1:
            self.model = Sequential()
            self.model.add(LSTM(self.n_neur * 2, activation = "relu", 
                           return_sequences = True,
                           kernel_regularizer = l2(self.alpha), 
                           input_shape = (X.shape[1], X.shape[2])))
            self.model.add(LSTM(self.n_neur, activation = "relu"))
            self.model.add(RepeatVector(X.shape[1]))
            self.model.add(LSTM(self.n_neur, activation = "relu", 
                           return_sequences = True))
            self.model.add(LSTM(self.n_neur * 2, activation = "relu", 
                           return_sequences = True,
                           kernel_regularizer = l2(self.alpha)))
            self.model.add(TimeDistributed(Dense(X.shape[2], 
                                                 activation = "linear")))
        # Sn espacio de codificación
        elif self.mode == 2:
            self.model = Sequential()
            self.model.add(LSTM(self.n_neur * 2, activation = "relu", 
                           return_sequences = True,
                           kernel_regularizer = l2(self.alpha), 
                           input_shape = (X.shape[1], X.shape[2])))
            self.model.add(LSTM(self.n_neur, activation = "relu",
                                return_sequences = True)),
            self.model.add(LSTM(self.n_neur, activation = "relu", 
                           return_sequences = True))
            self.model.add(LSTM(self.n_neur * 2, activation = "relu", 
                           return_sequences = True,
                           kernel_regularizer = l2(self.alpha)))
            self.model.add(TimeDistributed(Dense(X.shape[2], 
                                                 activation = "linear")))
            
        
    def compile_model(self):
        """
            Compila el modelo con ADAM añadiendo un clip de 1, learning
            rate especificado y minimizando el error cuadrático medio.
        """
        # Optimizador ADAM
        opt = Adam(learning_rate = self.lr, clipnorm = 1)
        # Compilamos usando error MSE
        self.model.compile(optimizer = opt, loss = "mse")
        
    def load_model(self, path):
        """
            Carga el modelo de unos pesos guardados en un archivo
            
            Parameters
            ----------
            path : str
                Ruta donde está el archivo de los pesos
        """
        self.model = load_model(path)
        
    def save_model(self, path):
        """
            Guarda los pesos del modelo en un archivo
            
            Parameters
            ----------
            path : str
                Ruta donde guardar el archivo con los pesos
        """
        self.model.save(path)

        
    def fit(self, X):
        """
            Entrena el modelo con el dataset de series temporales y los
            atributos de entrenamiento de atributos.
            
            Parameters
            ----------
            X : np.numpy
                Series temporales de entrenamiento    
        """
        
        # Creamos el modelo
        self.create_model(X)
        # Compilamos
        self.compile_model()
        # Entrenamos y guardamos
        self.hist = self.model.fit(X, X, epochs = self.epochs, 
                                   batch_size = 128,
                                   validation_split = 0.1, 
                                   shuffle = True).history
        # Estimamos la distribución de errores
        self.fit_kernel(X)
                                   
    def predict_autoencoder(self, X):
        """
            Obtiene las reconstrucciones del autoencoder para las series.
            
            Parameters
            ----------
            X : numpy.array
                Datasets de series temporales
            
            Returns
            -------
            reconstrucciones : numpy.array
                Reconstrucciones de las series temporales
        """
        return self.model.predict(X)
    
    def loss(self, X):
        """
            Se calcula la función de error MSE (error cuadrático medio) entre
            las series originales y sus reconstrucciones.
            
            Parameters
            ----------
            X : numpy.array
                Dataset de series temporales
                
            Returns
            -------
            losses : numpy.array
                Errores MSE de cada serie
        """
        # Adaptamos la dimensión
        X_predict = self.predict_autoencoder(X).reshape((X.shape[0], -1))
        X = X.reshape((X.shape[0], -1))
        # Error MAE para todas las series
        return np.mean(np.power(X_predict - X, 2), axis = 1)
        
    def fit_kernel(self, X):
        """
            Ajustamos la distribución de los errores de reconstrucción
            con los datos de entrenamiento.
            
            Parameters
            ----------
            X : numpy.array
                Dataset de series temporales
        """
        # Estimación de la distribucción de errores
        self.kernel = gaussian_kde(self.loss(X))
        
    def predict_prob(self, X):
        """
            Devolvemos las probabilidades de ser serie anómala para
            cada serie del dataset
            
            Parameters
            ----------
            X : numpy.array
                Dataset de series temporales
                
            Returns
            -------
            probs : numpy.array
                Probabilidades de anomalía para cada serie
        """
        # Integramos con la función de densidad para obtener las probabilidades
        probs = [self.kernel.integrate_box_1d(0, error)
                    for error in self.loss(X)]
        return np.array(probs)
    
    def save_img_model(self, path):
        """
            Guarda una imagen con la arquitectura del modelo
            
            Parameters
            ----------
            path : str
                Ruta de la imagen
        """
        plot_model(self.model, to_file = path, show_shapes = True, 
                   show_layer_names = False)
        
                                   
    def plot_historial(self):
        """
            Imprime el historial de entrenamiento.
        """
        fig, ax = plt.subplots(figsize = (14, 6), dpi = 80)
        # Imprimimos evolución de función de pérdida en train y validación
        ax.plot(self.hist["loss"], "b", label = "Train", linewidth = 2)
        ax.plot(self.hist["val_loss"], "r", label = "Val", linewidth = 2)
        plt.legend()
        plt.show()
       
    def plot_hist(self, X, density = False, title = "Histograma MSE",
                  axis = None):
        """
            Imprime el histograma de los errores de reconstrucción.
            
            Parameters
            ----------
            X : numpy.array
                Dataset de series
            density : boolean
                Imprimir los valores conforme a la densidad
            title : str
                Título de la gráfica
            axis : matplotlib.axis
                Objeto para imprimir la gŕafica
        """
        # Si no se pasa axis se imprime en una gráfica
        if axis is None:
            fig, ax = plt.subplots(1)
        else:
            ax = axis
        # Errores de reconstrucción
        errors = np.sort(self.loss(X))
        # Histograma
        ax.hist(errors, bins = 20, density = density, alpha = .6)
        # Función estimada
        ax.plot(errors, self.kernel(errors), color = "red")
        ax.set_xlabel("loss MSE")
        if density:
            ax.set_ylabel("nº series normalizadas")
        else:
            ax.set_ylabel("nº series")
        ax.set_title(title)
        if axis is None:
            plt.show()