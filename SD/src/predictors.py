# -----------------------------------------------------------------------------
# ----------------------------------- IMPORTS ---------------------------------
# -----------------------------------------------------------------------------

import matplotlib.pyplot as plt
import numpy as np

import keras
from keras.layers import Dense, Activation, BatchNormalization, Dropout, \
                         Conv1D, MaxPooling1D, LSTM
from keras.models import Sequential
from sklearn.base import BaseEstimator
from sklearn.preprocessing import OneHotEncoder


# -----------------------------------------------------------------------------
# -------------------------------------- LSTM ---------------------------------
# -----------------------------------------------------------------------------

class BasePredictor(BaseEstimator):
    """
        Predictor neuronal base.

        Attributes
        ----------
        model : Sequential
            Modelo red neuronal
        history : list
            Historial del entrenamiento
        epochs: int
            Nº de épocas para entrenamiento
        verbose : int
            Información sobre el entrenamiento
        encoder : OneHotEncoder
            Codificador one-hot
    """
    
    def __init__(self, epochs = 100, n_neurs = 32, verbose = 0):
        """
            Constructor.

            Attributes
            ----------
            epochs : int
                Número de épocas para entrenamiento
            n_neurs : int
                Número de neuronas
            verbose : int
                Información sobre el entrenamiento
        """
        
        self.model = None
        self.history = None
        self.n_neurs = n_neurs
        self.epochs = epochs
        self.encoder = OneHotEncoder()
        
    def create_model(self, n_vals, X):
        """
            Crea la estructural del modelo neuronal.
            
            Parameters
            ----------
            n_vals: int
                Nº de valores discretos
            X : numpy.array
                Datos de entrenamiento
        """
    
    def compile_model(self):
        """
            Compila el modelo con optimizador ADAM y función de pérdida 
            categorical_crossentropy.
        """

        self.model.compile(optimizer = "adam", 
                           loss = "categorical_crossentropy", 
                           metrics = ["mse"])
    def fit(self, X, y):
        """
            Entrenamos el modelo.

            Parameters
            ----------
            X : numpy.array
                Datos de entrenamiento
            y : numpy.array
                Etiquetas de entrenamiento
        """

        # Nº de valores discretos
        n_vals = np.size(np.unique(y))
        # One-Hot encode
        y = self.encoder.fit_transform(y.reshape((-1, 1))).toarray()

        # Creamos el modelo y compilamos
        self.create_model(n_vals, X)
        self.compile_model()

        # Entrenamos y guardamos historial
        self.history = self.model.fit(X, y, validation_split = 0.2,
                                      shuffle = False,
                                      epochs = self.epochs, 
                                      batch_size = 128, 
                                      verbose = self.verbose)
        
    def predict(self, X):
        """
            Predice los valores para cada lote de X.
            
            Parameters
            ----------
            X : numpy.array
                Datos de entrada
            
            Returns
            -------
            res : numpy.array
                Valores predichos
        """
        
        return self.encoder.inverse_transform(self.model.predict(X))
        
    def print_history(self):
        """
            Imprime el historial de entrenamiento.
        """

        # Imágenes para acc/val_acc
        fig, axs = plt.subplots(1, 2, figsize = (18, 8))
        axs = axs.flat
        axs[0].plot(self.history.history['mse'])
        axs[0].plot(self.history.history['val_mse'])
        axs[0].set_title('Model MSE')
        axs[0].set_ylabel('MSE')
        axs[0].set_xlabel('Epoch')
        axs[0].legend(['Train', 'Test'], loc='upper left')

        # Imágenes para loss/val_loss
        axs[1].plot(self.history.history['loss'])
        axs[1].plot(self.history.history['val_loss'])
        axs[1].set_title('Model loss')
        axs[1].set_ylabel('Loss')
        axs[1].set_xlabel('Epoch')
        axs[1].legend(['Train', 'Test'], loc='upper left')
        plt.show()
        plt.close()

class PredictorLSTM(BasePredictor):
    """
        Predictor con modelo LSTM.
    """

    def create_model(self, n_vals, X):
        """
            Crea la estructural del modelo neuronal.
            
            Parameters
            ----------
            n_vals: int
                Nº de valores discretos
            X : numpy.array
                Datos de entrenamiento
        """
        
        self.model = Sequential()
        self.model.add(Conv1D(filters = 8, kernel_size = 2, strides = 1,
                              activation = "relu", 
                              input_shape = (X.shape[1], X.shape[2])))
        self.model.add(MaxPooling1D(pool_size = 2))
        self.model.add(LSTM(self.n_neurs, 
                            return_sequences = False))
        # Dense 100 + BN + ReLU + Dropout
        self.model.add(Dense(100, use_bias = False))
        self.model.add(BatchNormalization())
        self.model.add(Activation("relu"))
        self.model.add(Dropout(0.5))
        # Dense activaciones
        self.model.add(Dense(n_vals, activation = "softmax"))
   


    