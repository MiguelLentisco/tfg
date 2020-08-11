# ------------------------------------------------------------------------------------
# ----------------------------------- IMPORTS ----------------------------------------
# ------------------------------------------------------------------------------------

import matplotlib.pyplot as plt
import numpy as np

import keras
from keras.layers import Dense, Activation, BatchNormalization, Dropout, \
                         Conv1D, MaxPooling1D
from keras.models import Sequential
from keras.regularizers import l2
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from sklearn.base import BaseEstimator

# ------------------------------------------------------------------------------------
# -------------------------------------- LSTM ----------------------------------------
# ------------------------------------------------------------------------------------

class LSTM(BaseEstimator):
    """
        Implementación de una red neuronal con capas LSTM.

        Attributes
        ----------
        counter : int, static
            Valor auxiliar para ruta de imagen
        model : Sequential
            Modelo red neuronal
        history : list
            Historial del entrenamiento
        n_clases : int
            Nº de clases de las etiquetas
        input_shape : tuple
            Forma de los datos
        epochs: int
            Nº de épocas para entrenamiento
        verbose : int
            Información sobre el entrenamiento
        save_hist : boolean
            Si guardar las gráficas de los entrenamientos
    """
    counter = 1

    def __init__(self, epochs, n_neurs = 80, verbose = 0, save_hist = False,
                 n_clases = -1):
        """
            Inicializamos la red LSTM.

            Attributes
            ----------
            epochs : int
                Número de épocas para entrenamiento
            n_neurs : int
                Número de neuronas LSTM
            verbose : int
                Información sobre el entrenamiento
            save_hist : boolean
                Si guardar las gráficas de los entrenamientos
        """
        self.model = None
        self.history = None
        self.n_clases = n_clases
        self.input_shape = (-1)
        self.epochs = epochs
        self.verbose = verbose
        self.save_hist = save_hist
        self.n_neurs = n_neurs

    def create_model(self):
        """
            Crea el modelo LSTM.
        """

        self.model = Sequential()
        self.model.add(Conv1D(filters = 64, kernel_size = 8, strides = 1,
                              activation = "relu", 
                              input_shape = self.input_shape))
        self.model.add(MaxPooling1D(pool_size = 4))
        # LSTM 120 unidades + L2

        self.model.add(keras.layers.CuDNNLSTM(self.n_neurs, 
                                              return_sequences = False,
                                              kernel_regularizer = l2(0)))
        # LSTM 120 unidades + L2
        #self.model.add(CuDNNLSTM(120, kernel_regularizer = l2(0.01)))
        # Dense 100 + BN + ReLU + Dropout
        self.model.add(Dense(300, use_bias = False))
        self.model.add(BatchNormalization())
        self.model.add(Activation("relu"))
        self.model.add(Dropout(0.5))
        # Dense activaciones
        self.model.add(Dense(self.n_clases, activation = "softmax"))


    def compile_model(self):
        """
            Compila el modelo con optimizador ADAM y función de pérdida 
            categorical_crossentropy.
        """

        self.model.compile(optimizer = "adam", 
                           loss = "categorical_crossentropy", 
                           metrics = ["accuracy"])

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

        # Transformamos los datos
        self.input_shape = (X.shape[1], 1)
        X = np.reshape(X, (*X.shape, 1))
        # One-Hot encode
        y = to_categorical(y, self.n_clases)

        # Creamos el modelo y compilamos
        self.create_model()
        self.compile_model()

        # EarlyStopping
        es = EarlyStopping("val_acc", 0.1, 50, restore_best_weights = True)
        callbacks = [es]

        # Entrenamos y guardamos historial
        self.history = self.model.fit(X, y, validation_split = 0.1,
            epochs = self.epochs, batch_size = 128, verbose = self.verbose,
            callbacks = callbacks)
        # Guardamos el historial si está activado
        if self.save_hist:
            self.save_history()

    def score(self, X, y):
        """
            Calcula el acc con los datos que se le pasan.

            Parameters
            ----------
            X : numpy.array
                Datos test
            y : numpy.array
                Etiquetas test

            Returns
            ----------
            acc : float
                accuracy obtenida
        """

        # Transformamos los datos
        X = np.reshape(X, (*X.shape, 1))
        # One-Hot encode
        y = to_categorical(y, self.n_clases)
        # Evaluamos
        scores = self.model.evaluate(X, y, verbose = self.verbose)
        return scores[1]

    def save_history(self):
        """
            Guarda el historial en una imagen.
        """

        # Imágenes para acc/val_acc
        fig, ax = plt.subplots()
        ax.plot(self.history.history['acc'])
        ax.plot(self.history.history['val_acc'])
        ax.set_title('Model accuracy')
        ax.set_ylabel('Accuracy')
        ax.set_xlabel('Epoch')
        ax.legend(['Train', 'Test'], loc='upper left')
        fig.savefig("pv/hist/acc" + str(LSTM.counter) + ".png")
        plt.close(fig)

        # Imágenes para loss/val_loss
        fig, ax = plt.subplots()
        ax.plot(self.history.history['loss'])
        ax.plot(self.history.history['val_loss'])
        ax.set_title('Model loss')
        ax.set_ylabel('Loss')
        ax.set_xlabel('Epoch')
        ax.legend(['Train', 'Test'], loc='upper left')
        fig.savefig("pv/hist/loss" + str(LSTM.counter) + ".png")
        plt.close(fig)

        LSTM.counter += 1   