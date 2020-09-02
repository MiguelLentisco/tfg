#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import string
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import LabelEncoder

class SAX:
    """
        Clase que implementa el método de discretiación SAX.
        
        Attributes
        ----------
        tam_window : int
            Tamaño de ventana.
        breakpoints : list
            Lista de los puntos de ruptura.
        alphabet : list
            Lista del alfabeto.
        mean : np.array
            Medias de las series.
        std : np.array
            Desviaciones estándar de las series.
    """
    def __init__(self, tam_window, alphabet_tam):
        """
            Constructor.
            
            Parameters
            ----------
            tam_window : int
                Tamaño de ventana.
            alphabet_tam : int
                Tamaño del alfabeto.
        """
        self.tam_window = tam_window
        # Calculamos los breakpoints
        self.breakpoints = self.calc_breakpoints(alphabet_tam)
        # Obtenemos el alfabeto
        self.alphabet = string.ascii_lowercase[:alphabet_tam]
        self.std = None
        self.mean = None
        
    def standarization(self, X):
        """
            Normaliza las series temporales que se pasan.
            
            Parameters
            ----------
            X : np.array
                Las series temporales.
            
            Returns
            -------
            res : np.array
                Las series temporales normalizadas.
        """
        res = [(X[i] - self.mean[i]) / self.std[i] for i in range(X.shape[0])]
        return np.array(res)
        
    def PAA(self, x):
        """
            Realiza la reducción de dimensión PAA.
            
            Parameters
            ----------
            x : np.array
                La serie temporal.
            
            Returns
            -------
            res : np.array
                La serie temporal reducida.
        """
        n_window = x.size // self.tam_window
        # Media de cada ventana
        res = [np.mean(x[(i*self.tam_window):((i+1)*self.tam_window)]) 
                for i in range(n_window)]
        return np.array(res)    
        
    def calc_breakpoints(self, alphabet_tam):
        """
            Cálculo de los puntos de ruptura, de manera que entre cada par
            de valores consecutivos hay 1/alphabet_tam de área debajo
            de la curva de una distribución normal.
            
            Parameters
            ----------
            alphabet_tam : int
                Tamaño del alfabeto.
            
            Returns
            -------
            bp : list
                Lista con los puntos de ruptura.
        """
        bp = [float("-inf")]
        # Función inversa de probabilidad cumulativa
        bp += [norm.ppf((i+1) / alphabet_tam) for i in range(alphabet_tam - 1)]
        bp += [float("inf")]
        return bp
    
    def string_transformation(self, x_paa):
        """
            Transforma una serie en una palabra tomando para cada valor
            si se encuentra en el intervalo determinado por los puntos
            de ruptura.
            
            Parameters
            ----------
            x_paa : np.array
                Serie temporal.
            
            Returns
            -------
            word : list
                Serie convertida en palabra (lista de carácteres)
        """
        word = []
        for point in x_paa:
            for i in range(1, len(self.breakpoints)):
                if point >= self.breakpoints[i-1] and \
                    point < self.breakpoints[i]:
                    word.append(self.alphabet[i-1])
        return word
    
    def discretization(self, x):
        """
            Discretiza aplicando PAA y la transformación a palabra.
            
            Parameters
            ----------
            x : np.array
                Serie temporal.
            
            Returns
            -------
            word : list
                Serie transformada en palabra.
        """
        x_paa = self.PAA(x)
        return self.string_transformation(x_paa)
    
    def fit(self, X, y = None):
        """
            Ajusta las medias y desviaciones típicas de las series.
            
            Parameters
            ----------
            X : np.array
                Series temporales
            y : None
                Por compatibilidad.
            
            Returns
            -------
            self : SAX
                El propio objeto.
        """
        self.mean = np.mean(X, axis = 1)
        self.std = np.std(X, axis = 1)
        return self
        
    def transform(self, X, y = None):
        """
            Realiza la transformación SAX a todas las series que se le pasan.
            
            Parameters
            ----------
            X : np.array
                Series temporales.
            y : None
                Por compatibilidad.
                
            Returns
            -------
            X_strings : list
                Lista con las series transformadas en series.
        """
        # Normalizamos las series
        X_norm = self.standarization(X)
        # Discretizamos
        return [self.discretization(x) for x in X_norm]  
    
    def fit_transform(self, X, y = None):
        """
            Realiza la transformación SAX a todas las series que se le pasan.
            
            Parameters
            ----------
            X : np.array
                Series temporales.
            y : None
                Por compatibilidad.
                
            Returns
            -------
            X_strings : list
                Lista con las series transformadas en series.
        """
        self.fit(X, y)
        return self.transform(X, y)
    
    def inverse_transform(self, X):
        """
            Por compatibilidad, deshace la transformación.
            
            Parameters
            ----------
            X : np.array
                Palabras discretizadas.
                
            Returns
            X : np.array
                Palabras discretizadas.
        """
        return X
    
class StringEncoder:
    """
        Clase para codificar las series
        
        Attributes
        ----------
        encoders : list
            Lista con los codificadores para cada serie.
    """
    def __init__(self):
        """
            Constructor.
        """
        self.encoders = []
        
    def fit(self, X, y = None):
        """
            Ajusta los codificadores a las palabras.
            
            Parameters
            ----------
            X : list
                Lista de palabras.
            y : None
                Por compatibilidad.
            
            Returns
            -------
            self : StringEncoder
                El objeto de la clase.
        """
        self.encoders = [LabelEncoder().fit(x) for x in X]
        return self
        
    def transform(self, X, y = None):
        """
            Transforma las palabras en etiquetas numéricas.
            
            Parameters
            ----------
            X : list
                Lista de palabras.
            y : None
                Por compatibilidad.
                
            Returns
            -------
            res : np.array
                Etiquetas transformadas.
        """
        return np.array([enc.transform(x) for x, enc in zip(X, self.encoders)])
    
    def fit_transform(self, X, y = None):
        """
            Transforma las palabras en etiquetas numéricas.
            
            Parameters
            ----------
            X : list
                Lista de palabras.
            y : None
                Por compatibilidad.
                
            Returns
            -------
            res : np.array
                Etiquetas transformadas.
        """
        self.fit(X, y)
        return self.transform(X, y)
    
    def inverse_transform(self, X):
        """
            Deshace la codificación de las palabras en etiquetas.
            
            Parameters
            ----------
            X : np.array
                Etiquetas codificadas.
            
            Returns
            -------
            res : list
                Lista de palabras.
        """
        return [enc.inverse_transform(x) for x, enc in zip(X, self.encoders)]