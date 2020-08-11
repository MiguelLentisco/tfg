#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Imports
import numpy as np
from scipy.signal import gaussian, gausspulse
from statsmodels.tsa.seasonal import seasonal_decompose

# ------------------------------------------------------------------------
# ---------------------- FUNCIONES AUXILIARES ----------------------------
# ------------------------------------------------------------------------

def random_slice(x, max_length = None, min_length = None,
                   length = None, pos = None, border = 0):
    """
        Se encarga de elegir un tramo aleatorio de una serie que queda
        determinado por una posición y longitud, de manera que el tramo
        elegido es [posición, posición + longitud).
    
        Se puede determinar una longitud máxima o mínima, o incluso
        especificar una longitud o posición fijada. También se puede
        indicar si excluir los extremos (añadir borde).
        
        Parameters
        ----------
        x : np.numpy
            Serie temporal que alterar
        max_length : int, None
            Longitud máxima de la perturbación
        min_length : int, None
            Longitud mínima de la perturbación
        length : int, None
            Longitud fija de la perturbación
        pos : int, None
            Posición fija de la perturbación
        border : int
            Borde para excluir la perturbación
            
        Returns
        -------
        pos : int
            Posición de la perturbación
        length : int
            Longitud de la perturbación
    
    """
    # Longtiud mínima de 3
    if min_length is None:
        min_length = 3
    # Longitud de la serie
    x_len = x.shape[0]
    # Longitud de la perturbación
    if length is None:
        max_l = 1 + min(max_length, x_len - 2 * border)
        length = int(np.random.randint(min_length, max_l, 1))
    # Posición donde empieza la perturbación
    if pos is None:
        max_l = 1 + x_len - max(border, length)
        pos = int(np.random.randint(border, max_l, 1))
    return pos, length

# ------------------------------------------------------------------------
# -------------------------- ALTERACIONES --------------------------------  
# ------------------------------------------------------------------------
   
def gaussian_noise(x, max_length, min_length = 3, std = 3, neg = False, 
                   border = 0, neg_random = True):
    """
        Crea una perturbación de ruido gaussiano añadiendo en un
        tramo aleatorio un muestreo de la función de densidad normal.
        Se puede controlar la intensidad de esta.
        
        Además se puede activar aleatoriamente (50%) o de manera fija que la 
        alteración gaussiana sea negativa.
        
        Parameters
        ----------
        x : np.numpy
            La serie para alterar
        max_length : int
            Longitud máxima de la alteración
        min_length : int
            Longitud minima de la alteración
        std : float
            Controla la intensidad de la alteración
        neg : boolean
            Si invertir la señal gaussiana
        border : int
            El borde para excluir la perturbación
        neg_random : boolean
            Si se invierte aleatoriamente las señales
            
        Returns
        -------
        x : np.numpy
            Una copia de la señal perturbada
    """
    # Copiamos la serie
    x = np.copy(x)
    # Obtenemos un trmao aleatorio
    pos, length = random_slice(x, max_length = max_length, 
                               min_length = min_length, border = border)
    # Muestreamos la gaussiana
    noise = gaussian(length, std = length / 6) * std
    # Si se indica, se invierte la gaussiana
    if neg or neg_random and int(np.random.randint(0, 2, 1)) == 1:
        noise = -noise
    # Alteramos añadiendo el ruido
    x[pos:(pos + length)] += noise
    return x
    
def gaussian_sine_pulse(x, max_length, min_length = 3, fc = 1.5, std = 3, 
                        border = 0):
    """
        Crea una perturbación con un pulso sinusoidal-gaussiano añadido en un
        tramo aleatorio. Se puede controlar la intensidad y la frecuencia
        del pulso.
        
        Parameters
        ----------
        x : np.numpy
            La serie para alterar
        max_length : int
            Longitud máxima de la alteración
        min_length : int
            Longitud minima de la alteración
        fc : float
            Frecuencia de la señal del pulso
        std : float
            Controla la intensidad de la alteración
        border : int
            El borde para excluir la perturbación
            
        Returns
        -------
        x : np.numpy
            Una copia de la señal perturbada
    """
    # Copiamos la serie
    x = np.copy(x)
    # Cogemos un tramo aleatorio
    pos, length = random_slice(x, max_length = max_length, 
                               min_length = min_length , border = border)
    # Alteramos añadiendo el pulso sinusoidal-gaussiano
    x[pos:(pos + length)] += \
            gausspulse(np.linspace(-1, 1, length), fc = fc) * std
    return x

def modify_season(x, period, max_length, min_length = 3, std = 1, border = 0):
    """
        Crea una perturbación multiplicando por un real la estacionalidad
        de un tramo aleatorio de la serie. Se necesita el periodo para
        realizar la descomposición STL.
        
        Parameters
        ----------
        x : np.numpy
            La serie para alterar
        period : int
            Periodo de repetición de la serie para descomposición STL
        max_length : int
            Longitud máxima de la alteración
        min_length : int
            Longitud minima de la alteración
        std : float
            Controla la intensidad de la alteración
        border : int
            El borde para excluir la perturbación
    
    """
    # Descomposición STL
    x = seasonal_decompose(x, period = period, model = "additive",
                           extrapolate_trend = "freq")
    # Copiamos la estacionalidad
    new_season = np.copy(x.seasonal)
    # Cogemos un tramo aleatorio
    pos, length = random_slice(new_season, max_length = max_length,
                                min_length = min_length, border = border)
    # Alteramos la estacionalidad
    new_season[pos:(pos + length)] *= std
    return new_season + x.trend + x.resid

def modify_trend(x, period, max_length, min_length = 3, std = 1, border = 0):
    """
        Crea una perturbación multiplicando por un real la tendencia
        de un tramo aleatorio de la serie. Se necesita el periodo para
        realizar la descomposición STL.
        
        Parameters
        ----------
        x : np.numpy
            La serie para alterar
        period : int
            Periodo de repetición de la serie para descomposición STL
        max_length : int
            Longitud máxima de la alteración
        min_length : int
            Longitud minima de la alteración
        std : float
            Controla la intensidad de la alteración
        border : int
            El borde para excluir la perturbación
    
    """
    # Descomposición STL
    x = seasonal_decompose(x, period = period, model = "additive",
                           extrapolate_trend = "freq")
    # Copiamos la tendencia
    new_trend = np.copy(x.trend)
    # Cogemos un tramo aleatorio
    pos, length = random_slice(new_trend, max_length = max_length,
                                min_length = min_length, border = border)
    # Alteramos la tendencia
    new_trend[pos:(pos + length)] *= std
    return new_trend + x.seasonal + x.resid