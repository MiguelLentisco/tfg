#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Imports
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import auc

def count_anomalies(probs, threshold):
    """
        Cuenta cuantas anomalías hay en función a la probabilidad de serlo
        y un umbral de probabilidad.
        
        Parameters
        ----------
        probs : np.numpy
            Array con probabilidades de cada serie de ser anómala
        threshold : float
            Umbral de probabilidad a partir del cual se considera anómala
            
        Returns
        -------
        n_anomalies : int
            Número de anomalías detectadas
    """
    return np.sum(probs > threshold)

def calc_recall(probs_anomalies, threshold):
    """
        Calcula la sensibilidad (recall) de un modelo en base a las
        probabilidades de las series anómalas.
        
        Parameters
        ----------
        probs_anomalies : np.numpy
            Array con probabilidades de anomalías de las series anómalas
        threshold : float
            Umbral de probabilidad
            
        Returns
        -------
        recall : float
            Sensibilidad del modelo
    """
    # Contamos el nº de verdaderos positivos
    true_positives = count_anomalies(probs_anomalies, threshold)
    return true_positives / probs_anomalies.size
    
def calc_precision(probs_normal, probs_anomalies, threshold):
    """
        Calcula la precisión de un modelo en base a las probabilidades
        de las series anómalas y normales.
        
        Parameters
        ----------
        probs_normal : np.numpy
            Array con probabilidades anomalías de las series normales
        probs_anomalies : np.numpy
            Array con probabilidades anomalías de las series anómalas
        threshold : float
            Umbral de probabilidad
            
        Returns
        -------
        precision : float
            Precisión del modelo
    """
    # Verdaderos positivos
    true_positives = count_anomalies(probs_anomalies, threshold)
    # Falsos positivos
    false_positives = count_anomalies(probs_normal, threshold)
    if true_positives == 0 and false_positives == 0:
        return 1.0
    return true_positives / (true_positives + false_positives)

    
def recall_precision_curve(X_normal, X_anomalies, model, clf_name = "clf",
                           title = "recall-precision curve", axis = None, 
                           plot = True):
    """
        Calcula la métrica PR y además muestra la curva Precision-Recall
        del modelo.
        
        Parameters
        ----------
        X_normal : np.numpy
            Series normales
        X_anomalies : np.numpy
            Series anómalas
        model : detector
            Detector de anomalías
        clf_name : str
            Nombre del detector
        title : str
            Título de la gráfica
        axis : matplotlib.axis
            Objeto para imprimir la gráfica
        plot : boolean
            Si imprimir cosas opcionales de la gráfica
        
        Returns
        -------
        pr_score : float
            Valor de la métrica PR
    """
    # Si axis no se pasa, se imprime una sola gráfica
    if axis is None:
        fig, ax = plt.subplots(1)
    else:
        ax = axis
    precisions, recalls = [], []
    # Tomamos muchas probabilidades en [0, 1]
    thresholds = np.linspace(0, 1, 2000)
    # Calculamos las probabilidades de las series normales y anómalas
    probs_normal = model.predict_prob(X_normal)
    probs_anomalies = model.predict_prob(X_anomalies)
    # Por cada umbral se calcula la precisión y sensibilidad
    for th in thresholds:
        precisions.append(calc_precision(probs_normal, probs_anomalies, th))
        recalls.append(calc_recall(probs_anomalies, th))
    # Se calcula el área debajo de la curva PR
    pr_score = auc(recalls, precisions)
    # Imprimimos la curva del modelo
    ax.plot(recalls, precisions, "-", label = clf_name + 
             " (PR = {0:0.3f})".format(pr_score))
    # Imprimir cosas opcionales
    if plot:
        # Detector aleatorio
        no_skill = X_anomalies.shape[0] / \
            (X_normal.shape[0] + X_anomalies.shape[0])
        ax.plot([0, 1], [no_skill, no_skill], "--", 
                 label = "no-skill (PR = {0:0.3f})".format(no_skill))
        # Leyenda, titulo y ejes
        ax.legend(fontsize = "small", ncol = 1)
        ax.set_title(title)
        ax.set_xlabel("recall")
        ax.set_ylabel("precision")
    # Imprimimos si no se ha pasado axis
    if axis is None:
        plt.show()
    return pr_score