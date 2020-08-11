# -----------------------------------------------------------------------------
# -------------------------------------- IMPORTS ------------------------------
# -----------------------------------------------------------------------------

import pandas as pd
import copy
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import numpy as np

# -----------------------------------------------------------------------------
# --------------------------------- FUNCIONES DE CARGA -----------------------
# ----------------------------------------------------------------------------

def load_dataset(ds_name, seed = None):
    """
        Carga el dataset con cada serie temporal estandarizada y 
        con las etiquetas. Se mezcla después de cargar.

        Parameters
        ----------
        ds_name: str
            Ruta del dataset
        seed : int
            Semilla

        Returns
        ----------
        X : np.array
            Dataset con las series temporales
        y : np.array
            Etiquetas de las series temporales
    """

    # Leemos los datos
    df = pd.read_csv(ds_name, sep=',', header = 0, index_col = 0)
    # Dataset
    X = df.iloc[:, 1:]
    # Series temporales estandarizadas
    X = X.sub(X.mean(1), axis = 0).div(X.std(1), axis = 0).fillna(0.0)
    X = X.to_numpy("float64")
    # Etiquetas codificadas
    le = preprocessing.LabelEncoder()
    y = le.fit_transform(df.iloc[:, 0].to_numpy())
    return shuffle(X, y, random_state = seed)

def load_train_test(ds_name, merge = False, seed = None):
    """
        Carga los datasets train y test. Añade shuffle

        Parameters
        ----------
        ds_name: str
            Ruta de los datasets train/test
        merge : boolean
            Si mezclar datasets y dividirlos de nuevo
        seed : int
            Semilla

        Returns
        -------
        X_train : np.array
            Dataset de entrenamiento
        y_train : np.array
            Etiquetas de entrenamiento
        X_test : np.array
            Dataset test
        y_test : np.array
            Etiquetas test
    """

    # Cargamos datasets
    X_train, y_train = load_dataset(ds_name + "_TRAIN.csv", seed)
    X_test, y_test = load_dataset(ds_name + "_TEST.csv", seed)
    # Si se juntan y dividen de nuevo
    if merge:
        X = np.vstack((X_train, X_test))
        y = np.concatenate((y_train, y_test))
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            test_size = .2,
                                                            shuffle = True,
                                                            stratify = y,
                                                            random_state = seed)
    
    return (X_train, X_test, y_train, y_test)

def load_CMFTS_TS(ds_name, ds_path = "/home/mlentisco/Datasets/", 
                  merge = False, seed = None):
    """
        Carga los datasets train y test de CMFTS y TS

        Parameters
        ----------
        ds_name : str
            Nombre del dataset
        ds_path : str
            Ruta de la carpeta de los datasets
        merge : boolean
            Si mezclar los datasets y dividirlos
        seed : int
            Semilla

        Returns
        -------
        CMFTS_ds : 4-uple
            Datasets train/test CMFTS
        TS_ds : 4-uple
            Datasets train/test TS
    """

    # Los path de cada directorio
    CMFTS_path = ds_path + "CMFTS/" + ds_name
    TS_path = ds_path + "TS/" + ds_name
    # Cargamos los datasets
    CMFTS_ds = load_train_test(CMFTS_path, merge, seed)
    TS_ds = load_train_test(TS_path, merge, seed)
    return CMFTS_ds, TS_ds

# -----------------------------------------------------------------------------
# --------------------------------- FUNCIONES AUX -----------------------------
# -----------------------------------------------------------------------------

def get_acc(X_train, X_test, y_train, y_test, clf, train_score = False):
    """
        Función auxiliar para obtener el acc_score. Se entrena el clasificador 
        clf en X_test y el se obtiene el acc en X_test.

        Parameters
        ----------
        X_train : np.array
            Dataset de entrenamiento
        X_test : np.array
            Dataset test
        y_train : np.array
            Etiquetas de entrenamiento
        y_test : np.array
            Etiquetas test
        clf : Classifier
            El clasificador

        Returns
        -------
            acc : float
                El accuracy obtenido en X_test
    """

    # Clonamos el clasificador
    clf_copy = copy.deepcopy(clf)
    # Entrenamos con las etiquetas perturbadas
    clf_copy.fit(X_train, y_train)
    # Obtenemos el acc
    if train_score:
        return clf_copy.score(X_train, y_train), clf_copy.score(X_test, y_test)    
    else:
        return clf_copy.score(X_test, y_test)
