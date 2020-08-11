# ------------------------------------------------------------------------------------
# -------------------------------------- IMPORTS -------------------------------------
# ------------------------------------------------------------------------------------

# Utilidades varias
from Utils import load_CMFTS_TS, get_acc
import numpy as np
import pandas as pd
import sys

# PV / CV
from PV import PV
from sklearn.model_selection import cross_val_score

# Classifiers
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from LSTM import LSTM
from RClassifiers import DTW, C50, RPart, CTree, C45
from KNN import KNN

# Seed
SEED = 123
np.random.seed(SEED)

# Umbrales filtro datasets
TAM_MIN = 100
CLAS_MIN = 5

# Rutas
#RUTA_DS =  "../Datasets/"
RUTA_IMG = "res/img/"
RUTA_CSV = "res/csv/"

# Print output
PRINT_OUTPUT = True

# -----------------------------------------------------------------------------
# ------------------------------ Funciones aux --------------------------------
# -----------------------------------------------------------------------------

def clases_balanceadas(y):
    """
        Devuelve si hay una clase cuya proporción sea mayor al 70%.
        
        Parameters
        ----------
        y : numpy.array
            Etiquetas
        
        Returns
        -------
        is_balanced : boolean
            Si hay más o menos equilibrio entre las clases
    """
    props = [np.sum(y == clase) / y.shape[0] for clase in np.unique(y)]
    return max(props) <= 0.7

def minimo_clases(y, clas_min):
    """
        Devuelve si hay al menos un número mínimo de etiquetas por cada clase.
        
        Parameters
        ----------
        y : numpy.array
            Etiquetas
        clas_min : int
            Número mínimo de etiquetas por clase
        
        Returns
        -------
        is_minimum : boolean
            Si hay al menos clas_min etiquetas por clase
    """
    clases = np.unique(y)
    for clase in clases:
        n_clase = np.sum(y == clase)
        if n_clase < clas_min: return False
    return True

def dataset_no_valido(X, y, tam_min, clas_min):
    """
        Devuelve si un dataset no es válido: si no tiene un tamaño mínimo,
        o un número mínimo de etiquetas por clase o no está balanceado
        
        Parameters
        ----------
        X : numpy.array
            Dataset series
        y : numpy.array
            Etiquetas
        tam_min : int
            Tamaño mínimo del dataset
        clas_min : int
            Número mínimo de etiquetas por clase
            
        Returns
        -------
        is_valid : boolean
            Si el dataset es válido o no
    """
    return X.shape[0] < tam_min or not minimo_clases(y, clas_min) or \
        not clases_balanceadas(y)

# -----------------------------------------------------------------------------
# ------------------------------- Experimento ---------------------------------
# -----------------------------------------------------------------------------

def main():
    # Solo con la ruta se ejecuta con todos los datasets
    if len(sys.argv) == 2:
        # Ruta Datasets
        RUTA_DS = sys.argv[1]
        # Se ejecutan todos los datset
        n_dataset_ini = 0
        n_dataset_fin = None
    # Si se pasa con un nº, se indica el nº del dataset concreto
    elif len(sys.argv) == 3:
        # Ruta Datasets
        RUTA_DS = sys.argv[1]
        # Se ejecuta un dataset
        n_dataset_ini = int(sys.argv[2])
        n_dataset_fin = n_dataset_ini + 1
    # Si se pasa con un intervalo de datasets
    elif len(sys.argv) == 4:
        # Ruta Datasets
        RUTA_DS = sys.argv[1]
        # Se ejecuta un intervalo de datasets
        n_dataset_ini = int(sys.argv[2])
        n_dataset_fin = int(sys.argv[2]) + 1
    elif len(sys.argv) > 4:
        sys.exit("Error, nº de parámetros incorrecto.\n" +
            "Uso: python3 experimento_PV.py CAPERTA_DATASETS [Nº DATASET] " +
            "[Nº DATASET FIN]")
    else:
        sys.exit("Error.")

    # Cargamos los nombres de los datasets usando el .csv con los nombres
    ds_names = pd.read_csv(RUTA_DS + "ListadoDatasets_TS.csv", header = 0, 
                           index_col = 0)
    # Cargamos los datasets indicados
    ds_names = list(ds_names.index.values)[n_dataset_ini:n_dataset_fin]

    # Clasificadores
    clfs = [(SVC(gamma = "auto"), "RBF-SVM"),
            (RandomForestClassifier(n_estimators = 200, max_depth = 20,
                                    random_state = SEED,
                                    n_jobs = -1), "Random Forest"),
            (DecisionTreeClassifier(max_depth = 20,
                                    random_state = SEED), "CART"),
            (C45(), "C4.5"),
            (C50(boosting = 1), "C5.0"),
            (C50(boosting = 10), "C5.0-Boosting"),
            (CTree(), "CTree"),
            (RPart(), "RPart"),
            (KNN(metric = "euclidean", n_jobs = -1), "kNN"),
            (LSTM(epochs = 300, verbose = 0), "LSTM"),
            (DTW(n_neighbors = 3, window_shift = 5), "3NN+DTW")]

    loaded_ds = []
    # Cargamos los datasets
    if PRINT_OUTPUT: print("Cargando datasets.", flush = True)
    for ds_name in ds_names:
        CMFTS_ds, TS_ds = load_CMFTS_TS(ds_name, RUTA_DS, True, SEED)
        loaded_ds.append((ds_name, TS_ds, CMFTS_ds))
    if PRINT_OUTPUT: print("Datasets cargados.\n", flush = True)

    # Por cada dataset aplicamos cada clasificador
    for (ds_name, TS_ds, CMFTS_ds) in loaded_ds:
        # Filtro datasets
        if dataset_no_valido(TS_ds[0], TS_ds[2]): continue

        # Creamos los datasets perturbados para PV
        TS_pv = PV(TS_ds[0], TS_ds[2], 5, ds_name)
        CMFTS_pv = PV(CMFTS_ds[0], CMFTS_ds[2], 5, ds_name)
        TS_ds_result = []
        CMFTS_ds_result = []
        if PRINT_OUTPUT: print("DS: " + ds_name, flush = True)

        # Para cada clasificador aplicamos las métricas PV/ACC-CV/ACC-TEST
        for (clf, clf_name) in clfs:
            # Añadimos n_clases a LSTM
            if "LSTM" in clf_name:
                etiquetas_full = np.concatenate((np.unique(TS_ds[2]), 
                                                 np.unique(TS_ds[3])))
                clf.n_clases = len(np.unique(etiquetas_full))
            if PRINT_OUTPUT: print("\tClassifier: " + clf_name, flush = True)

            # Obtenemos PV_score
            TS_pv_score, TS_accs = TS_pv.get_pv(clf, clf_name)
            TS_pv_score = round(TS_pv_score, 4)
            if PRINT_OUTPUT: print("\t\tTS-PV: " + str(TS_pv_score), 
                                   flush = True)
            CMFTS_pv_score, CMFTS_accs = CMFTS_pv.get_pv(clf, clf_name)
            CMFTS_pv_score = round(CMFTS_pv_score, 4)
            if PRINT_OUTPUT: print("\t\tCMFTS-PV: " + str(CMFTS_pv_score), 
                                   flush = True)

            # Obtenemos acc_CV
            TS_acc_cv = round(np.mean(cross_val_score(clf, TS_ds[0], TS_ds[2], 
                                                      cv = 5)), 4)
            if PRINT_OUTPUT: print("\t\tTS-acc-CV: " + str(TS_acc_cv), 
                                   flush = True)
            CMFTS_acc_cv = round(np.mean(cross_val_score(clf, CMFTS_ds[0], 
                                                         CMFTS_ds[2], 
                                                         cv = 5)), 4)
            if PRINT_OUTPUT: print("\t\tCMFTS-acc-CV: " + str(CMFTS_acc_cv), 
                                   flush = True)

            # Obtenemos acc_test
            TS_acc_train, TS_acc_test = get_acc(*TS_ds, clf, True)
            TS_acc_train = round(TS_acc_train, 4) 
            TS_acc_test = round(TS_acc_test, 4)
            if PRINT_OUTPUT: print("\t\tTS-acc-train: " + str(TS_acc_train), 
                                   flush = True)
            if PRINT_OUTPUT: print("\t\tTS-acc-test: " + str(TS_acc_test), 
                                   flush = True)
            CMFTS_acc_train, CMFTS_acc_test = get_acc(*CMFTS_ds, clf, True)
            CMFTS_acc_train = round(CMFTS_acc_train, 4)
            CMFTS_acc_test = round(CMFTS_acc_test, 4)
            if PRINT_OUTPUT: print("\t\tCMFTS-acc-train: " + 
                                   str(CMFTS_acc_train), flush = True)
            if PRINT_OUTPUT: print("\t\tCMFTS-acc-test: " + 
                                   str(CMFTS_acc_test), flush = True)
            if PRINT_OUTPUT: print("", flush = True)

            # Guardamos resultados
            TS_ds_result.append([clf_name, TS_pv_score, TS_acc_train, 
                                 TS_acc_cv, TS_acc_test] +
                                [(acc, 4) for acc in TS_accs])
            CMFTS_ds_result.append([clf_name, CMFTS_pv_score, CMFTS_acc_train, 
                                    CMFTS_acc_cv, CMFTS_acc_test] +
                                   [(acc, 4) for acc in CMFTS_accs])
        if PRINT_OUTPUT: print("", flush = True)

        # Guardamos las imágenes de TS
        TS_pv.save_graph(RUTA_IMG + "TS/" + ds_name)
        CMFTS_pv.save_graph(RUTA_IMG + "CMFTS/" + ds_name)

        # Guardamos los datos en .csv
        column_names = ["Classifier", "PV", "ACC_TRAIN", "ACC_CV", 
                        "ACC_TEST", "PV_1", "PV_2", "PV_3", "PV_4", "PV_5"]
        TS_df = pd.DataFrame(TS_ds_result, columns = column_names)
        CMFTS_df = pd.DataFrame(CMFTS_ds_result, columns = column_names)
        TS_df.to_csv(RUTA_CSV + "TS/" + ds_name + ".csv", index = False)
        CMFTS_df.to_csv(RUTA_CSV + "CMFTS/" + ds_name + ".csv", index = False)

main()
