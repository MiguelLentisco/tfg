# ------------------------------------------------------------------------------------
# -------------------------------------- IMPORTS -------------------------------------
# ------------------------------------------------------------------------------------

# Utilidades varias
from Utils import load_CMFTS_TS, get_acc
import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt

# PV / CV
from PV import PV
from sklearn.model_selection import cross_val_score

# Classifiers
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
#from sklearn.tree import DecisionTreeClassifier
#from LSTM import LSTM
#from RClassifiers import DTW, C50, RPart, CTree
#from KNN_auto import KNN

# Seed
SEED = 123
np.random.seed(SEED)

# Umbrales filtro datasets
TAM_MIN = 100
CLAS_MIN = 5

# Rutas
#RUTA_DS =  "../Datasets/"
#RUTA_IMG = "/home/mlentisco/src/res/img/"


# Print output
PRINT_OUTPUT = True

# ------------------------------------------------------------------------------------
# --------------------------------- Funciones aux ------------------------------------
# ------------------------------------------------------------------------------------

def clases_balanceadas(y):
    props = [np.sum(y == clase) / y.shape[0] for clase in np.unique(y)]
    return max(props) <= 0.7


# Si hay al menos CLAS_MIN datos de cada clase
def minimo_clases(y):
    clases = np.unique(y)
    for clase in clases:
        n_clase = np.sum(y == clase)
        if n_clase < CLAS_MIN: return False
    return True

# Si un dataset no pasa el filtro de tamaño mínimo,
# nº de elementos por clases mínimo o balanceo de clases
def dataset_no_valido(X, y):
    return X.shape[0] < TAM_MIN or not minimo_clases(y) or \
        not clases_balanceadas(y)

# ------------------------------------------------------------------------------------
# ----------------------------------- Experimento ------------------------------------
# ------------------------------------------------------------------------------------

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
        n_dataset_fin = int(sys.argv[2])
    elif len(sys.argv) > 4:
        sys.exit("Error, nº de parámetros incorrecto.\n" +
            "Uso: python3 experimento_PV.py CAPERTA_DATASETS [Nº DATASET] [Nº DATASET FIN]")
    else:
        RUTA_DS = "../../Datasets/"
        n_dataset_ini = 92
        n_dataset_fin = n_dataset_ini + 1

    # Cargamos los nombres de los datasets usando el .csv con los nombres
    ds_names = pd.read_csv(RUTA_DS + "ListadoDatasets_TS.csv", header = 0, index_col = 0)
    # Cargamos los datasets indicados
    #ds_names = list(ds_names.index.values)[n_dataset_ini:n_dataset_fin]
    ds_names = list(ds_names.index.values)
    ds_names = [ds_names[10], ds_names[77], ds_names[90], ds_names[80]]

    # Clasificadores
    """
    clfs = [(SVC(gamma = "auto"), "RBF-SVM"),
            (RandomForestClassifier(n_estimators = 200, max_depth = 20,
                                    random_state = SEED,
                                    n_jobs = -1), "Random Forest"),
            (DecisionTreeClassifier(max_depth = 20,
                                    random_state = SEED), "CART"),
            #(C45(), "C4.5"),
            (C50(boosting = 1), "C5.0"),
            (C50(boosting = 10), "C5.0-Boosting"),
            (CTree(), "CTree"),
            (RPart(), "RPart"),
            (KNN(metric = "euclidean", n_jobs = -1), "kNN"),
            #(LSTM(epochs = 200, verbose = 0), "LSTM"),
            (DTW(n_neighbors = 3, window_shift = 5), "3NN+DTW")]
    """
    
    params = {"RF": range(1, 15), "SVM": np.logspace(-4, 4, 15)}
    clfs = [("RF", [RandomForestClassifier(n_estimators = 100,
                                           max_depth = param)
                    for param in params["RF"]]),
            ("SVM", [SVC(C = c, kernel = "rbf", gamma = "auto",
                         random_state = SEED) for c in params["SVM"]])]
    loaded_ds = []
    # Cargamos los datasets
    if PRINT_OUTPUT: print("Cargando datasets.", flush = True)
    for ds_name in ds_names:
        CMFTS_ds, TS_ds = load_CMFTS_TS(ds_name, RUTA_DS, False, SEED)
        loaded_ds.append((ds_name, TS_ds, CMFTS_ds))
    if PRINT_OUTPUT: print("Datasets cargados.\n", flush = True)

    n_clfs = set()
    pvs = {}
    cvs = {}
    tests = {}
    
    # Por cada dataset aplicamos cada clasificador
    for (ds_name, TS_ds, CMFTS_ds) in loaded_ds:
        # Filtro datasets
        if dataset_no_valido(TS_ds[0], TS_ds[2]): continue

        # Creamos los datasets perturbados para PV
        TS_pv = PV(TS_ds[0], TS_ds[2], 5, ds_name)
        #CMFTS_pv = PV(CMFTS_ds[0], CMFTS_ds[2], 5, ds_name)
        if PRINT_OUTPUT: print("DS: " + ds_name, flush = True)

        # Para cada clasificador aplicamos las métricas PV/ACC-CV/ACC-TEST
        for (clf_name, clf_h) in clfs:
            if PRINT_OUTPUT: print("\t" + clf_name + ":", flush = True)
            if PRINT_OUTPUT: print("\t\tCalculando PV...", flush = True)
            TS_pvs = [1 - abs(TS_pv.get_pv(clf, plot = False)[0] - 1) 
                        for clf in clf_h]
            #CMFTS_pvs = [CMFTS_pv.get_pv(clf, plot = False)[0] 
            #                 for clf in clf_h]
            if PRINT_OUTPUT: print("\t\tCalculando CV...", flush = True)
            TS_accs_cv = [np.mean(cross_val_score(clf, TS_ds[0], TS_ds[2], 
                                                  cv = 5)) for clf in clf_h]
            #CMFTS_accs_cv = [np.mean(cross_val_score(clf, CMFTS_ds[0],
            #                                         CMFTS_ds[2], cv = 5))
            #                    for clf in clf_h]
            if PRINT_OUTPUT: print("\t\tCalculando TEST...", flush = True)
            TS_accs_test = [get_acc(*TS_ds, clf) for clf in clf_h]
            #CMFTS_accs_test = [get_acc(*CMFTS_ds, clf) for clf in clf_h]

            n_clfs.add(clf_name)
            pvs.setdefault(clf_name, []).append(TS_pvs)
            cvs.setdefault(clf_name, []).append(TS_accs_cv)
            tests.setdefault(clf_name, []).append(TS_accs_test)
    
    for n_clf in n_clfs:
        fig, axs = plt.subplots(2, 2, figsize = (16, 14))
        for pv_res, cv_res, test_res, axis, ds_name in zip(pvs[n_clf], 
                                                           cvs[n_clf], 
                                                           tests[n_clf], 
                                                           axs.flat,
                                                           ds_names):
            axis.plot(params[n_clf], pv_res, "-o", label = "PV")
            axis.plot(params[n_clf], cv_res, "-o", label = "acc-cv")
            axis.plot(params[n_clf], test_res, "-o", label = "acc-test")
            if "SVM" in n_clf:
                axis.set_xscale("log")
            axis.set_title(ds_name)
            axis.legend()
        plt.suptitle(n_clf)
        fig.savefig(n_clf + str(".png"))
        plt.close(fig)
        

main()
