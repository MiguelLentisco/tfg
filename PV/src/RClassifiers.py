    # -------------------------------------------------------------------------
# -------------------------------------- IMPORTS ------------------------------
# -----------------------------------------------------------------------------

import rpy2.robjects as ro
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()
from rpy2.robjects.packages import importr, SignatureTranslatedAnonymousPackage
import numpy as np
from sklearn.base import BaseEstimator

# R imports
d = {'package.dependencies': 'package_dot_dependencies',
     'package_dependencies': 'package_uscore_dependencies'}
baselib = importr("base")
statslib = importr("stats")
# C5.0/C5.0 + boosting package
C50lib = importr("C50", robject_translations = d)
# rpart package
rpartlib = importr('rpart', robject_translations = d)
# cpart package
partykitlib = importr("partykit")
# C4.5 package
RWekalib = importr("RWeka")

# Seed
ro.r("set.seed(123)")

# -----------------------------------------------------------------------------
# ---------------------------------- FUNCIONES AUX ----------------------------
# -----------------------------------------------------------------------------

def data_to_R(X, y, join = False):
    """
        Convierte los datos de python (numpy) en Dataframes de R.

        Parameters
        ----------
        X : numpy.array
            Datos
        y : numpy.array
            Etiquetas
        join: boolean, optional (default = False)
            Si unir en un solo dataframe

        Returns
        ----------
        X_R : Dataframe (si join = False)
            Dataframe de X
        y_R : Dataframe (si join = False)
            Factor Int
        X_y_R : Dataframe (si join = True)
            Dataframe con X e Y
    """

    y_R = ro.FactorVector(ro.vectors.IntVector(y))
    if join:
        X_y_R = ro.DataFrame({"Clase": y_R, "TS": X})
        return X_y_R
    X_R = ro.DataFrame({"TS": X})
    return X_R, y_R

# -----------------------------------------------------------------------------
# ---------------------------------- CLASIFICADORES ---------------------------
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# ------------------------------------ KNN+DTW --------------------------------
# -----------------------------------------------------------------------------

# Función en R que implementa KNN+DTW
ro.r("""library("IncDTW")
# KNN+DTW
dtwKNN <- function(train, test, k, ws) {
    # Conversión de tipos
    target = vector(mode = "list", length = nrow(train))
    for (i in 1:nrow(train)) {
        target[[i]] = c(as.numeric(train[i, -1]))
    }

    # Para cada instancia del test se aplica un dtw+knn sobre el conjunto de 
    # entrenamiento
    apply(test, 1, function(t) {
        query = as.vector(t)
        # Obtenemos los índices de los KNN usando DTW
        res = rundtw(query, target, dist_method = "norm1", ws = ws, k = k)
        knn_indices = c(res$knn_list_indices)
        # K-nn del punto
        clases = train[knn_indices, 1]
        # Moda de las clases
        uniqv = unique(clases)
        uniqv[which.max(tabulate(match(clases, uniqv)))]
    })
}
""")
dtwKNN = ro.globalenv["dtwKNN"]

class DTW(BaseEstimator):
    """
        Clase que implementa K-Nearest Neighbors con la distancia DTW
        usando la implementación del paquete "IncDTW".

        Attributes
        ----------
        data : R.DataFrame
            Datos transformados en un objeto dataframe de R
        k : int
            Números de vecinos
        window_shift : int
            Tamaño de la ventana para aplicar DTW
    """

    def __init__(self, k = 1, window_shift = 5):
        """
            Constructor de la clase, debe hacerse solo una vez por dataset.

            Parameters
            ----------
            k : int
                Números de vecinos
            window_shift : int
                Tamaño de la ventana para aplicar DTW
        """

        self.data = None
        self.k = k
        self.window_shift = window_shift

    def fit(self, X, y):
        """
            Con fit solo necesitamos guardar los datos de entrenamiento.

            Parameters
            ----------
            X : numpy.array
                Datos de entrenamiento
            y : numpy.array
                Etiquetas de entrenamiento
        """

        self.data = data_to_R(X, y, join = True)

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

        X_R, _ = data_to_R(X, y)
        # Ejecuta KNN+DTW
        y_pred = dtwKNN(self.data, X_R, k = self.k,
                        ws = self.window_shift)
        return np.mean((np.asarray(y_pred)-1) == y)


# -----------------------------------------------------------------------------
# ---------------------------- C5.0 / C5.0 Boosting ---------------------------
# -----------------------------------------------------------------------------

class C50(BaseEstimator):
    """
        Implementa el árbol de decisión C5.0 (con boosting o no).

        Attributes
        ----------
        model : clasificador en R
            El clasificador (clase en R)
        boosting : int
            El valor del boosting
    """

    def __init__(self, boosting = 10):
        """
            Inicializa el clasificador.

            Parameters
            ----------
            boosting : int
                El valor del boosting
        """

        self.model = None
        self.boosting = boosting

    def fit(self, X, y):
        """
            Entrena el modelo.

            Parameters
            ----------
            X : numpy.array
                Datos de entrenamiento
            y : numpy.array
                Etiquetas de entrenamiento
        """

        # Conversión a R
        X_R, y_R = data_to_R(X, y)
        # Entrenamos
        self.model = C50lib.C5_0(x = X_R, y = y_R, trials = self.boosting)


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

        # Conversión a R
        X_R, _ = data_to_R(X, y)
        # Predicción
        y_pred = C50lib.predict_C5_0(self.model, newdata = X_R, type = "class")
        # acc
        return np.mean((np.asarray(y_pred)-1) == y)

# -----------------------------------------------------------------------------
# -------------------------------------- RPART --------------------------------
# -----------------------------------------------------------------------------

class RPart(BaseEstimator):
    """
        Implementa el árbol de decisión RPart (Recursive Partioning Tree).

        Attributes
        ----------
        model : clasificador en R
            El clasificador (clase en R)
    """

    def __init__(self):
        """
            Inicializa el clasificador.
        """

        self.model = None

    def fit(self, X, y):
        """
            Entrena el modelo.

            Parameters
            ----------
            X : numpy.array
                Datos de entrenamiento
            y : numpy.array
                Etiquetas de entrenamiento
        """

        # Conversión a R
        data = data_to_R(X, y, join = True)
        # Fórmula para predecir
        formula = ro.Formula("Clase ~.")
        # Entrenamos
        self.model = rpartlib.rpart(formula = formula, data = data, 
                                    method = "class")

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

        # Conversión a R
        X_R, _ = data_to_R(X, y)
        # Predicción
        y_pred = rpartlib.predict_rpart(self.model, newdata = X_R, 
                                        type = "class")
        # acc
        return np.mean((np.asarray(y_pred)-1) == y)

# -----------------------------------------------------------------------------
# -------------------------------------- CTREE --------------------------------
# -----------------------------------------------------------------------------

class CTree(BaseEstimator):
    """
        Implementa el árbol de decisión CTree (Conditional Inference Tree).

        Attributes
        ----------
        model : clasificador en R
            El clasificador (clase en R)
    """

    def __init__(self):
        """
            Inicializa el clasificador.
        """

        self.model = None

    def fit(self, X, y):
        """
            Entrena el modelo.

            Parameters
            ----------
            X : numpy.array
                Datos de entrenamiento
            y : numpy.array
                Etiquetas de entrenamiento
        """

        # Conversión a R
        data = data_to_R(np.round(X, 15), y, join = True)
        # Fórmula para predecir
        formula = ro.Formula("Clase ~.")
        # Entrenamos
        self.model = partykitlib.ctree(formula = formula, data = data)

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

        # Conversión a R
        X_R, _ = data_to_R(X, y)
        # Predecimos
        y_pred = statslib.predict(self.model, newdata = X_R, type = "response")
        # acc
        return np.mean((np.asarray(y_pred)-1) == y)


# -----------------------------------------------------------------------------
# -------------------------------------- C4.5 ---------------------------------
# -----------------------------------------------------------------------------

class C45(BaseEstimator):
    """
        Implementa el árbol de decisión C4.5.

        Attributes
        ----------
        model : clasificador en R
            El clasificador (clase en R)
    """

    def __init__(self):
        """
            Inicializa el clasificador.
        """

        self.model = None

    def fit(self, X, y):
        """
            Entrena el modelo.

            Parameters
            ----------
            X : numpy.array
                Datos de entrenamiento
            y : numpy.array
                Etiquetas de entrenamiento
        """

        # Conversión a R
        data = data_to_R(X, y, join = True)
        # Fórmula para predecir
        formula = ro.Formula("Clase ~.")
        # Entrenamos
        self.model = RWekalib.J48(formula = formula, data = data)

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

        # Conversión a R
        X_R, _ = data_to_R(X, y)
        # Predicciones
        y_pred = statslib.predict(self.model, newdata = X_R, type = "class")
        # acc
        return np.mean((np.asarray(y_pred)-1) == y)
