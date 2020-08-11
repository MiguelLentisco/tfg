# ------------------------------------------------------------------------------------
# ----------------------------------- IMPORTS ----------------------------------------
# ------------------------------------------------------------------------------------

from sklearn.neighbors import KNeighborsClassifier
from sklearn.base import BaseEstimator

# ------------------------------------------------------------------------------------
# -------------------------------------- KNN -----------------------------------------
# ------------------------------------------------------------------------------------

class KNN(BaseEstimator):
    """
        Implementa el clasificador KNN (K-Nearest neighbors).

        Parameters
        ----------
        k : int
            Número de vecinos
        model : KNeighborsClassifier
            Modelo k-NN
        metric : str, metric
            Métrica que usar con KNN
        n_jobs : int
            Número de procesadores usados
    """

    def __init__(self, k = None, metric = "euclidean", n_jobs = 1):
        """
            Inicializa el clasificador.

            Parameters
            ----------
            k : int
                Número de vecinos
            metric : str, metric
                Métrica que usar con KNN
            n_jobs : int
                Número de procesadores
        """

        self.k = k
        self.model = None
        self.metric = metric
        self.n_jobs = n_jobs

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
        # Si no especifica se toma la raiz cuadrada de los datos
        if self.k is None:
            self.k = int(X.shape[0] ** .5)
        self.model = KNeighborsClassifier(n_neighbors = self.k,
                                          metric = self.metric)
        self.model.fit(X, y)

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
        return self.model.score(X, y)
