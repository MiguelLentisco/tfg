# Imports
from Utils import get_acc
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------------------------------
# -------------------------------------- PV ------------------------------------------
# ------------------------------------------------------------------------------------

class PV:
    """
        Clase que implementa Perturbation Validation (PV).

        Attributes
        ----------
        X : np.array
            Dataset
        y : np.array
            Conjuto de etiquetas perturbadas
        ds_name : str
            Nombre del dataset
        errs : np.array
            Errores tomados
        counter : int
            Contador auxiliar
        fig : Figure
            Figura actual
        ax : Axes
            Ejes actuales
    """

    def save_graph(self, name_fig):
        """
            Guarda el gráfico de los resultados en un .png

            Parameters
            ----------
            name_fig : str
                Nombre (ruta) de la imagen a guardar.
        """

        # Añadimos leyenda
        self.fig.legend(ncol = 2, fontsize = "x-small")
        # Guardamos la imagen en formato .png
        self.fig.savefig(name_fig + str(".png"), dpi = self.fig.dpi)
        plt.close(self.fig)
        # Reseteamos la imagen
        self.fig = None
        self.ax = None

    def __init__(self, X, y, n_pv = 5, ds_name = "", err_ini = 0.1, 
                 err_fin = 0.3):
        """
            Inicializa la clase creando las etiquetas perturbadas.

            Las perturbaciones se realizan tomando un %err de cada
            clase, poniendole otra etiqueta distinta.

            Se toman "n_pv" puntos entre [err_ini, err_fin].

            Parameters
            ----------
            X : np.array
                Dataset
            y : np.array
                Etiquetas
            n_pv: int
                Nº de puntos/errores
            ds_name: str
                Nombre del datases
            err_ini : float
                Error inicial
            err_fin : float
                Error final
        """

        # Dataset name
        self.ds_name = ds_name
        # Counter_clf
        self.counter = 1
        # Puntos de error
        self.errs = np.linspace(err_ini, err_fin, n_pv)
        # Dataset (NO hace copia)
        self.X = X
        # Etiquetas perturbadas
        self.y = np.empty((0, y.shape[0]), y.dtype)
        # Clases
        clases = np.unique(y)
        # Por cada %err permutamos las etiquetas
        for err in self.errs:
            y_per = np.copy(y)
            for clase in clases:
                # Indices de cada clase
                idx_clase = np.nonzero(y == clase)[0]
                # Nº de modificaciones por clase
                n_per = int(round(err * idx_clase.size))
                # Escogemos n_modif índices aleatoriamente
                idx_per = np.random.choice(idx_clase, n_per, replace = False)
                # Cogemos todas las clases excepto la actual
                clases_per = clases[np.nonzero(clases != clase)]
                # Permutamos
                y_per[idx_per] = np.random.choice(clases_per, n_per)
            # Añadimos a las etiquetas
            self.y = np.vstack((self.y, y_per))
        # Fig
        self.fig = None
        self.ax = None

    def get_pv(self, clf, clf_name = "", plot = True):
        """
            Calcula el PV score para el clasificador.

            Parameters
            ----------
            clf : Classifier
                Clasificador
            clf_name : str
                Nombre del clasificador

            Returns
            -------
            pv : float
                PV score
            accs : list(float)
                accs obtenidos
        """

        # Res accs
        accs = []
        # Obtenemos el acc con cada perturbación de etiquetas
        for y_per in self.y:
            # Obtenemos el acc del clf
            accs.append(get_acc(self.X, self.X, y_per, y_per, clf))
        # Ajustamos a una recta con mínimo error cuadrático
        poly = np.polyfit(x = self.errs, y = accs, deg = 1)
        # PV (pendiente de la recta)
        pv = abs(poly[0])
        # Guardamos la gráfica
        if plot:
            self.plot_pv(self.errs, accs, poly, pv, clf_name)
        return pv, accs

    def plot_pv(self, errs, accs, poly, pv, clf_name = ""):
        """
            Dibuja los puntos y la recta de regresión en la figura actual.

            Parameters
            ----------
            errs : np.array
                Errores
            accs : np.array
                acc obtenidos
            poly : np.array
                Recta de regresión
            pv : float
                Valor PV
            clf_name : str
                Nombre del clasificador
        """

        # Crea una figura
        if self.fig is None:
            self.fig, self.ax = plt.subplots()
            # Título del dataset si tiene
            if self.ds_name != "":
                self.ax.set_title(self.ds_name, loc = "left")
            else:
                self.ax.set_title("PV results", loc = "left")
            # Título ejes
            self.ax.set_xlabel("err")
            self.ax.set_ylabel("acc")
        # Recta ajustada
        f = poly[0] * errs + poly[1]
        # Label con clasificador, añadimos el valor PV
        if clf_name == "":
            clf_name = "clf_" + str(self.counter)
            self.counter += 1
        clf_name += ": " + str(round(pv, 2))
        # Pintamos recta ajustada
        p = self.ax.plot(errs, f, label = clf_name)
        # Pintamos los puntos (err, acc)
        self.ax.plot(errs, accs, "o", color = p[0].get_color())
