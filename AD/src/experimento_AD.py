# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing

# Alteraciones
from alteraciones import gaussian_noise, gaussian_sine_pulse, modify_season, \
                         modify_trend
# Detector
from detector import LSTM_AD
# Cálculo del PR
from calc_pr import recall_precision_curve
from sklearn.model_selection import train_test_split

# Semillas
SEED = 42
np.random.seed(SEED)


def load_dataset(ds_name):
    """
        Carga el dataset con cada serie temporal estandarizada y
        con las etiquetas.

        Parameters
        ----------
        ds_name: str
            Ruta del dataset

        Returns
        -------
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
    return X, y

def merge_datasets(X1, X2, y1, y2):
    """
        Junta dos datasets en uno solo.

        Parameters
        ----------
        X1 : np.numpy
            Series 1
        X2 : np.numpy
            Series 2
        y1 : np.numpy
            Etiquetas 1
        y2:  np.numpy
            Etiquetas 2

        Returns
        -------
        X : np.numpy
            Unión de X1 y X2
        y : np.numpy
            Unión de y1 e y2
    """
    X = np.vstack((X1, X2))
    y = np.concatenate((y1, y2))
    return X, y

def print_dataset(X, y = None, ds_name = ""):
    """
        Imprime las series temporales del dataset X.

        Parameters
        ----------
        X : np.numpy
            Dataset con las series
        y : np.numpy
            Etiquetas
        ds_name : str
            Nombre del dataset

    """
    fig, ax = plt.subplots(1, 1, figsize = (10, 6))
    if y is not None:
        colors = ["red", "blue"]
        for c in np.unique(y):
            for ts in X[y == c]:
                ax.plot(ts, color = colors[c])
            ax.plot([], [], color = colors[c], label = c)
        ax.legend(title = "Clase")
    else:
        for ts in X:
            ax.plot(ts, color = "blue")

    ax.set_title("Dataset " + ds_name)
    fig.savefig("../doc/img/dataset-two.png")
    plt.show()

def reshape_data(X):
    """
        Función auxiliar para ajustar las dimensiones de los datos
        según convenga.

        Parameters
        ----------
        X : numpy.array
            Series temporales

        Returns
        -------
        X : numpy.array
            Series temporales con otra dimensión
    """
    if len(X.shape) == 2:
        return X.reshape((X.shape[0], X.shape[1], 1))
    elif len(X.shape) == 3:
        return X.reshape((X.shape[0], X.shape[1]))
    elif len(X.shape) == 1:
        return X.reshape((1, X.size, 1))

def main():
    USE_COS = True
    SHOW_DATASET = False
    TRAIN_MODEL = False
    SHOW_EXAMPLES = False
    SAVE_MODEL_IMG = False
    SHOW_HISTS = False
    SHOW_ANOMALIES = False
    CALC_PR = True
    CALC_PR_AN = False

    # Usar dataset coseno
    if USE_COS:
        MODEL_SAVE = "../models/cos.h5"
        MODEL_LOAD = "../models/cos.h5"
        X = []
        T = 4
        # Muestreamos el coseno en 100 puntos
        cos_sample = np.cos(np.linspace(-2*np.pi*T, 2*np.pi*T, 100))
        # Añadimos tendencia
        cos_sample += np.linspace(0, 1, 100)
        #cos_sample = (cos_sample - np.mean(cos_sample)) / np.std(cos_sample)
        # 200 muestras
        for i in range(200):
            # RUido gaussiano
            x_aux = cos_sample + np.random.normal(0, 0.3, 100)
            # Estandarización
            x_aux = (x_aux - np.mean(x_aux)) / np.std(x_aux)
            X.append(x_aux)
        X = np.array(X)
        DS_NAME = "Dataset Cosenos"
    else:
        DATASET = "../../Datasets/TS/TwoLeadECG"
        MODEL_SAVE = "../models/twomodel.h5"
        MODEL_LOAD = "../models/twomodel.h5"
        # Mergeamos los datasets
        X1, y1 = load_dataset(DATASET + "_TRAIN.csv")
        X2, y2 = load_dataset(DATASET + "_TEST.csv")
        X, y = merge_datasets(X1, X2, y1, y2)
        # El resto de clases como anomalias
        X_anomalos = X[y != 0]
        # Nos quedamos con la clase normal
        X = X[y == 0]
        # Dividimos 80/20 las anomalias
        X_anomalos_train, X_anomalos_test = train_test_split(X_anomalos,
                                                         test_size = 0.2,
                                                         random_state = SEED)
        X_anomalos_train, X_anomalos_test = reshape_data(X_anomalos_train), \
                                            reshape_data(X_anomalos_test)
        DS_NAME = "TwoLeadECG (Clase 0)"
    # División 80/20
    X_train, X_test = train_test_split(X, test_size = 0.2, random_state = SEED)
    X_train, X_test = reshape_data(X_train), reshape_data(X_test)

    # Mostrar series
    if SHOW_DATASET:
        print_dataset(X_train[:10], ds_name = DS_NAME)

    model = LSTM_AD(n_neur = 10, epochs = 400, mode = 2, alpha = 0)
    # Entrenar modelo
    if TRAIN_MODEL:
        model.fit(X_train)
        model.plot_historial()
        model.save_model(MODEL_SAVE)
    # Cargamos modelos
    else:
        model.load_model(MODEL_LOAD)
        model.fit_kernel(X_train)

    # Mostrar histogramas de error reconstrucción
    if SHOW_HISTS:
        fig, axs = plt.subplots(1, 2, figsize = (14, 5), dpi = 80)
        model.plot_hist(X_train, title = "Histograma Errores MSE Train",
                        axis = axs[0])
        model.plot_hist(X_test, title = "Histograma Errores MSE Test",
                        axis = axs[1], density = True)
        plt.plot()

    # Guardar arquitectura del modelo
    if SAVE_MODEL_IMG:
        model.save_img_model("../doc/img/prueba.png")
        #print(model.model.summary())

    # Ejemplos de reconstrucción
    if SHOW_EXAMPLES:
        dss = [("TRAIN", X_train), ("TEST", X_test)]
        for ds_name, ds in dss:
            SHIFT = 5
            fig, axs = plt.subplots(2, 2, figsize = (14, 10), sharex = True)
            axs = axs.flat
            for ax, i in zip(axs, range(0, 4)):
                ts = reshape_data(ds[i + SHIFT].flatten())
                predict = reshape_data(model.predict_autoencoder(ts)).flat
                ax.plot(predict, label = "reconstrucción")
                ax.plot(ds[i + SHIFT].flat, label = "original")
                ax.legend()
            fig.suptitle("Reconstrucciones en " + ds_name)
            plt.show()

    # Ejemplos de anomalías
    if SHOW_ANOMALIES:
        stds = [-1.0, 0.001, 3, 4]
        for std in stds:
            fig, axs = plt.subplots(2, 2, figsize = (14, 10), dpi = 80)
            for i, ax in zip(range(4), axs.flat):
                """
                ts_an = gaussian_sine_pulse(X_train[i].flat, fc = 1, std = std,
                                       min_length = 4, max_length = 10)
                """
                ts_an = modify_trend(X_train[i].flat, period = 13, std = std,
                                       min_length = 7, max_length = 11)
                ts_rec = model.predict_autoencoder(reshape_data(ts_an)).flat
                ax.plot(ts_an, label = "perturbada")
                ax.plot(X_train[i], label = "original")
                ax.plot(ts_rec, label = "reconst")
                ax.legend()
            plt.suptitle("Tendencia sigma = " + str(std))
            plt.show()
    # Calcular PR con anomalías creadas
    if CALC_PR:
        # Proporciones de anomalías
        rt_an = [0.05, 0.1, 0.2, 0.3]
        # Parametros sigma
        stds = [-1.0, 0.001, 3, 4]
        sets = [(X_train, "TRAIN"), (X_test, "TEST")]
        # Para TRAIN Y TEST
        for ds, ds_name in sets:
            fig, axs = plt.subplots(2, 2, figsize = (12, 8), dpi = 80,
                                    sharex = True)
            # Para cada ratio
            for rt, ax in zip(rt_an, axs.flat):
                for idx, std in zip(range(len(stds)), stds):
                    plot = True if idx == len(stds) - 1 else False
                    X_anomalies = np.empty((0, ds.shape[1]))
                    # Creamos las anomalías
                    for i in range(0, int(rt * ds.shape[0])):
                        # Creación de anomalía
                        ts_an = modify_trend(ds[i].flat, period = 13,
                                               std = std,
                                               min_length = 7,
                                               max_length = 11)
                        X_anomalies = np.vstack((X_anomalies, ts_an))
                    X_anomalies = reshape_data(X_anomalies)
                    # Curva PR
                    recall_precision_curve(ds, X_anomalies, model,
                                           clf_name = "LSTM / std = " +
                                               str(std),
                                           title = "PRCurve Ratio Anomalias = "
                                               + str(rt),
                                           axis = ax, plot = plot)
            plt.suptitle(ds_name)
            plt.tight_layout()
            plt.show()
    # Calcular PR cuando anomalias son otras clases
    if CALC_PR_AN:
        sets = [(X_train, X_anomalos_train, "TRAIN"),
                (X_test, X_anomalos_test,"TEST")]
        fig, axs = plt.subplots(1, 2, figsize = (18, 6), sharey = True)
        # Curva PR para TEST y TRAIN
        for (ds, ds_an, ds_name), ax in zip(sets, axs.flat):
            recall_precision_curve(ds, ds_an, model,
                                   clf_name = "LSTM detector",
                                   title = "precision-recall curve " + ds_name,
                                   axis = ax, plot = True)
main()
