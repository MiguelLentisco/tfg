import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from scipy.signal import gaussian, gausspulse	
from statsmodels.tsa.seasonal import seasonal_decompose
#from pmdarima.arima import auto_arima
from keras.models import Sequential
from keras.layers import LSTM, RepeatVector, TimeDistributed, Dense
from keras.regularizers import l2
from keras.models import load_model
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.utils.vis_utils import plot_model
from scipy.stats import gaussian_kde
from sklearn.metrics import auc

def load_dataset(ds_name):
    """
        Carga el dataset con cada serie temporal estandarizada y con las etiquetas.
        Añade un shuffle al cargar los datos.

        Parameters
        ----------
        ds_name: str
            Ruta del dataset

        Returns
        ----------
        X : np.array
            Dataset con las series temporales
        y : np.array
            Etiquetas de las series temporales
    """

    # Leemos los datos
    df = pd.read_csv(ds_name, sep=',', header = 0, index_col = 0)
    # Reordenamos los datos
    #df = df.sample(frac = 1).reset_index(drop = True)
    # Dataset
    X = df.iloc[:, 1:]
    # Series temporales estandarizadas
    #X = X.sub(X.mean(1), axis = 0).div(X.std(1), axis = 0).fillna(0.0).to_numpy("float64")
    X = X.to_numpy("float64")
    # Etiquetas codificadas
    le = preprocessing.LabelEncoder()
    y = le.fit_transform(df.iloc[:, 0].to_numpy())
    return X, y
    
def merge_datasets(X1, X2, y1, y2):
    X = np.vstack((X1, X2))
    y = np.concatenate((y1, y2))
    return X, y
    
def print_dataset(X, y = None, ds_name = ""):
    if y is not None:
        colors = ["red", "blue"]
        for c in np.unique(y):
            for ts in X[y == c]:
                plt.plot(ts, color = colors[c])
            plt.plot([], [], color = colors[c], label = c)
        plt.legend(title = "Clase")
    else:
        for ts in X:
            plt.plot(ts, color = "blue")
        
    plt.title("Dataset " + ds_name)
    plt.xlabel("Time (t)")
    plt.ylabel("TS value")
    plt.show()
    
def reshape_data(X):
    if len(X.shape) == 2:
        return X.reshape((X.shape[0], X.shape[1], 1))
    elif len(X.shape) == 3:
        return X.reshape((X.shape[0], X.shape[1]))
    elif len(X.shape) == 1:
        return X.reshape((1, X.size, 1))
    

SEED = 42
np.random.seed(SEED)
from sklearn.preprocessing import KBinsDiscretizer

# model_4.h5 el mejor (20, 0.0001, 300 epocas)
def main():
    SEED = 42
    DATASET = "./datasets/TwoLeadECG"
    
    # Juntamos particiones
    X1, y1 = load_dataset(DATASET + "_TRAIN.csv")
    X2, y2 = load_dataset(DATASET + "_TEST.csv")
    X, y = merge_datasets(X1, X2, y1, y2)
    # Nos quedamos con un tipo
    
    # División 80/20
    X_train, X_test = train_test_split(X, test_size = 0.2, random_state = SEED)
    
    est = KBinsDiscretizer(n_bins = 4, encode = "ordinal", strategy = "quantile").fit(X_train.T)
    X_train_new = est.transform(X_train.T).T
    X_train_new = X_train_new - 2
    plt.plot(X_train[2])
    plt.plot(X_train_new[2])
    plt.show()
        
        
    
main()
