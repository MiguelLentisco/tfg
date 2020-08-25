import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import KBinsDiscretizer, LabelEncoder, \
                                  StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from predictors import PredictorLSTM

SEED = 42
np.random.seed(SEED)

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
    #X = X.sub(X.mean(1), axis = 0).div(X.std(1), axis = 0).fillna(0.0)
    X = X.to_numpy("float64")
    # Etiquetas codificadas
    le = LabelEncoder()
    y = le.fit_transform(df.iloc[:, 0].to_numpy())
    return X, y
    
    
def create_window(x, size_w):
    inputs = np.empty((0))
    target = []
    for i in range(x.size - size_w):
        inputs = np.hstack((inputs, x[i:(i + size_w)]))
        target += [x[i + size_w]]
    inputs = inputs.reshape((-1, size_w, 1))
    return inputs, np.array(target)


def standarize(x):
    return (x - np.mean(x)) / np.std(x)

def main():
    
    DATASET = "../../Datasets/TS/UWaveGestureLibraryAll"
    # Juntamos particiones
    X, _ = load_dataset(DATASET + "_TRAIN.csv")
    #X_test, _ = load_dataset(DATASET + "_TEST.csv")
    X_train, X_test = train_test_split(X[0], test_size = 0.3, shuffle = False)
    X_train.shape = (-1, X_train.size)
    X_test.shape = (-1, X_test.size)
    
   
    
    fig, ax = plt.subplots(1, figsize=(9, 6))
    ax.plot(X[0], label = "original")
    w = 157
    paa_x = []
    paa_y = []
    for i in range(X[0].size // w):
        ax.plot(i*w + w // 2, np.mean(X[0, (i*w):((i+1)*w)]), "ro")
        ax.hlines(np.mean(X[0, (i*w):((i+1)*w)]), i*w, (i+1)*w)
        ax.vlines([i*w, (i+1)*w], np.min(X[0]), np.max(X[0]), linestyles = "dotted")
        paa_y.append(np.mean(X[0, (i*w):((i+1)*w)]))
        paa_x.append(i*w + w // 2)
    ax.set_title("PAA w = 157")
    ax.plot(paa_x, paa_y, "--", label = "paa")
    ax.legend()
    plt.plot()
    
    """
    plt.plot(X_train, color = "red")
    plt.plot(np.concatenate((X_train, X_test)))
    plt.show()
    """
    
    """
    plt.plot(X_train[0])
    plt.show()
    pipe = Pipeline([("scaler", StandardScaler()), 
                     ("disc", KBinsDiscretizer(n_bins = 4, encode = "ordinal", 
                                               strategy = "kmeans")),
                     ("scaler2", StandardScaler())])
    X_train_d = pipe.fit_transform(X_train.T).T
    X_test_d = pipe.transform(X_test.T).T
    plt.plot(standarize(X_train[0]), label = "Continua")
    plt.plot(X_train_d[0], label = "Discreta")
    plt.title("TRAIN continua y discreta")
    plt.legend()
    plt.show()
    
    plt.plot(standarize(np.concatenate((X_train[0], X_test[0]))), 
             label = "Continua")
    plt.plot(np.concatenate((X_train_d[0], X_test_d[0])), label = "Discreta")
    plt.title("TEST continua y discreta")
    plt.legend()
    plt.show()
    
    
    inputs, target = create_window(X_train_d[0], 5)
    LSTM_model = PredictorLSTM(n_neurs = 10, epochs = 200, verbose = 1)
    LSTM_model.fit(inputs, target)
    LSTM_model.print_history()
    
    fig, axs = plt.subplots(1, 2, figsize=(18, 6))
    axs = axs.flat
    
    predicts = LSTM_model.predict(inputs)
    axs[0].plot(predicts, "-o", label = "Modelo")
    axs[0].plot(target, "-o", label = "Original")
    axs[0].legend()
    axs[0].set_title("Resultados TRAIN")
    
    inputs, target = create_window(X_test_d[0], 5)
    predicts = LSTM_model.predict(inputs)
    axs[1].plot(predicts, "-o", label = "Modelo")
    axs[1].plot(target, "-o", label = "Original")
    axs[1].legend()
    axs[1].set_title("Resultados TEST")
    plt.show()
    """
main()
