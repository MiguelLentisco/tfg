import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import KBinsDiscretizer, LabelEncoder, \
                                  StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from predictors import PredictorLSTM, PredictorDense
from discretization import SAX, StringEncoder
from sklearn.metrics import mean_squared_error

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
    X_train, X_test = train_test_split(X[0], test_size = 0.25, shuffle = False)
    X_train.shape = (1, -1)
    X_test.shape = (1, -1)
    
    plt.plot(X_train[0])
    plt.show()
    print(X_train[0].size, X_test[0].size)
    pipe = Pipeline([("scaler", StandardScaler()), 
                     ("disc", KBinsDiscretizer(n_bins = 4, encode = "ordinal", 
                                               strategy = "kmeans")),
                     ("scaler2", StandardScaler())])
    pipe = Pipeline([("disc", KBinsDiscretizer(n_bins = 4, encode = "ordinal", 
                                               strategy = "kmeans"))])
    print(X_train[0].size)
    pipe = Pipeline([("dis", SAX(tam_window = 3, alphabet_tam = 4)),
                     ("encoder", StringEncoder())])
                      
    X_train_d = pipe.fit_transform(X_train)
    X_test_d = pipe.transform(X_test)
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
    
    lookback = 3
    inputs, target = create_window(X_train_d[0], lookback)
    LSTM_model = PredictorLSTM(n_neurs_lstm = 18, epochs = 350, verbose = 1,
                               n_neurs_dense = 100, n_neurs_conv = 18)
    Dense_model = PredictorDense(n_neurs_dense = 100, epochs = 350, 
                                 verbose = 1)
    
    Dense_model.fit(inputs, target)
    Dense_model.print_history()
    LSTM_model.fit(inputs, target)
    LSTM_model.print_history()
    
    
    fig, axs = plt.subplots(2, 2, figsize=(18, 12))
    axs = axs.flat
    
    predicts_lstm = LSTM_model.predict(inputs).flat
    predicts_dense = Dense_model.predict(inputs).flat
    mse_train_lstm = mean_squared_error(target, predicts_lstm)
    mse_train_dense = mean_squared_error(target, predicts_dense)
    axs[0].plot(predicts_lstm, "--", label = "LSTM")
    axs[0].plot(predicts_dense, "--", label = "Dense")
    axs[0].plot(target, "-", label = "Serie")
    axs[0].legend()
    axs[0].set_title("Predicciones TRAIN")
    
    axs[2].plot(np.abs(predicts_lstm - target), "--", label = "LSTM")
    axs[2].plot(np.abs(predicts_dense - target), "--", label = "Dense")
    axs[2].set_title("Diferencias original con predicción TRAIN")
    axs[2].legend()
    
    
    inputs, target = create_window(X_test_d[0], lookback)
    predicts_lstm = LSTM_model.predict(inputs).flat
    predicts_dense = Dense_model.predict(inputs).flat
    mse_test_lstm = mean_squared_error(target, predicts_lstm)
    mse_test_dense = mean_squared_error(target, predicts_dense)
    axs[1].plot(predicts_lstm, "--", label = "LSTM")
    axs[1].plot(predicts_dense, "--", label = "Dense")
    axs[1].plot(target, "-", label = "Serie")
    axs[1].legend()
    axs[1].set_title("Resultados TEST")
    
    axs[3].plot(np.abs(predicts_lstm - target), "--", label = "LSTM")
    axs[3].plot(np.abs(predicts_dense - target), "--", label = "Dense")
    axs[3].set_title("Diferencias original con predicción TEST")
    axs[3].legend()
    
    plt.show()
    
    print("MSE TRAIN: LSTM {:.3f} / Dense {:.3f}".format(mse_train_lstm, 
                                                         mse_train_dense))
    print("MSE TEST: LSTM {:.3f} / Dense {:.3f}".format(mse_test_lstm, 
                                                        mse_test_dense))
    
    
main()
