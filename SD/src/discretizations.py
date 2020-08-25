#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import string
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import LabelEncoder

class SAX:
    def __init__(self, tam_window, alphabet_tam):
        self.tam_window = tam_window
        self.breakpoints = self.calc_breakpoints(alphabet_tam)
        self.alphabet = string.ascii_lowercase[:alphabet_tam]
        self.std = None
        self.mean = None
        
    def standarization(self, X):        
        res = [(X[i] - self.mean[i]) / self.std[i] for i in range(X.shape[0])]
        return np.array(res)
        
    def PAA(self, x):
        n_window = x.size // self.tam_window
        res = [np.mean(x[(i*self.tam_window):((i+1)*self.tam_window)]) 
                for i in range(n_window)]
        return np.array(res)    
        
    def calc_breakpoints(self, alphabet_tam):
        bp = [float("-inf")]
        bp += [norm.ppf((i+1) / alphabet_tam) for i in range(alphabet_tam - 1)]
        bp += [float("inf")]
        return bp
    
    def string_transformation(self, x_paa):
        word = []
        for point in x_paa:
            for i in range(1, len(self.breakpoints)):
                if point >= self.breakpoints[i-1] and \
                    point < self.breakpoints[i]:
                    word.append(self.alphabet[i-1])
        return word
    
    def discretization(self, x):
        x_paa = self.PAA(x)
        return self.string_transformation(x_paa)
    
    def fit(self, X, y = None):
        self.mean = np.mean(X, axis = 1)
        self.std = np.std(X, axis = 1)
        return self
        
    def transform(self, X, y = None):
        X_norm = self.standarization(X)
        return [self.discretization(x) for x in X_norm]  
    
    def fit_transform(self, X, y = None):
        self.fit(X, y)
        return self.transform(X, y)
    
    def inverse_transform(self, X):
        return X
    
class DiscretizationEncoder:
    def __init__(self):
        self.encoders = []
        
    def fit(self, X, y = None):
        self.encoders = [LabelEncoder().fit(x) for x in X]
        return self
        
    def transform(self, X, y = None):
        return [enc.transform(x) for x, enc in zip(X, self.encoders)]
    
    def fit_transform(self, X, y = None):
        self.fit(X, y)
        return self.transform(X, y)
    
    def inverse_transform(self, X):
        return [enc.inverse_transform(x) for x, enc in zip(X, self.encoders)]