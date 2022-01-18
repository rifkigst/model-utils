import math
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import KFold
import sklearn.linear_model as linear_model
import sklearn.metrics as metrics
from sklearn.cluster import KMeans


def feature_scalling(X_train, X_test, method=StandardScaler(with_mean=True, with_std=False)):
    
    """
    Scales features using sklearn preprocessing.
    """
    
    X_train = method.fit_transform(X_train)
    X_test = method.transform(X_test)
    
    return X_train, X_test

def train_predict_ridge(alpha, X_train, y_train, X_test):
    
    """
    Trains ridge model and predicts test set.
    """
    
    ridge = linear_model.Ridge(alpha)
    ridge.fit(X_train, y_train)
    y_hat = ridge.predict(X_test)
    return y_hat

def tunning_params(X, y, n_split, alphas, to_print=False):
    
    """
    Finds the best alpha in an inner fully randomized CV loop.
    """

    kf = KFold(n_splits=n_split, shuffle=True)
    best_alpha = 0
    best_r2 = 0
    
    for idx, alpha in enumerate(alphas):
        y_hat = np.zeros_like(y)
        for train_idx, test_idx in kf.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            X_train, X_test = feature_scalling(X_train, X_test)
            y_hat[test_idx] = train_predict_ridge(alpha, X_train, y_train, X_test)
        
        r2 = metrics.r2_score(y, y_hat)
        
        if r2 > best_r2:
            best_alpha = alpha
            best_r2 = r2
            
    if to_print:
        print(best_alpha)
    return best_alpha

def running_fold(X, y, train_idx, test_idx, k_inner, alphas, to_print=False):
    
    """
    Evaluates one fold of outer CV.
    """
    
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    best_alpha = tunning_params(X_train, y_train, k_inner, alphas, to_print)
    X_train, X_test = feature_scalling(X_train, X_test)
    y_test_hat = train_predict_ridge(best_alpha, X_train, y_train, X_test)
    r2 = metrics.r2_score(y_test, y_test_hat)
    mse = metrics.metrics.mean_squared_error(y_test, y_test_hat)
    return r2, math.sqrt(mse), y_test_hat


def fit(X, y, k=5, k_inner=5, random_seed=7, points=10, alpha_low=1, alpha_high=5, to_print=False):
    
    """
    Run randomized CV on given X and y
    Returns r2, yhat
    """
    
    np.random.seed(random_seed)
    alphas = np.logspace(alpha_low, alpha_high, points)
    r2s = []
    rmselist = []
    y_hat = np.zeros_like(y)
    kf = KFold(n_splits=k, shuffle=True)
    fold = 0
    for train_idx, test_idx in kf.split(X):
        if to_print:
            print(f"fold: {fold}", end='\r')

        r2,rmse,y_p = running_fold(X, y, train_idx, test_idx, k_inner, alphas, to_print)
        r2s.append(r2)
        rmselist.append(rmse)

        y_hat[test_idx] = y_p
        fold += 1
    return np.mean(r2s), np.mean(rmselist), y_hat

def run_test():
    return print("Just run and success")