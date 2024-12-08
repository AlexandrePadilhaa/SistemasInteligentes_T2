import pandas as pd
import numpy as np
import random

# NORMALIZAÇÂO
def min_max(df: pd.DataFrame):
    df_norm = df.copy()
    for c in df_norm.columns:
        if c != 'classe':
            min = df[c].min()
            max = df[c].max()
            df_norm[c] = (df[c] - min) / (max - min)
    return df_norm

def load_and_preprocess_data(filepath: str,tipo: str):
    if tipo == 'classificacao': 
        var = 'grav'
    else :
        var = 'classe'
        
    # lendo como dataframe para o preparo dos dados ser facil
    df = pd.read_csv(filepath, sep=',', decimal='.')
    df.drop(columns=['id', 'si1', 'si2', var], inplace=True)
    return df

def split_cross_validation(df: pd.DataFrame, folds: int):
    """
    Divide o DataFrame em subconjuntos (folds) para validação cruzada.
    """
    return np.array_split(df, folds)

def prepare_train_test_data(df: pd.DataFrame, fold: pd.DataFrame):

    df_train = df.copy().drop(fold.index)
    df_test = fold.copy()
    return df_train, df_test

def separate_features_and_labels(df: pd.DataFrame, tipo: str):
    if tipo == 'classificacao': 
        var = 'classe'
    else :
        var = 'grav'
    df_x = df.copy().drop(columns=[var])  # atributos de entrada
    df_y = df[var].copy()                # rótulos
    return np.array(df_x), np.array(df_y)


