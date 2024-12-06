import pandas as pd
import numpy as np
import random

FOLDS = 5
EPOCAS = 5

def min_max(df: pd.DataFrame):
    df_norm = df.copy()
    for c in df_norm.columns:
        if c != 'classe':
            min = df[c].min()
            max = df[c].max()
            df_norm[c] = (df[c] - min) / (max - min)

    return df_norm

if __name__ == "__main__":
    
    # lendo como dataframe para o preparo dos dados ser facil
    df = pd.read_csv('../SistemasInteligentes_T2/data/treino_sinais_vitais_com_label.txt',sep=',',decimal='.')
    df.drop(columns=['id','si1','si2','grav'],inplace=True)
    
    # CROSSVALIDATION
    df_folds = np.array_split(df,FOLDS)
    for fold in df_folds:
        
        df_train = df.copy().drop(fold.index)
        df_test = fold.copy()
        
        # NORMALIZAÇÂO
        df_train = min_max(df_train)
        df_test = min_max(df_test)
        
        df_x = df_train.copy().drop(columns=['classe']) # treinamento
        df_y = df_train['classe'].copy() # resultado real
            
        # passando pra array para a manipulação ficar melhor
        df_x = np.array(df_x)
        df_y = np.array(df_y)
       
        
    