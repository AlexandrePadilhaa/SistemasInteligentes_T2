import pandas as pd
import numpy as np
import random
import math

FOLDS = 5
NUM_CAMADA = 3
TAXA_APREND = 0.15
EPOCAS = 5
BIAS = 0

# funcao para normalização
def min_max(df: pd.DataFrame):
    df_norm = df.copy()
    for c in df_norm.columns:
        # if c != 'classe':
        min = df[c].min()
        max = df[c].max()
        df_norm[c] = (df[c] - min) / (max - min)

    return df_norm

# soma dos pesos * valores
def funcao_entrada(neuronio):
    
    pesos = neuronio.get('pesos')
    valores = neuronio.get('valores')
    
    p_entrada = 0
    for i in range(len(pesos)):
        p_entrada += valores[i]*pesos[i] + BIAS
        
    return p_entrada

# funcao não linear de saida (Sigmoid)
def funcao_ativacao(p_entrada):
    # saida = 1 / (1 + e^(-ativacao))
    return 1.0 / (1.0 + math.exp(-p_entrada))

def feed_foward(rede,r_real):
    
    # for camada in rede:
    
    for neuronio in rede:
            
        p_entrada = funcao_entrada(neuronio)
        saida = funcao_ativacao(p_entrada)
        erro = r_real - saida
            
    return erro

def cria_rede(df_train: pd.DataFrame):
    # rede com entrada, camada oculta, saida
    rede = []
    
    camada_oculta = []
    for itens in df_train:
        pesos = [random.random() for _ in range(len(df_train))]
        neuronio = {'valores': df_train,
                    'pesos': pesos}
        camada_oculta.append(neuronio)
    rede.append(camada_oculta)

    
    return camada_oculta

if __name__ == "__main__":
    
    # lendo como dataframe para o preparo dos dados ser facil
    df = pd.read_csv('../SistemasInteligentes_T2/data/treino_sinais_vitais_com_label.txt',sep=',',decimal='.')
    df.drop(columns=['id','si1','si2','classe'],inplace=True)
    
    # CROSSVALIDATION
    df_folds = np.array_split(df,FOLDS)
    for fold in df_folds:
        
        df_train = df.copy().drop(fold.index)
        df_test = fold.copy()
        
        # NORMALIZAÇÂO
        df_train = min_max(df_train)
        df_test = min_max(df_test)
        
        df_x = df_train.copy().drop(columns=['grav']) # treinamento
        df_y = df_train['grav'].copy() # resultado real
            
        # passando pra array para a manipulação ficar melhor
        df_x = np.array(df_x)
        df_y = np.array(df_y)
        
        # INICIO
        erro_geral = []
        for i,linha in enumerate(df_x):
           
            resultado_real = df_y[i]
            epoca = 0
            erros = []
            
            while epoca < EPOCAS:
                rede = cria_rede(linha)
                erro = feed_foward(rede,resultado_real)
                erros.append(erro)
                print(f'Erro epoca {epoca}: {erro}')
                epoca += 1
                
            media_linha = sum(erros)/len(erros)
            print(f'Media erro linha {i}: {media_linha}')
            erro_geral.append(media_linha)
            
        print(f'Erro geral: {sum(erro_geral)/len(erro_geral)}')
            
        
    