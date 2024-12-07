import pandas as pd
import numpy as np
import random
import math

FOLDS = 5
NUM_CAMADA = 2
TAXA_APREND = 0.15
EPOCAS = 5
BIAS = 0
NEURONIOS = 3

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
def funcao_entrada(entrada, pesos):
    
    p_entrada = 0
    for i in range(len(pesos)):
        p_entrada += entrada[i]*pesos[i]
        
    return p_entrada + BIAS

# funcao não linear de saida (Sigmoid)
def funcao_ativacao(p_entrada):
    # saida = 1 / (1 + e^(-ativacao))
    return 1.0 / (1.0 + math.exp(-p_entrada))

def feed_foward(rede,entrada):
    # entrada = rede.get('entrada')
    camadas = rede.get('camadas')
    saidas = []
    for camada in camadas:
        saida_c = []
        for neuronio in camada:
            p_entrada = funcao_entrada(entrada,neuronio)
            saida_n = funcao_ativacao(p_entrada)
            saida_c.append(saida_n)
        entrada = saida_c
        saidas.append(saida_c)
    
    return saidas

# derivada da funcao de ativavao (sigmoid)
def derivada_ativacao(p_saida):
    return p_saida * (1.0 - p_saida)

def ajuste_pesos(rede,deltas,saidas,entrada):
    camadas = rede.get('camadas')
    # entrada = rede.get('entrada')
    
    # novo peso(x) = peso(x) + taxa*delta_camada(x)*entrada(x) 
    for i in range(len(camadas)):
        if i != 0:
            entrada = saidas[i-1]
    
        # para cada peso de cada neuronio da camada
        for j, neuronio in enumerate(camadas[i]):
            for k in range(len(entrada)):
                neuronio[k] += TAXA_APREND * deltas[i][j] * entrada[k]
            neuronio[-1] += TAXA_APREND * deltas[i][j] * BIAS

def back_propagation(rede,saidas,s_esperado,entrada):
    camadas = rede.get('camadas')
    lista_invertida = reversed(range(len(camadas)))
    
    # delta(x) = soma(delta(x+1)*saida(x+1)) * derivada(x)
    deltas = []
    erro_saida = (s_esperado - saidas[-1]) * derivada_ativacao(saidas[-1][0])
    deltas.append([erro_saida])
    
    for i in lista_invertida:
        # sem camada de saida
        if i != len(camadas)-1:
            camada = camadas[i]
            # neuronios da camada 
            delta = 0
            delta_c = []
            for j in range(len(camada)):
                for k in range(len(deltas[0])):
                    # delta anterior * peso de todos os neuronios da camada anterior
                    delta += deltas[0][k] * camadas[i+1][k][j]
                delta_c.append(derivada_ativacao(saidas[i][j]) * delta)
            deltas.insert(0,delta_c)
    
    ajuste_pesos(rede,deltas,saidas,entrada) 
    
def treinamento_rede(rede,df_x,df_y):
    epoca = 0
    erro_geral = []
    while epoca < EPOCAS:
            
        for i,entrada in enumerate(df_x):
            resultado_real = df_y[i]
            erros = []
            
            saidas = feed_foward(rede,entrada)
            erro_treinamento = (resultado_real-saidas[-1])**2
            back_propagation(rede,saidas,resultado_real,entrada)
                
            erros.append(erro_treinamento)
            # print(f'Erro linha {i}, epoca {epoca}: {erro_treinamento}')
                
        media_epoca = sum(erros)/len(erros)
        print(f'Media erro EPOCA {epoca}: {media_epoca}')
        erro_geral.append(media_epoca)
        epoca += 1
        
    return(erro_geral)

def acuracia(dados_reais,dados_previstos):
    certos = 0
    for i in range(len(dados_reais)):
        if dados_previstos[i] == dados_reais[i]:
            certos += 1
    return certos/len(dados_previstos) * 100    

def previsao(rede,df_test):
    df_x = df_test.copy().drop(columns=['grav']) # treinamento
    df_y = df_test['grav'].copy() # resultado real
            
    # passando pra array para a manipulação ficar melhor
    df_x = np.array(df_x)
    df_y = np.array(df_y)
    erros = []
    
    previsoes = []
    for i, entrada in enumerate(df_x):
        saidas = feed_foward(rede,entrada)
        erro = (df_y[i]-saidas[-1])**2
        previsoes.append(saidas[-1])
        erros.append(erro)
    
    # para classificacao
    # ac = acuracia(df_y,previsoes)
    
    # para regressao
    media_erro = sum(erros)/len(erros)
    
    return media_erro
    
def cria_rede(num_entradas):
    # rede com entrada, camada oculta, saida
    
    camadas = []
    # duas camadas ocultas
    for _ in range(NUM_CAMADA):
        neuronios = []
        for _ in range(NEURONIOS):
            pesos = [random.random() for _ in range(num_entradas)]
            neuronios.append(pesos)
        camadas.append(neuronios)
    # camada de saida com 1 neuronio
    n_saida = [random.random() for _ in range(num_entradas)]
    camadas.append([n_saida])
        
    return {'camadas':camadas}

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
        
        rede = cria_rede(df_x.shape[1])
        
        # INICIO
        erro_treinamento = treinamento_rede(rede,df_x,df_y)
        print(f'Erro geral fold: {sum(erro_treinamento)/len(erro_treinamento)}')
        
        ac = previsao(rede,df_test)
        print(f'Acuracia fold: {ac}')
            
        
    