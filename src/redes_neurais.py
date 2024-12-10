import pandas as pd
import numpy as np
import random
import math
import json
import os
from utils import load_and_preprocess_data, min_max, split_cross_validation, prepare_train_test_data, separate_features_and_labels

FOLDS = 5
NUM_CAMADA = 1
TAXA_APREND = 0.05
EPOCAS = 100
BIAS = 0
NEURONIOS = 3
N_SAIDA = 1
TIPO = 'regressao'
DATA_FILE = "data/treino_sinais_vitais_com_label.txt"
PATH_JSON = 'results/rn/result_rn_27.json'
AT = 'tanh'

# soma dos pesos * valores
def funcao_entrada(entrada, pesos):
    
    p_entrada = 0
    for i in range(len(pesos)):
        p_entrada += entrada[i]*pesos[i]
        
    return p_entrada + BIAS

# funcao não linear de saida (Sigmoid)
def funcao_ativacao(p_entrada):
    if AT == 'sig':
        val =  1.0 / (1.0 + math.exp(-p_entrada))
    elif AT == 'relu':
        val = np.maximum(0,p_entrada)
    elif AT == 'tanh':
        val = np.tanh(p_entrada)
        
    return val

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
    if AT == 'sig':
        val =  p_saida * (1 - p_saida)
    elif AT == 'relu':
        val = np.where(p_saida > 0,1,0)
    elif AT == 'tanh':
        val = 1 - p_saida**2
    
    return val

def ajuste_pesos(rede,deltas,saidas,entrada):
    camadas = rede.get('camadas')
    # entrada = rede.get('entrada')
    
    # novo peso(x) = peso(x) + taxa*delta_camada(x)*entrada(x) 
    for i in range(len(camadas)):
        if i != 0:
            entrada = saidas[i-1]
    
        # para cada peso de cada neuronio da camada
        for j, neuronio in enumerate(camadas[i]):
            for k in range(len(neuronio)):
                neuronio[k] += TAXA_APREND * deltas[i][j] * entrada[k]
            neuronio[-1] += TAXA_APREND * deltas[i][j]

def back_propagation(rede,saidas,s_esperado,entrada):
    camadas = rede.get('camadas')
    lista_invertida = reversed(range(len(camadas)))
    
    # delta(x) = soma(delta(x+1)*saida(x+1)) * derivada(x)
    deltas = []
    erro_saida = [derivada_ativacao(saidas[-1][i]) * (s_esperado - saidas[-1][i]) 
                  for i in range(len(saidas[-1]))]
    deltas.append(erro_saida)
    
    for i in lista_invertida:
        # sem camada de saida
        if i != len(camadas)-1:
            camada = camadas[i]
            # neuronios da camada 
            delta = 0
            delta_c = []
            for j in range(len(camada)):
                for k in range(len(deltas[0])):
                    for p in range(len(camada[j])):
                    # delta anterior * peso de todos os neuronios da camada anterior
                        delta += deltas[0][k] * camadas[i+1][k][p]
                delta_c.append(derivada_ativacao(saidas[i][j]) * delta)
            deltas.insert(0,delta_c)
    
    ajuste_pesos(rede,deltas,saidas,entrada) 

def calcula_erro(saidas, esperado):
    erro = 0
    for s in saidas:
        erro += (esperado - s)**2
    return erro
    
def treinamento_rede(rede,df_x,df_y):
    epoca = 0
    erro_geral = []
    while epoca < EPOCAS:
        erros = []  
        for i,entrada in enumerate(df_x):
            resultado_real = df_y[i]
            saidas = feed_foward(rede,entrada)
            erro_treinamento = calcula_erro(saidas[-1],resultado_real)
            back_propagation(rede,saidas,resultado_real,entrada)
                
            erros.append(erro_treinamento)
            # print(f'Erro linha {i}, epoca {epoca}: {erro_treinamento}')
                
        media_epoca = sum(erros)/len(erros)
        # print(f'Media erro EPOCA {epoca}: {media_epoca}')
        erro_geral.append({'epoca': epoca,
                           'erros_epoca': media_epoca})
        epoca += 1
        
    return erro_geral

def acuracia(dados_reais,dados_previstos):
    certos = 0
    for i in range(len(dados_reais)):
        classe_prevista = np.argmax(dados_previstos[i])
        if classe_prevista + 1 == dados_reais[i]:
            certos += 1
    return certos/len(dados_previstos) * 100

def previsao(rede,df_test,tipo):
    
    x_test, y_test = separate_features_and_labels(df_test,tipo)
    erros = []
    previsoes = []
    
    for i, entrada in enumerate(x_test):
        saidas = feed_foward(rede,entrada)
        erro = calcula_erro(saidas[-1],y_test[i])**2
        previsoes.append(saidas[-1])
        erros.append(erro)
    
    if tipo == 'classificacao':
        metrica = acuracia(y_test,previsoes)
    else:
        metrica = sum(erros)/len(erros)
    
    return metrica, erros
    
def cria_rede(num_entradas):
    # rede com entrada, camada oculta, saida
    
    camadas = []
    # duas camadas ocultas
    for _ in range(NUM_CAMADA):
        neuronios = []
        for _ in range(NEURONIOS):
            pesos = [random.uniform(-1, 1) for _ in range(num_entradas)]
            neuronios.append(pesos)
        camadas.append(neuronios)
    # camada de saida 
    saida = []
    for _ in range(N_SAIDA):
        n_saida = [random.uniform(-1, 1) for _ in range(num_entradas)]
        saida.append(n_saida)
    camadas.append(saida)
        
    return {'camadas':camadas}

# converte arrays para listas para garantir que sejam serializáveis
def convert_to_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, list):
        return [convert_to_serializable(i) for i in obj]
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    else:
        return obj
    
def salvar_metricas(metricas): 
    
    os.makedirs(os.path.dirname(PATH_JSON), exist_ok=True)
    
    param = {'neuronios_camada' : NEURONIOS,
             'neuronios_saida' : N_SAIDA,
             'epocas': EPOCAS,
             'taxa_aprendizado': TAXA_APREND,
             'camadas' : NUM_CAMADA,
             'folds' : FOLDS,
             'bias' : BIAS,
             'tipo': TIPO,
             'ativacao' : AT,
             'metricas' : metricas
        }
    
    m_serializavel = convert_to_serializable(param)
    with open(PATH_JSON, 'w') as f:
        json.dump(m_serializavel, f, indent=4)
    

if __name__ == "__main__":

    df = load_and_preprocess_data(DATA_FILE,TIPO)
    
    metricas = []
    # CROSSVALIDATION
    df_folds = np.array_split(df,FOLDS)
    for i,fold in enumerate(df_folds):
        
        df_train, df_test = prepare_train_test_data(df, fold)
        df_train = min_max(df_train)
        df_test = min_max(df_test)
        x_train, y_train = separate_features_and_labels(df_train,TIPO)
        
        rede = cria_rede(x_train.shape[1])
        
        # INICIO
        erro_treinamento = treinamento_rede(rede,x_train,y_train)
        # print(f'Erro geral fold: {sum(erro_treinamento)/len(erro_treinamento)}')
        
        ac, erros = previsao(rede,df_test,TIPO)
        print(f'Acuracia fold: {ac}')
        
        m_fold = {'fold': i,
                  'erro_treinamento': erro_treinamento,
                  'metrica_previsao': ac}
        metricas.append(m_fold)
    
    salvar_metricas(metricas)