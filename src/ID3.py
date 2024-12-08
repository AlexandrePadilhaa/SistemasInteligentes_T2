import json
import os
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.metrics import (confusion_matrix, classification_report, 
                             accuracy_score, ConfusionMatrixDisplay)
from sklearn.utils.multiclass import unique_labels
import matplotlib.pyplot as plt
from utils import load_and_preprocess_data, min_max, split_cross_validation, prepare_train_test_data, separate_features_and_labels


DATA_FILE = "data/treino_sinais_vitais_com_label.txt"
TIPO = 'classificacao'
CRITERION = "entropy"      # Critério de divisão baseado em entropia (ID3)
MAX_DEPTH = None           # Profundidade máxima (None = sem limite)
MIN_SAMPLES_SPLIT = 2      # Número mínimo de amostras para dividir um nó
MAX_FEATURES = "sqrt"      # Define o número máximo de recursos (features) a serem considerados para cada divisão. "sqrt" , "log2",  "1/3 1/4 ..."
MAX_LEAF_NODES = None         # Limitar o número de folhas
SPLITTER = "best"          # Controla como as divisões dos nós são feitas. "best" ou "random"
CCP_ALPHA = 0.1           # Controla o pruning (poda) da árvore baseado na complexidade de custo. Corta subarvores menos relevantes
RANDOM_STATE = 42         
FOLDS = 10

SUMMARY_FILE = 'results/id3/summary1.json'

def train_id3_model(x_train, y_train):
    #Arvore de decisão com critério de entropia = ID3
    model = DecisionTreeClassifier(
        criterion=CRITERION,
        max_depth=MAX_DEPTH,
        min_samples_split=MIN_SAMPLES_SPLIT,
        max_features=MAX_FEATURES,
        max_leaf_nodes=MAX_LEAF_NODES,
        splitter=SPLITTER,
        ccp_alpha=CCP_ALPHA,
        random_state=RANDOM_STATE
    )
    model.fit(x_train, y_train)
    return model

import os
import json
import numpy as np

def save_summary(metrics_list, filepath=SUMMARY_FILE):

    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    summary = {
        "metrics_per_fold": metrics_list
    }

    with open(filepath, 'w') as f:
        json.dump(summary, f, indent=4)

    print(f"Summary file saved at: {filepath}")



def evaluate_id3_model(fold, model, x_test, y_test, class_names):
    y_pred = model.predict(x_test)

    print("\nConfusion Matrix:")
    conf_matrix = confusion_matrix(y_test, y_pred, labels=[1, 2, 3, 4])
    print(conf_matrix)

    #Plotar matriz de confusão
    # disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=class_names)
    # disp.plot(cmap=plt.cm.Blues)
    # plt.title("Matriz de Confusão")
    # plt.show()

    print("\nClassification Report:")
    report = classification_report(y_test, y_pred, labels=[1, 2, 3, 4], target_names=class_names, zero_division=0, output_dict=True)
    print(report)


    return {
        "fold": fold,
        #"parameters": parameters,
        "confusion_matrix": conf_matrix.tolist(),
        "classification_report": report,
        "tree_depth": model.tree_.max_depth
    }


def visualize_decision_tree(model, feature_names, class_names):
    # texto
    tree_text = export_text(model, feature_names=feature_names)
    print("\nDecision Tree Structure:\n")
    print(tree_text)

    #  gráfico
    plt.figure(figsize=(20, 10))
    plot_tree(model, feature_names=feature_names, class_names=class_names, filled=True, fontsize=10)
    plt.show()


def cross_validate_id3(df, folds=5):
    fold_data = np.array_split(df, folds)
    metrics_list = []
    parameters = {
        "criterion": CRITERION,
        "max_depth": MAX_DEPTH,
        "min_samples_split": MIN_SAMPLES_SPLIT,
        "max_features": MAX_FEATURES,
        "max_leaf_nodes": MAX_LEAF_NODES,
        "splitter": SPLITTER,
        "ccp_alpha": CCP_ALPHA,
        "random_state": RANDOM_STATE,
    }
    metrics_list.append(parameters)


    for fold_idx, test_fold in enumerate(fold_data):
        print(f"\n--- Fold {fold_idx + 1} ---")
   
        # Preparar dados de treino e teste
        train_fold = df.drop(test_fold.index)
        x_train, y_train = separate_features_and_labels(train_fold,TIPO)
        x_test, y_test = separate_features_and_labels(test_fold,TIPO)

        # Treinar modelo ID3
        model = train_id3_model(x_train, y_train)

        # Avaliar o modelo
        metrics = evaluate_id3_model(fold_idx, model, x_test, y_test, class_names=['1', '2', '3', '4'])
        metrics_list.append(metrics)


        # Visualizar a estrutura da árvore no último fold (opcional)
        #if fold_idx == folds - 1:
            #visualize_decision_tree(model, feature_names=train_fold.columns[:-1], class_names=['1', '2', '3', '4'])
    
    save_summary(metrics_list)

    return metrics_list


def main():
    filepath = DATA_FILE

    df = load_and_preprocess_data(filepath, TIPO)

    df = min_max(df)
    
    print(f"Executando validação cruzada com {FOLDS} folds...")
    metrics = cross_validate_id3(df, folds=FOLDS)


    print("\n--- Resultados Gerais ---")
    avg_accuracy = np.mean([
        m["classification_report"]["accuracy"] 
        for m in metrics 
        if "classification_report" in m
    ])
    #avg_accuracy = np.mean([m.get("accuracy", 0) for m in metrics])
    print(f"Accuracy Médio: {avg_accuracy:.2f}")


if __name__ == "__main__":
    main()

