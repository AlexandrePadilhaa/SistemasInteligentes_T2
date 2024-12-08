import json
from matplotlib import pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.tree import export_text, plot_tree
from utils import load_and_preprocess_data, min_max, split_cross_validation, prepare_train_test_data, separate_features_and_labels, save_summary, make_serializable


DATA_FILE = "data/treino_sinais_vitais_com_label.txt"
N_ESTIMATORS = 10         # Número de árvores 100 é o padrao
CRITERION = "gini"         # Critério para medir qualidade da divisão ("gini" ou "entropy")
MAX_DEPTH = None           # Profundidade máxima das árvores
MIN_SAMPLES_SPLIT = 2      # Número mínimo de amostras para dividir um nó
MIN_SAMPLES_LEAF = 1       # Número mínimo de amostras em cada folha
MAX_FEATURES = "sqrt"      # Recursos considerados por divisão ("sqrt", "log2", ou um número inteiro)
BOOTSTRAP = True           # Amostlragem com reposição
OOB_SCORE = True          # Habilitar validação Out-Of-Bag (se `True`, usa `bootstrap=True`)
CCP_ALPHA = 0.0            # Complexidade mínima para poda
RANDOM_STATE = 42          # Semente para reprodutibilidade
FOLDS = 5                 # Número de folds para validação cruzada
TIPO = 'classificacao'     # Tipo de modelo (classificação/regressão)


SUMMARY_FILE = 'results/random_forest/summary1.json'

def train_random_forest(x_train, y_train):
    rf = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        criterion=CRITERION,
        max_depth=MAX_DEPTH,
        min_samples_split=MIN_SAMPLES_SPLIT,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        max_features=MAX_FEATURES,
        bootstrap=BOOTSTRAP,
        oob_score=OOB_SCORE,
        ccp_alpha=CCP_ALPHA,
        random_state=RANDOM_STATE,
    )
    rf.fit(x_train, y_train)
    return rf

def summarize_forest_metrics(model):
    tree_depths = []
    tree_impurities = []
    tree_node_counts = []
    feature_importances = model.feature_importances_
    splits_per_level = []
    information_gains = []
    
    # Verificar se o modelo foi treinado com OOB (Out-of-Bag)
    oob_error = 1 - model.oob_score_ if hasattr(model, 'oob_score_') else None

    for idx, estimator in enumerate(model.estimators_):
        tree_depth = estimator.tree_.max_depth
        node_count = estimator.tree_.node_count
        impurity = np.mean(estimator.tree_.impurity)
        splits = node_count / tree_depth if tree_depth > 0 else 0

        # Calcular ganho médio de informação por divisão
        feature_splits = estimator.tree_.feature
        unique_splits = set(feature_splits[feature_splits != -2])  # Filtrar divisões válidas
        avg_info_gain = impurity / len(unique_splits) if unique_splits else 0

        # Acumular métricas por árvore
        tree_depths.append(tree_depth)
        tree_node_counts.append(node_count)
        tree_impurities.append(impurity)
        splits_per_level.append(splits)
        information_gains.append(avg_info_gain)

    # Cálculo das médias
    summary = {
        "average_tree_depth": np.mean(tree_depths),
        "average_tree_impurity": np.mean(tree_impurities),
        "average_tree_size": np.mean(tree_node_counts),
        "average_splits_per_level": np.mean(splits_per_level),
        "average_information_gain_per_split": np.mean(information_gains),
        "feature_importance": feature_importances.tolist(),
        "oob_error": oob_error
    }

    return summary


def evaluate_model(fold, model, x_test, y_test):
    y_pred = model.predict(x_test)
    conf_matrix = confusion_matrix(y_test, y_pred).tolist()
    report = classification_report(y_test, y_pred, output_dict=True)
    accuracy = accuracy_score(y_test, y_pred)

    print("\nConfusion Matrix:\n", conf_matrix)
    print("\nClassification Report:\n", json.dumps(report, indent=4))
    print("\nAccuracy Score:", accuracy)

    return {
        "fold": fold,
        "confusion_matrix": conf_matrix,
        "classification_report": report,
    }

def cross_validate_random_forest(df, folds=FOLDS):
    fold_data = split_cross_validation(df, folds)
    metrics_list = []

    parameters = {
        "n_estimators": N_ESTIMATORS,
        "criterion": CRITERION,
        "max_depth": MAX_DEPTH,
        "min_samples_split": MIN_SAMPLES_SPLIT,
        "min_samples_leaf": MIN_SAMPLES_LEAF,
        "max_features": MAX_FEATURES,
        "bootstrap": BOOTSTRAP,
        "oob_score": OOB_SCORE,
        "ccp_alpha": CCP_ALPHA,
        "random_state": RANDOM_STATE,
    }
    metrics_list.append({"parameters": parameters})

    for i, fold in enumerate(fold_data):
        print(f"\n--- Fold {i+1} ---")
        
        df_train, df_test = prepare_train_test_data(df, fold)
        x_train, y_train = separate_features_and_labels(df_train, TIPO)
        x_test, y_test = separate_features_and_labels(df_test, TIPO)

    
        rf_model = train_random_forest(x_train, y_train)

        metrics = evaluate_model(i + 1, rf_model, x_test, y_test)
        metrics_list.append({"average_tree_stats": summarize_forest_metrics(rf_model)})
        metrics_list.append(metrics)

    save_summary(metrics_list, SUMMARY_FILE)

if __name__ == "__main__":
    # Carregar e preprocessar os dados
    filepath = DATA_FILE
    df = load_and_preprocess_data(filepath, TIPO)
    df = min_max(df)

    # Executar validação cruzada
    print(f"Executando validação cruzada com {FOLDS} folds...")
    cross_validate_random_forest(df)
