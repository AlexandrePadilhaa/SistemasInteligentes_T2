from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from utils import load_and_preprocess_data, min_max, split_cross_validation, prepare_train_test_data, separate_features_and_labels

DATA_FILE = "data/treino_sinais_vitais_com_label.txt"
N_ESTIMATORS = 100        # Número de árvores
MAX_DEPTH = None           
MIN_SAMPLES_SPLIT = 2      # Número mínimo de amostras para dividir um nó
RANDOM_STATE = 42          # Semente para reprodutibilidade
MAX_FEATURES = "sqrt"      # Atributos considerados por divisão ("sqrt" = raiz quadrada do total)
FOLDS = 5
TIPO = 'classificacao'

def train_random_forest(x_train, y_train):
    """
    Treina um modelo Random Forest com os parâmetros especificados.
    """
    rf = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        min_samples_split=MIN_SAMPLES_SPLIT,
        random_state=RANDOM_STATE,
        max_features=MAX_FEATURES,
    )
    rf.fit(x_train, y_train)
    return rf

def evaluate_model(model, x_test, y_test):
    """
    Avalia o modelo nos dados de teste.
    """
    y_pred = model.predict(x_test)
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("\nAccuracy Score:", accuracy_score(y_test, y_pred))

if __name__ == "__main__":
    # Carregar e preprocessar os dados
    filepath = DATA_FILE
    df = load_and_preprocess_data(filepath,TIPO)
    df = min_max(df)

    # Validação cruzada
    folds = split_cross_validation(df, FOLDS)

    for i, fold in enumerate(folds):
        print(f"\n--- Fold {i+1} ---")
        # Preparar dados de treino e teste
        df_train, df_test = prepare_train_test_data(df, fold)
        x_train, y_train = separate_features_and_labels(df_train,TIPO)
        x_test, y_test = separate_features_and_labels(df_test,TIPO)

        # Treinar o modelo Random Forest
        rf_model = train_random_forest(x_train, y_train)

        # Avaliar o modelo
        evaluate_model(rf_model, x_test, y_test)
