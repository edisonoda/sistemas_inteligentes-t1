from gerar_dados_vitimas import gerar_dataset_vitimas

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

from sklearn.neural_network import MLPClassifier

import matplotlib.pyplot as plt
from sklearn import tree

from sklearn.metrics import ConfusionMatrixDisplay

seed = 21

def mostrar_melhor_hiperparametrizacao(best, X_train, X_test, y_train, y_test):
    # Melhor Hiperparametrização
    print(f"\nMelhor Hiperparametrização:")

    # com dados do treinamento
    y_pred_train = best.predict(X_train)
    acc_train = accuracy_score(y_train, y_pred_train) * 100
    print(f"- Acuracia com dados de treino: {acc_train:.2f}%")

    # com dados de teste
    y_pred_test = best.predict(X_test)
    acc_test = accuracy_score(y_test, y_pred_test) * 100
    print(f"- Acuracia com dados de teste: {acc_test:.2f}%")

    print("- Parâmetros:", best)

    # Matriz de confusão
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred_test)
    print(f"\nMatriz de confusão:")
    print(classification_report(y_test, y_pred_test))


def classificador_cart(k_folds, X_train, X_test, y_train, y_test):
    parameters = {
        'criterion': ['entropy', 'gini'],
        'max_depth': [1, 5, 10, 15, 20],
        'min_samples_leaf': [1, 2, 4, 8, 16]
    }

    # instantiate model
    model = DecisionTreeClassifier(random_state=seed)

    # grid search using cv
    clf_cart = GridSearchCV(model, parameters, cv=k_folds, scoring='f1_weighted', verbose=4)
    clf_cart.fit(X_train, y_train)

    # fig = plt.figure(figsize=(16, 9))
    # tree.plot_tree(best, filled=True, rounded=True, class_names=True, fontsize=8)
    # plt.show()

    return clf_cart.best_estimator_


def classificador_rn(k_folds, X_train, X_test, y_train, y_test):
    parameters = {
        'hidden_layer_sizes': [(32,), (16, 16), (16, 4)],
        'activation': ['identity', 'tanh', 'relu'],
        'learning_rate_init': [0.01, 0.03, 0.05]
    }

    # instantiate model
    model = MLPClassifier(random_state=seed)

    # grid search using cv
    clf_rn = GridSearchCV(model, parameters, cv=k_folds, scoring='f1_weighted', verbose=4)
    clf_rn.fit(X_train, y_train)
    
    return clf_rn.best_estimator_


def main():
    # Criação do dataset
    df = gerar_dataset_vitimas(
        n_vitimas=5000,
        media_idade=35,
        desvio_idade=20,
        tipo_acidente="uniforme",
        nivel_ruido=0.05,
        seed=seed
    )
    print("\nPrimeiras linhas do dataset gerado:")
    print(df.head())

    # Validação cruzada
    k_folds = 5

    x = df.drop(columns=['gcs', 'avpu', 'tri', 'sobr'])
    y = df.get(['tri'])

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, shuffle=True)

    best_cart = classificador_cart(k_folds, X_train, X_test, y_train, y_test)
    best_rn = classificador_rn(k_folds, X_train, X_test, y_train, y_test)

    mostrar_melhor_hiperparametrizacao(best_cart, X_train, X_test, y_train, y_test)
    mostrar_melhor_hiperparametrizacao(best_rn, X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    main()