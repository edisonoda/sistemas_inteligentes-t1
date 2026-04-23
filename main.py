import numpy as np
import pandas as pd

from gerar_dados_vitimas import gerar_dataset_vitimas

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report

from sklearn.neural_network import MLPClassifier

import matplotlib.pyplot as plt
from sklearn import tree

from sklearn.metrics import ConfusionMatrixDisplay

seed = 21

def calcular_variancia(clf, k_folds, which):
    scores = []

    for i in range(k_folds):
        scores.append(clf.cv_results_[f'split{i}_{which}_score'])
    
    return np.var(np.array(scores), ddof=1)

def mostrar_melhor_hiperparametrizacao(model, clf, X_train, X_test, y_train, y_test, k_folds):
    # Melhor Hiperparametrização
    print(f"\nMelhor Hiperparametrização ({ model }):")
    print("- Parâmetros:", clf.best_params_)

    best = clf.best_estimator_

    # com dados do treinamento
    y_pred_train = best.predict(X_train)
    acc_train = accuracy_score(y_train, y_pred_train) * 100
    prec_train = precision_score(y_train, y_pred_train, average='weighted') * 100
    recall_train = recall_score(y_train, y_pred_train, average='weighted') * 100
    f1_train = f1_score(y_train, y_pred_train, average='weighted')

    train_f1_vies = clf.cv_results_['mean_train_score'][clf.best_index_]
    var_train = calcular_variancia(clf, k_folds, 'train')

    # com dados de teste
    y_pred_test = best.predict(X_test)
    acc_test = accuracy_score(y_test, y_pred_test) * 100
    prec_test = precision_score(y_test, y_pred_test, average='weighted') * 100
    recall_test = recall_score(y_test, y_pred_test, average='weighted') * 100
    f1_test = f1_score(y_test, y_pred_test, average='weighted')

    test_f1_vies = clf.cv_results_['mean_test_score'][clf.best_index_]
    var_test = calcular_variancia(clf, k_folds, 'test')

    print(f"- Acuracia (treino | teste): {acc_train:.2f}% | {acc_test:.2f}%")
    print(f"- Precisao (treino | teste): {prec_train:.2f}% | {prec_test:.2f}%")
    print(f"- Recall (treino | teste): {recall_train:.2f}% | {recall_test:.2f}%")
    print(f"- F1-score weighted (treino | teste): {f1_train:.4f} | {f1_test:.4f}")
    print(f"- F1-score (treino | teste): {train_f1_vies:.4f} | {test_f1_vies:.4f}")
    print(f"\t- Dif: {abs(train_f1_vies - test_f1_vies):.4f}")
    print(f"- Variancia (treino | teste): {var_train:.8f} | {var_test:.8f}")
    print(f"\t- Dif: {abs(var_train - var_test):.8f}")

    # Matriz de confusão
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred_test)
    print(f"\nMatriz de confusão:")
    print(classification_report(y_test, y_pred_test))

    # plt.show()

def classificador_cart(k_folds, X_train, X_test, y_train, y_test):
    parameters = {
        # 'criterion': ['gini'],
        # 'max_depth': [15],
        # 'min_samples_leaf': [8]
        'criterion': ['entropy', 'gini'],
        'max_depth': [1, 5, 10, 15, 20],
        'min_samples_leaf': [1, 2, 4, 8, 16]
    }

    # instantiate model
    model = DecisionTreeClassifier(random_state=seed)

    # grid search using cv
    clf_cart = GridSearchCV(model, parameters, cv=k_folds, scoring='f1_weighted', verbose=4, return_train_score=True)
    clf_cart.fit(X_train, y_train)

    # fig = plt.figure(figsize=(16, 9))
    # tree.plot_tree(best, filled=True, rounded=True, class_names=True, fontsize=8)
    # plt.show()

    return clf_cart


def classificador_rn(k_folds, X_train, X_test, y_train, y_test):
    parameters = {
        # 'hidden_layer_sizes': [(16, 16)],
        # 'activation': ['identity'],
        # 'learning_rate_init': [0.03]
        'hidden_layer_sizes': [(32,), (16, 16), (16, 8, 4)],
        'activation': ['identity', 'tanh', 'relu'],
        'learning_rate_init': [0.01, 0.03, 0.05]
    }

    # instantiate model
    model = MLPClassifier(random_state=seed)

    # grid search using cv
    clf_rn = GridSearchCV(model, parameters, cv=k_folds, scoring='f1_weighted', verbose=4, return_train_score=True)
    clf_rn.fit(X_train, y_train)
    
    return clf_rn


def mostrar_metricas_predicao(model, y, y_pred):
    prec = precision_score(y, y_pred, average='weighted') * 100
    recall = recall_score(y, y_pred, average='weighted') * 100
    f1 = f1_score(y, y_pred, average='weighted')

    print(f"\nTeste Cego ({ model }):")
    print(f"- Precisao: {prec:.2f}%")
    print(f"- Recall: {recall:.2f}%")
    print(f"- F1-score: {f1:.4f}")

    # Matriz de confusão
    ConfusionMatrixDisplay.from_predictions(y, y_pred)
    print(f"\nMatriz de confusão:")
    print(classification_report(y, y_pred))

    plt.show()

def teste_cego(clf, X_train, y_train, x, y):
    best = clf.best_estimator_
    best.fit(X_train, y_train)
    return best.predict(x)


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

    clf_cart = classificador_cart(k_folds, X_train, X_test, y_train, y_test)
    clf_rn = classificador_rn(k_folds, X_train, X_test, y_train, y_test)

    mostrar_melhor_hiperparametrizacao("CART", clf_cart, X_train, X_test, y_train, y_test, k_folds)
    mostrar_melhor_hiperparametrizacao("MLP", clf_rn, X_train, X_test, y_train, y_test, k_folds)

    # Testes cegos
    df_blind = pd.read_csv("datasets/1000v/data.csv")

    x_blind = df_blind.drop(columns=['gcs', 'avpu', 'tri', 'sobr'])
    y_blind = df_blind.get(['tri'])

    pred_cart = teste_cego(clf_cart, X_train, y_train, x_blind, y_blind)
    pred_rn = teste_cego(clf_rn, X_train, y_train, x_blind, y_blind)

    mostrar_metricas_predicao('CART', y_blind, pred_cart)
    mostrar_metricas_predicao('MLP', y_blind, pred_rn)

if __name__ == "__main__":
    main()