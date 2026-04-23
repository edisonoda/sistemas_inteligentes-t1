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

def mostrar_melhor_hiperparametrizacao(model, clf, X_train, X_test, y_train, y_test):
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

    # com dados de teste
    y_pred_test = best.predict(X_test)
    acc_test = accuracy_score(y_test, y_pred_test) * 100
    prec_test = precision_score(y_test, y_pred_test, average='weighted') * 100
    recall_test = recall_score(y_test, y_pred_test, average='weighted') * 100
    f1_test = f1_score(y_test, y_pred_test, average='weighted')

    train_f1_v = clf.cv_results_['mean_train_score'][clf.best_index_]
    test_f1_v = clf.cv_results_['mean_test_score'][clf.best_index_]

    print(f"- Acuracia (treino | teste): {acc_train:.2f}% | {acc_test:.2f}%")
    print(f"- Precisao (treino | teste): {prec_train:.2f}% | {prec_test:.2f}%")
    print(f"- Recall (treino | teste): {recall_train:.2f}% | {recall_test:.2f}%")
    print(f"- F1 ponderado (treino | teste): {f1_train:.4f} | {f1_test:.4f}")
    print(f"- F1-score (treino | teste): {train_f1_v:.4f} | {test_f1_v:.4f}")

    # Matriz de confusão
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred_test)
    print(f"\nMatriz de confusão:")
    print(classification_report(y_test, y_pred_test))

    # plt.show()


def classificador_cart(k_folds, X_train, X_test, y_train, y_test):
    parameters = {
        'criterion': ['gini'],
        'max_depth': [15],
        'min_samples_leaf': [8]
        # 'criterion': ['entropy', 'gini'],
        # 'max_depth': [1, 5, 10, 15, 20],
        # 'min_samples_leaf': [1, 2, 4, 8, 16]
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
        'hidden_layer_sizes': [(16, 16)],
        'activation': ['identity'],
        'learning_rate_init': [0.03]
        # 'hidden_layer_sizes': [(32,), (16, 16), (16, 8, 4)],
        # 'activation': ['identity', 'tanh', 'relu'],
        # 'learning_rate_init': [0.01, 0.03, 0.05]
    }

    # instantiate model
    model = MLPClassifier(random_state=seed)

    # grid search using cv
    clf_rn = GridSearchCV(model, parameters, cv=k_folds, scoring='f1_weighted', verbose=4, return_train_score=True)
    clf_rn.fit(X_train, y_train)
    
    return clf_rn


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

    mostrar_melhor_hiperparametrizacao("CART", clf_cart, X_train, X_test, y_train, y_test)
    mostrar_melhor_hiperparametrizacao("MLP", clf_rn, X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    main()