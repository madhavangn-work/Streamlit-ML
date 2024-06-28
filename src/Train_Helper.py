from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
import streamlit as st




def hpt(X, y, clf, params):

    grid_search = GridSearchCV(clf, params, cv=3, scoring='accuracy')

    with st.spinner("Finding best parameters for the model... Please Wait..."):
        grid_search.fit(X, y)

    return grid_search.best_params_



def do_KNN(X, y, n=3, tune=False) -> KNeighborsClassifier:

    if tune:
        tune_clf = KNeighborsClassifier()
        param_grid = {
            'n_neighbors': [3, 5, 7, 9, 11, 13, 15],
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan', 'minkowski']
        }

        best_params = hpt(X, y, clf=tune_clf, params=param_grid)

        clf = KNeighborsClassifier(**best_params)
    else:
        clf = KNeighborsClassifier(n)
    
    clf.fit(X, y)

    return clf

def do_DT(X, y, tune=False) -> DecisionTreeClassifier:
    if tune:
        tune_clf = DecisionTreeClassifier()
        param_grid = {
            'criterion': ['gini', 'entropy'],
            'splitter': ['best', 'random'],
            'max_depth': [None, 10, 20, 30, 40, 50],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        best_params = hpt(X, y, clf=tune_clf, params=param_grid)
        clf = DecisionTreeClassifier(**best_params)
    else:
        clf = DecisionTreeClassifier()
    clf.fit(X, y)
    return clf

def do_SVM(X, y, tune=False) -> SVC:
    if tune:
        tune_clf = SVC()
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
            'gamma': ['scale', 'auto']
        }
        best_params = hpt(X, y, clf=tune_clf, params=param_grid)
        clf = SVC(**best_params)
    else:
        clf = SVC()
    clf.fit(X, y)
    return clf

def do_MLP(X, y, tune=False) -> MLPClassifier:
    if tune:
        tune_clf = MLPClassifier(max_iter=1000)
        param_grid = {
            'hidden_layer_sizes': [(50, 50), (100, 100), (100,)],
            'activation': ['identity', 'logistic', 'tanh', 'relu'],
            'solver': ['lbfgs', 'sgd', 'adam'],
            'alpha': [0.0001, 0.001, 0.01, 0.1],
            'learning_rate': ['constant', 'invscaling', 'adaptive']
        }
        best_params = hpt(X, y, clf=tune_clf, params=param_grid)
        clf = MLPClassifier(**best_params)
    else:
        clf = MLPClassifier(max_iter=1000)
    clf.fit(X, y)
    return clf

def do_RFC(X, y, tune=False) -> RandomForestClassifier:
    if tune:
        tune_clf = RandomForestClassifier()
        param_grid = {
            'n_estimators': [10, 50, 100, 200],
            'criterion': ['gini', 'entropy'],
            'max_depth': [None, 10, 20, 30, 40, 50],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        best_params = hpt(X, y, clf=tune_clf, params=param_grid)
        clf = RandomForestClassifier(**best_params)
    else:
        clf = RandomForestClassifier()
    clf.fit(X, y)
    return clf