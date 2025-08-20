from sklearn.model_selection import train_test_split
import numpy as np
import pandas
import random
import copy
from catboost import CatBoostClassifier
from typing import Callable, Union, NoReturn, Optional, Dict, Any, List

# Task 0
def gini(x: np.ndarray) -> float:
    """
    Считает коэффициент Джини для массива меток x.
    """
    _, counts = np.unique(x, return_counts=True)
    p = counts / len(x)
    gini_index = np.sum(p * (1 - p))
    
    return gini_index
    
def entropy(x: np.ndarray) -> float:
    """
    Считает энтропию для массива меток x.
    """
    _, counts = np.unique(x, return_counts=True)
    p = counts / len(x)
    return -np.sum(p * np.log2(p))

def gain(left_y: np.ndarray, right_y: np.ndarray, criterion: Callable) -> float:
    """
    Считает информативность разбиения массива меток.

    Parameters
    ----------
    left_y : np.ndarray
        Левая часть разбиения.
    right_y : np.ndarray
        Правая часть разбиения.
    criterion : Callable
        Критерий разбиения.
    """
    n = len(left_y) + len(right_y)
    p_left = len(left_y) / n
    p_right = len(right_y) / n

    a = criterion(np.concatenate((left_y, right_y)))

    b = p_left * criterion(left_y)
    c = p_right * criterion(right_y)

    return a - b - c


# Task 1

class DecisionTree:
    def __init__(self, X, y, criterion="gini", max_depth=None, min_samples_leaf=1, max_features="auto"):
        self.X = X
        self.y = y
        self.criterion = gini if criterion == "gini" else entropy
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = int(np.sqrt(X.shape[1])) if max_features == "auto" else max_features
        self.tree = None
        self.oob_samples = None
        
        self._bootstrap()
        self._fit()

    def _bootstrap(self):
        n_samples = self.X.shape[0]
        #Рандомно с возращением определяем нашу выборку
        indices = np.random.choice(n_samples, size=n_samples, replace=True)

        # Считаем, сколько раз каждый индекс появился, bincount возвращает массив частот в отсортированном порядке
        counts = np.bincount(indices, minlength=n_samples)

        # OOB-индексы - те, у которых количество вхождений = 0, Out-of-bag выборка
        self.oob_samples = np.where(counts == 0)[0] #Создает кортеж с массивом: (array([2, 4]),), поэтому берем [0]

        self.X_train = self.X[indices]
        self.y_train = self.y[indices]

    
    def _fit(self):
        # Строим наше дерево
        self.tree = self._build_tree(self.X_train, self.y_train)

    # depth учитываем, чтобы выйти вовремя из рекурсии, на каждом шаге прибавляем +1
    # Рекурсивно строим дерево
    def _build_tree(self, X, y, depth=0):
        if len(set(y)) == 1 or len(y) < self.min_samples_leaf or (self.max_depth and depth >= self.max_depth):
            return {"leaf": True, "label": np.bincount(y).argmax()}
        
        feature = self._best_split(X, y)

        #Если мы не нашли лучшего разбиения, то
        if feature is None:
            return {"leaf": True, "label": np.bincount(y).argmax()}
        
        left_idx = X[:, feature] == 0
        right_idx = X[:, feature] == 1
        
        return {
            "leaf": False,
            "feature": feature,
            "left": self._build_tree(X[left_idx], y[left_idx], depth + 1),
            "right": self._build_tree(X[right_idx], y[right_idx], depth + 1)
        }
    
    def _best_split(self, X, y):
        # Выбирает именно те признаки, по которым будет искать лучшее!
        features = np.random.choice(X.shape[1], self.max_features, replace=False)
        best_feature = None
        # лучший gain - максимальный, ставим минимум, как начальный параметр
        best_gain = -np.inf
        
        for feature in features:
            #Выбираем строки, где значение признака feature равно 0, 1
            left_y, right_y = y[X[:, feature] == 0], y[X[:, feature] == 1]
            if len(left_y) >= self.min_samples_leaf and len(right_y) >= self.min_samples_leaf:
                score = gain(left_y, right_y, self.criterion)
                if score > best_gain:
                    best_gain = score
                    best_feature = feature
        
        return best_feature
    
    def _predict_one(self, x, node):
        if node["leaf"]:
            return node["label"]
        return self._predict_one(x, node["left" if x[node["feature"]] == 0 else "right"])
    
    def predict(self, X):
        return np.array([self._predict_one(x, self.tree) for x in X])
    
# Task 2

class RandomForestClassifier:
    def __init__(self, criterion="gini", max_depth=None, min_samples_leaf=1, max_features="auto", n_estimators=10):
        self.criterion = gini if criterion == "gini" else entropy
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.n_estimators = n_estimators
        self.forest = None

    def _predict_one(self, x, node):
        if node["leaf"]:
            return node["label"]
        branch = "left" if x[node["feature"]] == 0 else "right"
        return self._predict_one(x, node[branch])
    
    def fit(self, X, y):
        self.forest = []
        for i in range(self.n_estimators):
            tree = DecisionTree(X, y, self.criterion, self.max_depth, self.min_samples_leaf, self.max_features)
        #   tree.fit()  
            self.forest.append(tree.tree)
    
    def predict(self, X):
        answer = []
        for x in X:
            predictions = []
            for tree in self.forest:
                predictions.append(self._predict_one(x, tree))
            answer.append(np.bincount(predictions).argmax())
        return answer

    
# Task 3

def feature_importance(rfc):
    error = []
    for tree in rfc.forest:
        X = tree.X[tree.oob_samples] 
        y = tree.y[tree.oob_samples]
        n_samples, n_features = X.shape
        predictions = tree.predict(X)
        
        error_tree = []
        for i in range(n_features):
            X_i = np.copy(X)
            np.random.shuffle(X_i[:, i])
            
            predictions_i = tree.predict(X_i)
            err_oob = np.mean(y != predictions)
            err_oob_j = np.mean(y != predictions_i)
            
            error_tree.append(err_oob_j - err_oob)
        error.append(error_tree)

    return np.mean(error, axis=0)


# Task 4

rfc_age = None
rfc_gender = None

# Task 5
# Здесь нужно загрузить уже обученную модели
# https://catboost.ai/en/docs/concepts/python-reference_catboost_save_model
# https://catboost.ai/en/docs/concepts/python-reference_catboost_load_model
catboost_rfc_age = None
catboost_rfc_gender = None