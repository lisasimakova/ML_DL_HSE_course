from sklearn.datasets import make_blobs, make_moons
import numpy as np
import pandas
import random
from typing import Callable, Union, NoReturn, Optional, Dict, Any, List

# Task 1

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



# Task 2
class DecisionTreeLeaf:
    def __init__(self, ys):
        values, counts = np.unique(ys, return_counts=True)
        self.y = values[np.argmax(counts)]
        self.probs = {val: count / sum(counts) for val, count in zip(values, counts)}

class DecisionTreeNode:
    def __init__(self, split_dim: int, split_value: float, 
                 left: Union['DecisionTreeNode', DecisionTreeLeaf], 
                 right: Union['DecisionTreeNode', DecisionTreeLeaf]):
        self.split_dim = split_dim
        self.split_value = split_value
        self.left = left
        self.right = right



# Task 3
class DecisionTreeClassifier:
    def __init__(self, criterion: str = "gini", max_depth: Optional[int] = None, min_samples_leaf: int = 1):
        self.criterion = gini if criterion == "gini" else entropy
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.root = None

    def _best_split(self, X: np.ndarray, y: np.ndarray):
        best_gain = 0
        best_split = None
        n_samples, n_features = X.shape
        if n_samples < 2:
            return None
        
        for feature in range(n_features):
            sorted_idx = np.argsort(X[:, feature])
            X_sorted, y_sorted = X[sorted_idx], y[sorted_idx]
            
            unique_thresholds = np.percentile(X_sorted[:, feature], np.linspace(10, 90, 9))
            
            for threshold in unique_thresholds:
                mask = X_sorted[:, feature] < threshold
                n_left = np.sum(mask)
                n_right = n_samples - n_left
                if n_left < self.min_samples_leaf or n_right < self.min_samples_leaf:
                    continue
                
                left_y, right_y = y_sorted[:n_left], y_sorted[n_left:]
                
                split_gain = gain(left_y, right_y, self.criterion)
                if split_gain > best_gain:
                    best_gain = split_gain
                    best_split = (feature, threshold)

        return best_split

    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int = 0):
        stack = [(X, y, depth, None, None)]
        nodes = {}
        root = None

        while stack:
            X, y, depth, parent_id, is_left = stack.pop()

            if len(set(y)) == 1 or (self.max_depth is not None and depth >= self.max_depth):
                leaf = DecisionTreeLeaf(y)
                if parent_id is not None:
                    if is_left:
                        nodes[parent_id].left = leaf
                    else:
                        nodes[parent_id].right = leaf
                else:
                    root = leaf
                continue

            best_split = self._best_split(X, y)
            if best_split is None:
                leaf = DecisionTreeLeaf(y)
                if parent_id is not None:
                    if is_left:
                        nodes[parent_id].left = leaf
                    else:
                        nodes[parent_id].right = leaf
                else:
                    root = leaf
                continue

            feature, threshold = best_split
            left_idx = X[:, feature] < threshold
            right_idx = ~left_idx

            node = DecisionTreeNode(feature, threshold, None, None)
            if parent_id is not None:
                if is_left:
                    nodes[parent_id].left = node
                else:
                    nodes[parent_id].right = node
            else:
                root = node

            node_id = len(nodes)
            nodes[node_id] = node

            stack.append((X[left_idx], y[left_idx], depth + 1, node_id, True))
            stack.append((X[right_idx], y[right_idx], depth + 1, node_id, False))

        return root


    def fit(self, X: np.ndarray, y: np.ndarray):
        self.root = self._build_tree(X, y)

    def _traverse_tree(self, x: np.ndarray, node: Union[DecisionTreeNode, DecisionTreeLeaf]):
        if isinstance(node, DecisionTreeLeaf):
            return node
        return self._traverse_tree(x, node.left if x[node.split_dim] < node.split_value else node.right)

    def predict_proba(self, X: np.ndarray) -> List[Dict[Any, float]]:
        return [self._traverse_tree(x, self.root).probs for x in X]

    def predict(self, X: np.ndarray) -> List[Any]:
        return [max(proba, key=proba.get) for proba in self.predict_proba(X)]
# Task 4
task4_dtc = DecisionTreeClassifier(criterion="entropy", max_depth=6, min_samples_leaf=1)