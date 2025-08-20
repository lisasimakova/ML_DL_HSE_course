import numpy as np
import pandas
from typing import NoReturn, Tuple, List

# Task 1

def read_cancer_dataset(path_to_csv: str) -> Tuple[np.array, np.array]:
    """
     
    Parameters
    ----------
    path_to_csv : str
        Путь к cancer датасету.

    Returns
    -------
    X : np.array
    
        Матрица признаков опухолей.
    y : np.array
        Вектор бинарных меток, 1 соответствует доброкачественной опухоли (M), 
        0 --- злокачественной (B).

    
    """
    data = pandas.read_csv(path_to_csv).sample(frac=1)
    X = data.drop(columns=['label']).values
    y = (data['label'] == 'M').astype(int).values 
    return X, y

def read_spam_dataset(path_to_csv: str) -> Tuple[np.array, np.array]:
    """
     
    Parameters
    ----------
    path_to_csv : str
        Путь к spam датасету.

    Returns
    -------
    X : np.array
        Матрица признаков сообщений.
    y : np.array
        Вектор бинарных меток, 
        1 если сообщение содержит спам, 0 если не содержит.
    
    """
    data = pandas.read_csv(path_to_csv).sample(frac=1)
    X = data.drop(columns=['label']).values 
    y = data['label'].values  
    return X, y


# Task 2

def train_test_split(X: np.array, y: np.array, ratio: float) -> Tuple[np.array, np.array, np.array, np.array]:
    """

    Parameters
    ----------
    X : np.array
        Матрица признаков.
    y : np.array
        Вектор меток.
    ratio : float
        Коэффициент разделения.

    Returns
    -------
    X_train : np.array
        Матрица признаков для train выборки.
    y_train : np.array
        Вектор меток для train выборки.
    X_test : np.array
        Матрица признаков для test выборки.
    y_test : np.array
        Вектор меток для test выборки.

    """
    n_all = len(X)
    n_train = round(n_all * ratio)
    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]


    return X_train, y_train, X_test, y_test


    
# Task 3

def get_precision_recall_accuracy(y_pred: np.array, y_true: np.array) -> Tuple[np.array, np.array, float]:
    """

    Parameters
    ----------
    y_pred : np.array
        Вектор классов, предсказанных моделью.
    y_true : np.array
        Вектор истинных классов.

    Returns
    -------
    precision : np.array
        Вектор с precision для каждого класса.
    recall : np.array
        Вектор с recall для каждого класса.
    accuracy : float
        Значение метрики accuracy (одно для всех классов).

    """
    precision = []
    recall = []
    for i in np.unique(y_true):
        TP = np.sum((y_pred == i) & (y_true == i))
        FP = np.sum(y_pred == i) - TP
        if TP+FP:
            precision.append(TP/(TP+FP))
        else:
            precision.append(0)
        FN = np.sum(y_true == i) - TP
        if TP+FN:
            recall.append(TP/(TP+FN))
        else:
            recall.append(0)       
    accuracy = np.sum(y_pred == y_true)/len(y_pred)
    return np.array(precision), np.array(recall), accuracy
    
# Task 4
"""
class KDTreeNode:
    def __init__(self, objs=None, median=None, dimension=None):
        self.objs = objs  # Массив индексов точек (если это лист)
        self.median = median  # Значение медианы (если это внутренний узел)
        self.dimension = dimension  # Ось разбиения
        self.left = None  # Левое поддерево
        self.right = None  # Правое поддерево

class KDTree:
    def __init__(self, X: np.array, leaf_size: int = 40):
        self.X = X
        self.leaf_size = leaf_size
        self.tree = self.build_kdtree(X, X.shape[1], np.arange(X.shape[0]), leaf_size)

    def build_kdtree(self, X, n_features, ind, leaf_size, n=0):
        n_ind = len(ind)

        # Если количество точек меньше или равно leaf_size, то это листовой узел
        if n_ind <= leaf_size:
            return KDTreeNode(objs=ind)

        dimension = n % n_features  # Определяем ось для разбиения
        select_obj = X[ind]  # Выбираем подмассив точек по индексам

        # Используем np.partition для нахождения медианы
        median_idx = n_ind // 2
        partitioned_indices = np.argpartition(select_obj[:, dimension], median_idx)

        # Индексы разделяем напрямую, без использования списков
        left_indices = ind[partitioned_indices[:median_idx]]
        right_indices = ind[partitioned_indices[median_idx:]]

        # Медианное значение
        median_value = select_obj[partitioned_indices[median_idx], dimension]

        # Создаем узел дерева
        node = KDTreeNode(median=median_value, dimension=dimension)

        # Рекурсивное построение дерева
        node.left = self.build_kdtree(X, n_features, left_indices, leaf_size, n + 1)
        node.right = self.build_kdtree(X, n_features, right_indices, leaf_size, n + 1)

        return node




    def euclidean_distance(self, p1, p2):
        return np.sum((p1 - p2) ** 2)

    def query(self, X: np.array, k: int = 1) -> List[List[int]]:
        # Результаты для всех запросов
        answer = []

        # Вспомогательная рекурсивная функция для поиска ближайших соседей
        def nearest_neighbors(x, k, tree, result):
            if tree.objs is not None:  # Если узел - лист
                for i in tree.objs:
                    dist = self.euclidean_distance(x, self.X[i])
                    if len(result) < k:
                        bisect.insort_right(result, (dist, i))
                    elif dist < result[-1][0]:
                        bisect.insort_right(result, (dist, i))
                        result.pop()
                        
                return result

            dimension = tree.dimension
            median = tree.median

            # Определяем порядок обхода поддеревьев
            f, s = (tree.left, tree.right) if x[dimension] <= median else (tree.right, tree.left)
            result = nearest_neighbors(x, k, f, result)

            # Проверяем, стоит ли заходить во второе поддерево
            if len(result) < k or abs(x[dimension] - median) ** 2 < result[-1][0]:
                result = nearest_neighbors(x, k, s, result)

            return result

        # Обрабатываем все точки в X_test
        for x in X:
            result = []
            result = nearest_neighbors(x, k, self.tree, result)
            result_ind = [b for a, b in result]
            answer.append(result_ind)

        return answer
    
    """


class KDTreeNode:
    def __init__(self, objs=None, median=None, dimension=None):
        self.objs = objs  # Массив индексов точек (если это лист)
        self.median = median  # Значение медианы (если это внутренний узел)
        self.dimension = dimension  # Ось разбиения
        self.left = None  # Левое поддерево
        self.right = None  # Правое поддерево

class KDTree:
    def __init__(self, X: np.array, leaf_size: int = 40):
        self.X = X
        self.leaf_size = leaf_size
        self.tree = self.build_kdtree(X, X.shape[1], np.arange(X.shape[0]), leaf_size)

    def build_kdtree(self, X, n_features, ind, leaf_size, n=0):
        n_ind = len(ind)

        if n_ind <= leaf_size:
            return KDTreeNode(objs=ind)

        dimension = n % n_features  
        select_obj = X[ind]  

        median_idx = n_ind // 2
        partitioned_indices = np.argpartition(select_obj[:, dimension], median_idx)

        left_indices = ind[partitioned_indices[:median_idx]]
        right_indices = ind[partitioned_indices[median_idx:]]

        median_value = select_obj[partitioned_indices[median_idx], dimension]

        node = KDTreeNode(median=median_value, dimension=dimension)

        node.left = self.build_kdtree(X, n_features, left_indices, leaf_size, n + 1)
        node.right = self.build_kdtree(X, n_features, right_indices, leaf_size, n + 1)

        return node
    

    def query(self, X: np.array, k: int = 1) -> List[List[int]]:
        answer = []

        def nearest_neighbors(x, k, tree, result):
            if tree.objs is not None:
                distances = np.sum((self.X[tree.objs] - x) ** 2, axis=1)
                cur_result = np.column_stack((distances, tree.objs))
                result = np.concatenate((result, cur_result))
                result = result[np.argsort(result[:, 0])][:k]
                return result

            dimension = tree.dimension
            median = tree.median

            f, s = (tree.left, tree.right) if x[dimension] < median else (tree.right, tree.left)
            result = nearest_neighbors(x, k, f, result)

            if abs(x[dimension] - median) ** 2 < result[-1][0] or len(result) < k:
                result = nearest_neighbors(x, k, s, result)

            return result

        for x in X:
            result = np.empty((0, 2)) 
            result = nearest_neighbors(x, k, self.tree, result)
            answer.append(result[:, 1].astype(int).tolist())

        return answer

"""
    def query(self, X: np.array, k: int = 1) -> list:
        results = []
        
        def search(tree, point, k, heap):
            if tree is None:
                return
            
            if tree.objs is not None:  # Листовой узел
                for idx in tree.objs:  # Используем только индексы
                    dist = self.euclidean_distance(point, self.X[idx])  # Получаем точку из массива X по индексу
                    if len(heap) < k:
                        heapq.heappush(heap, (-dist, idx))
                    elif dist < -heap[0][0]:
                        heapq.heappushpop(heap, (-dist, idx))
                return

            axis = tree.dimension
            diff = point[axis] - tree.median

            primary, secondary = (tree.left, tree.right) if diff <= 0 else (tree.right, tree.left)

            search(primary, point, k, heap)

            if len(heap) < k or diff ** 2 < -heap[0][0]:
                search(secondary, point, k, heap)

        for point in X:
            heap = []
            search(self.tree, point, k, heap)
            results.append([b for a, b in heap])
        
        return results

"""

        
# Task 5

class KNearest:
    def __init__(self, n_neighbors: int = 5, leaf_size: int = 30):

        """

        Parameters
        ----------
        n_neighbors : int
            Число соседей, по которым предсказывается класс.
        leaf_size : int
            Минимальный размер листа в KD-дереве.

        """        
        self.tree = None
        self.X = None
        self.y = None
        self.n_neighbors = n_neighbors
        self.leaf_size = leaf_size


        return    

    def fit(self, X: np.array, y: np.array) -> NoReturn:

        """

        Parameters
        ----------
        X : np.array
            Набор точек, по которым строится классификатор.
        y : np.array
            Метки точек, по которым строится классификатор.

        """
        self.X = X
        self.y = y
        self.tree = KDTree(X)
        return 
        
    def predict_proba(self, X: np.array) -> List[np.array]:
        """

        Parameters
        ----------
        X : np.array
            Набор точек, для которых нужно определить класс.
        
        Returns
        -------
        list[np.array]
            Список np.array (длина каждого np.array равна числу классов):
            вероятности классов для каждой точки X.
            

        """
        classes = np.unique(self.y)
        n_classes = len(classes)
        result_proba = np.zeros((len(X), n_classes))  
        result_neightbors = self.tree.query(X)

        for i, n in enumerate(result_neightbors):
            unique_values, counts = np.unique(self.y[n], return_counts=True)
            k = 0
            for j, cl in enumerate(classes):
                if cl in unique_values:
                    result_proba[i, j] = counts[k] / self.n_neighbors
                    k += 1
        
        return result_proba
        
    def predict(self, X: np.array) -> np.array:
        """

        Parameters
        ----------
        X : np.array
            Набор точек, для которых нужно определить класс.
        
        Returns
        -------
        np.array
            Вектор предсказанных классов.
            

        """
        predict = self.predict_proba(X)

        return np.argmax(predict, axis=1)
