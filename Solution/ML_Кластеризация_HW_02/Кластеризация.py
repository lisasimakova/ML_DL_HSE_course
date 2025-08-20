from sklearn.neighbors import KDTree
import numpy as np
import random
import copy
from collections import deque
from typing import NoReturn

class KMeans:
    def __init__(self, n_clusters: int, init: str = "random", max_iter: int = 300):
        """
        Parameters
        ----------
        n_clusters : int
            Число итоговых кластеров при кластеризации.
        init : str
            Способ инициализации кластеров. Один из трех вариантов:
            1. random --- центроиды кластеров являются случайными точками,
            2. sample --- центроиды кластеров выбираются случайно из X,
            3. k-means++ --- центроиды кластеров инициализируются 
                при помощи метода K-means++.
        max_iter : int
            Максимальное число итераций для kmeans.
        """
        self.n_clusters = n_clusters
        self.init = init
        self.max_iter = max_iter
        self.centroids = None
        
    def fit(self, X: np.array, y = None) -> NoReturn:
        """
        Ищет и запоминает в self.centroids центроиды кластеров для X.
        
        Parameters
        ----------
        X : np.array
            Набор данных, который необходимо кластеризовать.
        y : Ignored
            Не используемый параметр.
        """
        # Инициализация центроидов в зависимости от метода
        if self.init == 'random':
            self.centroids = np.random.uniform(low=np.min(X, axis=0),
                                               high=np.max(X, axis=0), 
                                               size=(self.n_clusters, X.shape[1]))
        elif self.init == 'sample':
            #Передаем в рандом индексы, колво кластеров, без возвращения
            self.centroids = X[np.random.choice(X.shape[0], self.n_clusters, replace=False)]
        else:
            self.centroids = self.find_centroids(X)
        
        for _ in range(self.max_iter):
            labels = self.defining_clusters(X)
            self.find_empty_clusters(X, labels)  # Поиск пустых кластеров
            new_centroids = self.new_centroids(X, labels)  # Обновление центроидов
            #Проверяем, являются ли массивы почти равными
            if np.allclose(self.centroids, new_centroids):
                break
            self.centroids = new_centroids

    def find_centroids(self, X: np.array) -> np.array:
        """
        Инициализация центроидов методом K-Means++.
        """
        #Алгоритм поиска центройдов по википедиии
        first_idx = np.random.choice(X.shape[0], 1)
        self.centroids = np.array([X[first_idx[0]]])

        while len(self.centroids) < self.n_clusters:
            #np.newaxis добавляем размерность, чтобы из каждого значения вычесть каждый центройд, потом берем минимум

            distances = np.min(np.sum((X[:, np.newaxis] - self.centroids) ** 2, axis=2), axis=1)
            total_dist = np.sum(distances)
            rnd = np.random.uniform(0, total_dist)
            
            cumulative_dist = 0
            for i, d in enumerate(distances):
                cumulative_dist += d
                if cumulative_dist > rnd:
                    self.centroids = np.vstack([self.centroids, X[i]])
                    break
        return self.centroids

    def predict(self, X: np.array) -> np.array:
        """
        Для каждого элемента из X возвращает номер кластера.
        """
        return self.defining_clusters(X)

    def defining_clusters(self, X: np.array) -> np.array:
        """
        Для каждого элемента из X вычисляется ближайший кластер.
        """
        #Добавляем размерность, чтобы из каждого X вычесть каждый центройд потом берем евклидову метрику
        distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
        #Возвращаем индексы минимальных центройдов
        return np.argmin(distances, axis=1)

    def new_centroids(self, X: np.array, labels: np.array) -> np.array:
        """
        Пересчитывает новые центроиды по меткам.
        """
        centroids = []
        for i in range(self.n_clusters):
            #Если есть кто-то из кластера i
            if np.any(labels == i):
                centroids.append(X[labels == i].mean(axis=0))
            else:
                # В случае пустого кластера, выбираем случайный элемент из X
                centroids.append(X[np.random.choice(X.shape[0])])
        return np.array(centroids)

    def find_empty_clusters(self, X: np.array, labels: np.array) -> None:
        """
        Проверяет, есть ли пустые кластеры и обновляет их центроиды.
        """
        for k in range(self.n_clusters):
            if not np.any(labels == k):
                # Если кластер пуст, выбираем новый случайный элемент
                self.centroids[k] = X[np.random.choice(X.shape[0])]

# Task 2


class DBScan:
    def __init__(self, eps: float = 0.5, min_samples: int = 5, 
                 leaf_size: int = 40, metric: str = "euclidean"):
        """
        Инициализация параметров DBSCAN.
        """
        self.eps = eps
        self.min_samples = min_samples
        self.leaf_size = leaf_size
        self.metric = metric
        # 0-шум, 1-сосед, 2-центр
        self.tags = None
        self.labels = None

    def fit_predict(self, X: np.array, y=None) -> np.array:
        """
        Кластеризует элементы из X, для каждого возвращает индекс соотв. кластера.
        """
        n = X.shape[0]

        tree = KDTree(X, metric=self.metric, leaf_size=self.leaf_size)
        neighbors = tree.query_radius(X, self.eps, count_only=False)  # Соседи для каждой точки
        self.labels = np.full(n, -1)  # Инициализация меток кластеров (-1 - неопределено)
        self.tags = np.full(n, -1)    # Инициализация тегов (-1 - не посещена, 0 - шум, 1 - точка, часть кластера, 2 - центр)
        cur_cluster = 0

        for i in range(n):
            if self.tags[i] != -1:  # Точка уже посещена (была помечена как шум или как часть кластера)
                continue

            indices = neighbors[i]
            if len(indices) < self.min_samples:  # Если количество соседей меньше минимального, помечаем как шум
                self.tags[i] = 0  # Помечаем как шум
                continue
            
            # Если точка является центром, начинаем кластеризацию
            self.labels[i] = cur_cluster
            self.tags[i] = 2  # Помечаем как центр
            queue = deque(indices)
            visited = set([i])  # Множество для отслеживания посещенных точек

            while queue:
                cur_nei = queue.popleft()  # Берём первую точку из очереди
                if self.tags[cur_nei] == 0:  # Если точка была шумом, меняем метку на "часть кластера"
                    self.tags[cur_nei] = 1  # Меняем метку на часть кластера
                    self.labels[cur_nei] = cur_cluster
                if self.tags[cur_nei] == -1:  # Если точка ещё не посещена
                    self.tags[cur_nei] = 1  # Помечаем как точку, которая может быть частью кластера
                    self.labels[cur_nei] = cur_cluster
                
                indices = neighbors[cur_nei]
                if len(indices) >= self.min_samples:  # Если в окрестности достаточно точек, расширяем кластер
                    for nei in indices:
                        if nei not in visited:  # Добавляем только не посещенные точки
                            queue.append(nei)
                            visited.add(nei)

            cur_cluster += 1  # Переходим к следующему кластеру

        return self.labels
















# Task 3


class AgglomertiveClustering:
    def __init__(self, n_clusters: int = 16, linkage: str = "average"):
        """
        
        Parameters
        ----------
        n_clusters : int
            Количество кластеров, которые необходимо найти (то есть, кластеры 
            итеративно объединяются, пока их не станет n_clusters)
        linkage : str
            Способ для расчета расстояния между кластерами. Один из 3 вариантов:
            1. average --- среднее расстояние между всеми парами точек, 
               где одна принадлежит первому кластеру, а другая - второму.
            2. single --- минимальное из расстояний между всеми парами точек, 
               где одна принадлежит первому кластеру, а другая - второму.
            3. complete --- максимальное из расстояний между всеми парами точек,
               где одна принадлежит первому кластеру, а другая - второму.
        """
        pass
    
    def fit_predict(self, X: np.array, y = None) -> np.array:
        """
        Кластеризует элементы из X, 
        для каждого возвращает индекс соотв. кластера.
        Parameters
        ----------
        X : np.array
            Набор данных, который необходимо кластеризовать.
        y : Ignored
            Не используемый параметр, аналогично sklearn
            (в sklearn считается, что все функции fit_predict обязаны принимать 
            параметры X и y, даже если y не используется).
        Return
        ------
        labels : np.array
            Вектор индексов кластеров
            (Для каждой точки из X индекс соотв. кластера).

        """
        pass
