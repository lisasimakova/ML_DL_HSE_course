import numpy as np
import copy
import cvxopt
from cvxopt import spmatrix, matrix, solvers
from sklearn.datasets import make_classification, make_moons, make_blobs
from typing import NoReturn, Callable

solvers.options['show_progress'] = False


class LinearSVM:
    def __init__(self, C: float):
        """
        Parameters
        ----------
        C : float
            Soft margin coefficient.
        """
        self.C = C
        self.w = None
        self.b = None
        self.support = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:

        n_samples, _ = X.shape


        P = matrix(np.outer(y, y) * np.dot(X, X.T))
        q = matrix(-np.ones(n_samples))
        G = matrix(np.vstack((-np.eye(n_samples), np.eye(n_samples))))
        h = matrix(np.hstack((np.zeros(n_samples), self.C * np.ones(n_samples))))
        A = matrix(y.reshape(1, -1).astype(float))
        b = matrix(0.0)


        solution = cvxopt.solvers.qp(P, q, G, h, A, b)


        alphas = np.array(solution['x']).flatten()


        support = alphas > 1e-5
        self.support = support
        self.w = np.dot(X.T, alphas * y)
        self.b = np.mean(y[support] - np.dot(X[support], self.w))

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        Возвращает значение решающей функции.

        Parameters
        ----------
        X : np.ndarray
            Данные, для которых нужно посчитать значение решающей функции.

        Return
        ------
        np.ndarray
            Значение решающей функции для каждого элемента X 
            (т.е. то число, от которого берем знак с целью узнать класс).
        """
        return np.dot(X, self.w) + self.b

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Классифицирует элементы X.

        Parameters
        ----------
        X : np.ndarray
            Данные, которые нужно классифицировать

        Return
        ------
        np.ndarray
            Метка класса для каждого элемента X.
        """
        return np.sign(self.decision_function(X))


def get_polynomial_kernel(c=1, power=2):
    """Возвращает полиномиальное ядро."""
    def polynomial_kernel(X, x_prime):
        return (c + np.dot(X, x_prime)) ** power
    return polynomial_kernel

def get_gaussian_kernel(sigma=1.):
    """Возвращает ядро Гаусса."""
    def gaussian_kernel(X, x_prime):
        return np.exp(-sigma *np.linalg.norm(X - x_prime, axis=1) ** 2 )
    return gaussian_kernel


       



class KernelSVM:
    def __init__(self, C: float, kernel: Callable):
        self.C = C  # Коэффициент для мягкого зазора
        self.kernel = kernel  # Функция ядра
        self.support = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        n_samples, n_features = X.shape
        
        # Создаем ядро
        K = np.array([[self.kernel(X[i], X[j]) for j in range(n_samples)] for i in range(n_samples)])
        
        P = cvxopt.matrix(np.outer(y, y) * K)  # P = y * K * y^T
        q = cvxopt.matrix(-np.ones(n_samples))  # q = -1 для всех примеров

        # Ограничения
        G = cvxopt.matrix(np.vstack((-np.eye(n_samples), np.eye(n_samples))))  # Ограничения для альфа
        h = cvxopt.matrix(np.hstack((np.zeros(n_samples), np.ones(n_samples) * self.C)))  # 0 <= alpha_i <= C

        # Ограничение на сумму альфа_i * y_i
        A = cvxopt.matrix(y.reshape(1, -1).astype(float))
        b = cvxopt.matrix(np.ones(1))

        # Решаем задачу квадратичного программирования
        solution = cvxopt.solvers.qp(P, q, G, h, A, b)

        # Извлекаем альфа
        alphas = np.ravel(solution['x'])

        # Сохраняем коэффициенты альфа для опорных векторов
        self.alphas = alphas
        self.X = X
        self.y = y

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        # Вычисляем решающую функцию для новых данных
        result = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            result[i] = np.sum(self.alphas * self.y * np.array([self.kernel(X[i], x_train) for x_train in self.X]))
        return result

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.sign(self.decision_function(X))
