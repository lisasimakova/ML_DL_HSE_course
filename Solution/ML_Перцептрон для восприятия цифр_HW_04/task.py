import numpy as np
import copy
from typing import NoReturn


# Task 1

class Perceptron:
    def __init__(self, iterations: int = 100):
        """
        Parameters
        ----------
        iterations : int
        Количество итераций обучения перцептрона.

        Attributes
        ----------
        w : np.ndarray
        Веса перцептрона размерности X.shape[1] + 1 (X --- данные для обучения), 
        w[0] должен соответстовать константе, 
        w[1:] - коэффициентам компонент элемента X.

        Notes
        -----
        Вы можете добавлять свои поля в класс.
        
        """

        self.w = None
        self.iterations = iterations
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Обучает простой перцептрон. 
        Для этого сначала инициализирует веса перцептрона,
        а затем обновляет их в течении iterations итераций.
        
        Parameters
        ----------
        X : np.ndarray
            Набор данных, на котором обучается перцептрон.
        y: np.ndarray
            Набор меток классов для данных.
        
        """
        y = np.where(y > 0, 1, -1) 
        d = X.shape[1]
        self.w = np.zeros(d + 1) 
        X_bias = np.hstack([np.ones((X.shape[0], 1)), X])

        for _ in range(self.iterations):
            predictions = np.sign(X_bias @ self.w)
            indecis = (predictions != y) 

            if not np.any(indecis): 
                break

            self.w += np.sum(y[indecis, None] * X_bias[indecis], axis=0)
        
            
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Предсказывает метки классов.
        
        Parameters
        ----------
        X : np.ndarray
            Набор данных, для которого необходимо вернуть метки классов.
        
        Return
        ------
        labels : np.ndarray
            Вектор индексов классов 
            (по одной метке для каждого элемента из X).
        
        """
        X_bias = np.hstack([np.ones((X.shape[0], 1)), X])
        labels = X_bias @ self.w
        return np.where(labels > 0, 1, 0)
    
# Task 2

class PerceptronBest:

    def __init__(self, iterations: int = 100):
        """
        Parameters
        ----------
        iterations : int
        Количество итераций обучения перцептрона.

        Attributes
        ----------
        w : np.ndarray
        Веса перцептрона размерности X.shape[1] + 1 (X --- данные для обучения), 
        w[0] должен соответстовать константе, 
        w[1:] - коэффициентам компонент элемента X.

        Notes
        -----
        Вы можете добавлять свои поля в класс.
        
        """

        self.w = None
        self.iterations = iterations
    

    def fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Обучает перцептрон.

        Для этого сначала инициализирует веса перцептрона, 
        а затем обновляет их в течении iterations итераций.

        При этом в конце обучения оставляет веса, 
        при которых значение accuracy было наибольшим.
        
        Parameters
        ----------
        X : np.ndarray
            Набор данных, на котором обучается перцептрон.
        y: np.ndarray
            Набор меток классов для данных.
        
        """
        
        y = np.where(y > 0, 1, -1)
        X_bias = np.hstack([np.ones((X.shape[0], 1)), X])
        self.w = np.zeros(X_bias.shape[1])
        best_weights = np.copy(self.w)
        min_error = X_bias.shape[0] + 1 

        for _ in range(self.iterations):
            predictions = np.sign(X_bias @ self.w)
            errors = y - predictions
            error_count = np.count_nonzero(errors)

            if error_count < min_error:
                min_error = error_count
                best_weights = np.copy(self.w)

            self.w += errors @ X_bias

        self.w = best_weights

            
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Предсказывает метки классов.
        
        Parameters
        ----------
        X : np.ndarray
            Набор данных, для которого необходимо вернуть метки классов.
        
        Return
        ------
        labels : np.ndarray
            Вектор индексов классов 
            (по одной метке для каждого элемента из X).
        
        """
        X_bias = np.hstack([np.ones((X.shape[0], 1)), X])
        return np.where(X_bias @ self.w > 0, 1, 0)
    
# Task 3

def transform_images(images: np.ndarray) -> np.ndarray:
    """
    Переводит каждое изображение в вектор из двух элементов.
        
    Parameters
    ----------
    images : np.ndarray
        Трехмерная матрица с черное-белыми изображениями.
        Её размерность: (n_images, image_height, image_width).

    Return
    ------
    np.ndarray
        Двумерная матрица с преобразованными изображениями.
        Её размерность: (n_images, 2).
    """
    n_images, image_height, image_width = images.shape
    '''
    Вычислим симметричность. Поделим каждую картинку на две части, 
    посчитаем разницу и сложим полученные результаты по двум осям.
    Если картинка симметрична, то разница минимальна
    '''
    
    left_half = images[:, :, :image_width // 2]
    right_half = images[:, :, image_width // 2:]
    symmetry = -np.sum(np.abs(left_half - np.flip(right_half, axis=2)), axis=(1, 2))


    '''Вычислим изменение между соседними столбцами, у 5 их больше, чем у 1'''
    vertical_lines = np.sum(np.abs(np.diff(images, axis=2)), axis=(1, 2))
    

    #return np.column_stack((symmetry, vertical_lines))
    return np.stack((images.sum(axis=1).max(axis=-1), images.sum(axis=2).max(axis=-1))).T