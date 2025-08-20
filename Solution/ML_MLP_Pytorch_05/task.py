import numpy as np
import copy
from typing import List, NoReturn
import torch
from torch import nn
import torch.nn.functional as F


# Task 1

class Module:
    """
    Абстрактный класс. Его менять не нужно. Он описывает общий интерфейс взаимодествия со слоями нейронной сети.
    """
    def forward(self, x):
        pass
    
    def backward(self, d):
        pass
        
    def update(self, alpha):
        pass

class Linear(Module):
    """
    Линейный полносвязный слой.
    """
    def __init__(self, in_features: int, out_features: int):
        """
        Parameters
        ----------
        in_features : int
            Размер входа.
        out_features : int 
            Размер выхода.
    
        Notes
        -----
        W и b инициализируются случайно.
        """
        s = 1 / np.sqrt(in_features + out_features)
        self.W = np.random.normal(0, s, (in_features, out_features))
        self.b = np.zeros(out_features)
        self.x = None
        self.W_d = None
        self.b_d = None
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Возвращает y = Wx + b.

        Parameters
        ----------
        x : np.ndarray
            Входной вектор или батч.
            То есть, либо x вектор с in_features элементов,
            либо матрица размерности (batch_size, in_features).
    
        Return
        ------
        y : np.ndarray
            Выход после слоя.
            Либо вектор с out_features элементами,
            либо матрица размерности (batch_size, out_features)

        """
        
        self.x = np.atleast_2d(x)
        return (self.x @ self.W) + self.b

    def backward(self, d: np.ndarray) -> np.ndarray:
        """
        Считает градиенты.
        d — это градиент ошибки по отношению к выходу слоя.
        """
        if d.ndim == 1:
            d = np.expand_dims(d, axis=0)
        self.W_d = self.x.T @ d 
        self.b_d = np.sum(d, axis=0) 
        return d @ self.W.T

    def update(self, alpha: float) -> None:
        """
        Обновляет веса и смещения с заданной скоростью обучения.
        alpha — это скорость обучения.
        """
        self.W -= alpha * self.W_d
        self.b -= alpha * self.b_d


class ReLU(Module):
    """
    Слой, соответствующий функции активации ReLU. Данная функция возвращает новый массив, в котором значения меньшие 0 заменены на 0.
    """
    def __init__(self):

        self.x = None
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Возвращает y = max(0, x).

        Parameters
        ----------
        x : np.ndarray
            Входной вектор или батч.
    
        Return
        ------
        y : np.ndarray
            Выход после слоя (той же размерности, что и вход).

        """
        self.x = x 
        return np.maximum(0, x)
        
    def backward(self, d: np.ndarray) -> np.ndarray:
        """
        Cчитает градиент при помощи обратного распространения ошибки.

        Parameters
        ----------
        d : np.ndarray
            Градиент.
        Return
        ------
        np.ndarray
            Новое значение градиента.
        """
        return d * (self.x > 0)



# Task 2



class MLPClassifier:
    def __init__(self, modules: List[Module], epochs: int = 40, alpha: float = 0.01, batch_size: int = 32):
        """
        Parameters
        ----------
        modules : List[Module]
            Cписок, состоящий из ранее реализованных модулей и 
            описывающий слои нейронной сети. 
            В конец необходимо добавить Softmax.
        epochs : int
            Количество эпох обучения.
        alpha : float
            Cкорость обучения.
        batch_size : int
            Размер батча, используемый в процессе обучения.
        """
        self.modules = modules
        self.modules.append(Softmax())
        self.epochs = epochs
        self.alpha = alpha
        self.batch_size = batch_size
    # def one_hot_encode(y, n_classes):
    #     # Преобразуем метки в one-hot представление
    #     return np.eye(n_classes)[y]



    # def cross_entropy_loss_derivative(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    #     epsilon = 1e-15
    #     y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    #     return (y_pred - y_true) / y_true.shape[0]

    def fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Обучает нейронную сеть заданное число эпох. 
        В каждой эпохе необходимо использовать cross-entropy loss для обучения, 
        а так же производить обновления не по одному элементу, а используя батчи (иначе обучение будет нестабильным и полученные результаты будут плохими.

        Parameters
        ----------
        X : np.ndarray
            Данные для обучения.
        y : np.ndarray
            Вектор меток классов для данных.
        """
        n_classes = len(np.unique(y))
        y_encode = np.eye(n_classes)[y]
        for _ in range(self.epochs):
            n_samples = X.shape[0]
            indices = np.random.permutation(n_samples)
            X_s = X[indices]
            y_s = y_encode[indices]

            for i in range(0, n_samples, self.batch_size):
                X_batch = X_s[i:i + self.batch_size]
                y_batch = y_s[i:i + self.batch_size]
                for module in self.modules:
                    X_batch = module.forward(X_batch)
                d = X_batch - y_batch
                for module in reversed(self.modules):
                    d = module.backward(d)
                for module in self.modules:
                    module.update(self.alpha)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Предсказывает вероятности классов для элементов X.

        Parameters
        ----------
        X : np.ndarray
            Данные для предсказания.
        
        Return
        ------
        np.ndarray
            Предсказанные вероятности классов для всех элементов X.
            Размерность (X.shape[0], n_classes)
        
        """
        output = X
        for module in self.modules:
            output = module.forward(output)
        return output

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Предсказывает метки классов для элементов X.

        Parameters
        ----------
        X : np.ndarray
            Данные для предсказания.
        
        Return
        ------
        np.ndarray
            Вектор предсказанных классов
        
        """
        return np.argmax(self.predict_proba(X), axis=1)
    

class Softmax(Module):
    def __init__(self):
        pass
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Вычисление softmax для входных данных.
        
        Для каждого элемента x, softmax = exp(x) / sum(exp(x) по всем элементам в ряду.
        """
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True)) 
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True) 
    def backward(self, d: np.ndarray) -> np.ndarray:
        return d
    def update(self, alpha: float) -> NoReturn:
        return


#Task 3

classifier_moons = MLPClassifier( 
    modules=[
        Linear(in_features=2, out_features=8),
        ReLU(),
        Linear(in_features=8, out_features=4),
        ReLU(),
        Linear(in_features=4, out_features=2)
    ],
    epochs=150,
    alpha=0.01,
    batch_size=29
)

classifier_blobs = MLPClassifier(
    modules=[
        Linear(in_features=2, out_features=32),
        ReLU(),
        Linear(in_features=32, out_features=16),
        ReLU(),
        Linear(in_features=16, out_features=3)
    ],
    epochs=150,
    alpha=0.01,
    batch_size=30
)





# Task 4

class TorchModel(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass
    
    def load_model(self):
        """
        Используйте torch.load, чтобы загрузить обученную модель
        Учтите, что файлы решения находятся не в корне директории, поэтому необходимо использовать следующий путь:
        `__file__[:-7] + "model.pth"`, где "model.pth" - имя файла сохраненной модели `
        """
        pass
    
    def save_model(self):
        """
        Используйте torch.save, чтобы сохранить обученную модель
        """
        pass
        
def calculate_loss(X: torch.Tensor, y: torch.Tensor, model: TorchModel):
    """
    Cчитает cross-entropy.

    Parameters
    ----------
    X : torch.Tensor
        Данные для обучения.
    y : torch.Tensor
        Метки классов.
    model : Model
        Модель, которую будем обучать.

    """
    pass



        



