
import numpy as np
# Task 1
def mse(y_true:np.ndarray, y_predicted:np.ndarray):
    metric = np.mean((y_true - y_predicted)**2)
    return metric

def r2(y_true:np.ndarray, y_predicted:np.ndarray):
    a = np.sum((y_true - np.mean(y_true))**2)
    b = np.sum((y_true - y_predicted)**2)

    return 1 - (b/a)

# Task 2

class NormalLR:
    def __init__(self):
        self.weights = None # Save weights here
    
    def fit(self, X:np.ndarray, y:np.ndarray):
        X = np.hstack([np.ones((X.shape[0], 1)), X])
        self.weights = np.linalg.inv(X.T @ X ) @ X.T @ y
    
    def predict(self, X:np.ndarray) -> np.ndarray:
        X = np.hstack([np.ones((X.shape[0], 1)), X])
        return X @ self.weights

# Task 3
class GradientLR:
    def __init__(self, alpha: float, iterations=10000, l=0.):
        self.weights = None
        self.alpha = alpha
        self.l = l
        self.iterations = iterations
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.hstack([np.ones((X.shape[0], 1)), X])
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        
        for _ in range(self.iterations):
            predictions = X @ self.weights
            mse_gradient = (1 / n_samples) * (X.T @ (predictions - y))
            gradient = mse_gradient + self.l * np.sign(self.weights)
            self.weights -= self.alpha * gradient
    def predict(self, X: np.ndarray):
        X = np.hstack([np.ones((X.shape[0], 1)), X])
        return X @ self.weights



# Task 4

def get_feature_importance(linear_regression):
    return np.abs(linear_regression.weights[1:])

def get_most_important_features(linear_regression):
    most_important_features = get_feature_importance(linear_regression)
    return np.argsort(-most_important_features)