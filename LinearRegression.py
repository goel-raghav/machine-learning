import numpy as np
import matplotlib.pyplot as plt

class Model:
    def __init__(self, X: np.ndarray, Y: np.ndarray, scale: bool = False):
        self.X = X
        self._mean = np.mean(self.X, axis=0)
        self._std = np.std(self.X, axis=0)
        
        
        if scale:
            self.X = self._scale(self.X)
            
        ones = np.ones((self.X.shape[0], 1))
        self.X = np.column_stack((ones, self.X))
        
        self.W = np.zeros((self.X.shape[1]))
        self.Y = Y
        
        
        
    def fit(self, max_iter: int = 1000, learning_rate: float = 0.01) -> None:
        finished = False
        count = 0
        
        while not finished and count < max_iter:
            descent = self._batch_descent()
            self.W -= learning_rate * descent
            count += 1
            
            if np.linalg.norm(descent) < 0.0000001:
                finished = True
            
        return count
        
    def _batch_descent(self):
        Y_pred = np.dot(self.X, self.W)
        difference = (Y_pred - self.Y).reshape(-1, 1)
        descent = self.X * difference
        descent = np.mean(descent, axis = 0)
        return descent

    def _scale(self, X):
        X -= self._mean
        X /= self._std
        return X
    
    def predict(self, X):
        X = self._scale(X)
        ones = np.ones((X.shape[0], 1))
        X = np.column_stack((ones, X))
        
        
        Y_pred = np.dot(X, self.W)
        return Y_pred
        