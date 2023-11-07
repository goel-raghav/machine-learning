import numpy as np
import matplotlib.pyplot as plt

class Model:
    def __init__(self, X: np.ndarray, W: np.ndarray, Y: np.ndarray):
        self.X = X
        ones = np.ones((X.shape[0], 1))
        self.X = np.column_stack((ones, self.X))
        
        self.W = W
        self.Y = Y
        
    def fit(self, max_iter: int = 1000, learning_rate: float = 0.01) -> None:
        finished = False
        count = 0
        
        while not finished and count < max_iter:
            descent = self._batch_descent()
            self.W -= learning_rate * descent
            count += 1
            
            if np.linalg.norm(descent) < 0.00001:
                finished = True
            
        return count
        
    def _batch_descent(self):
        Y_pred = np.dot(self.X, self.W)
        difference = (Y_pred - self.Y).reshape(-1, 1)
        descent = self.X * difference
        descent = np.mean(descent, axis = 0)
        return descent
    
    def scale(self):
        mean = np.mean(self.X, axis=0)
        std = np.std(self.X, axis=0)
        
        self.X -= mean
        self.X /= std
        