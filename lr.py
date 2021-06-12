import pandas as pd
import numpy as np


class LinearRegression:
  
    def predict(self, X):
        return np.dot(X, self._W)
  
    def _gradient_descent_step(self, X, targets, lr):
        predictions = self.predict(X)
        error = predictions - targets
        gradient = np.dot(X.T,  error) / len(X)
        self._W -= lr * gradient
      
    def fit(self, X, y, n_iter=100000, lr=0.01):
        self._W = np.zeros(X.shape[1])
        for i in range(n_iter):        
            self._gradient_descent_step(x, y, lr)
        return self

def loss(h, y):
    er = (h - y)**2
    n = len(y)
    return 1.0 / (2 * n) * er.sum()





data = pd.read_csv('moscow.csv')




y = data['price']
x = data[['lat', 'lon', 'area', 'kitchen_area']]

with open('mean.txt', 'w') as w:
    for i in x.mean():
        w.write(str(i))
        w.write(' ')

with open('std.txt', 'w') as w:
    for i in x.std():
        w.write(str(i))
        w.write(' ')

x = (x - x.mean()) / x.std()
print(x)
x = np.c_[np.ones(x.shape[0]), x]
print(x)


clf = LinearRegression()
clf.fit(x, y, n_iter=2000, lr = 0.01)
print(*clf._W)
