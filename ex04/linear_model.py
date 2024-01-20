import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error

class MyLinearRegression:
    def __init__(self, thetas) -> None:
        if not isinstance(thetas, np.ndarray):
            return None
        self.thetas = thetas

    def predict_(self,x):
        try:
            if not isinstance(x, np.ndarray):
                return None
            x_one = np.hstack((np.ones(x.shape),x)) # [7x1]->[7x2]
            return x_one.dot(self.thetas) # [7x2] @ [2x1]
        except Exception as e:
            print(e)
    
    def mse_(y_hat, y):
        if len(y.shape) == 1:
            y = y.reshape((-1, 1))
        if len(y_hat.shape) == 1:
            y_hat = y_hat.reshape((-1, 1))
        if y.shape != y_hat.shape:
            return None
        return np.mean((y_hat - y) ** 2)

        
if __name__=='__main__':    
    data = pd.read_csv("are_blue_pills_magic.csv")
    Xpill = np.array(data['Micrograms']).reshape(-1,1)
    Yscore = np.array(data['Score']).reshape(-1,1)
    linear_model1 = MyLinearRegression(np.array([[89.0], [-8]]))
    linear_model2 = MyLinearRegression(np.array([[89.0], [-6]]))
    Y_model1 = linear_model1.predict_(Xpill)
    # print(Y_model1)
    Y_model2 = linear_model2.predict_(Xpill)
    # print(Y_model2)
    assert MyLinearRegression.mse_(Yscore, Y_model1) == mean_squared_error(Yscore, Y_model1)
    # 57.603042857142825
    MyLinearRegression.mse_(Yscore, Y_model2) == mean_squared_error(Yscore, Y_model2)
    # 232.16344285714285

