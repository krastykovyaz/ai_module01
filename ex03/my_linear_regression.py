import numpy as np


class MyLinearRegression:
    """
    Description:
    My personnal linear regression class to fit like a boss.
    """
    def __init__(self, thetas, alpha=0.001, max_iter=1000):
        self.alpha = alpha
        self.max_iter = max_iter
        self.thetas = thetas

    def simple_grad(self, x, y):
        m = len(y) # 2
        x_one = np.hstack((np.ones((x.shape)), x)) # 5x2
        x_one_theta = x_one @ self.thetas # [5x2] @ [2x1] -> [5x1]
        return x_one.T @ (x_one_theta - y) / x.shape[0] # [2x5] @ [5x1] -> [2x1]

    def fit_(self, x, y):
        if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray):
            return None
        if x.size==0 or y.size == 0 or x.shape != y.shape:
            return None 
        for _ in range(self.max_iter):
            grad = self.simple_grad(x, y) # [2x1]
            self.thetas = self.thetas - self.alpha * grad
        return self.thetas

    def predict_(self, x):
        x_one=np.hstack((np.ones((x.shape[0],1)), x)) # 5x2
        return x_one.dot(self.thetas) # 5x2 * 2x1 -> 5x1

    @staticmethod
    def mse(y, y_hat):
        return np.mean((y - y_hat) ** 2) / 2

    def loss_(self, y, y_hat):
        """
        Calculates the value of the loss function.
        Args:
        y: has to be a numpy.array, a vector.
        y_hat: has to be a numpy.array, a vector.
        Returns:
        J_value : has to be a float.
        None if there is a dimension matching problem between X, Y, or theta.
        None if any argument is not of the expected type.
        """
        if len(y.shape) == 1:
            y = y.reshape((-1, 1))
        if len(y_hat.shape) == 1:
            y_hat = y_hat.reshape((-1, 1))
        if y.shape != y_hat.shape:
            return None

        return self.mse(y_hat, y)
    
    def loss_elem_(self, y, y_hat):
        """
        Calculates all the elements (y_pred - y)^2 of the loss function.
        Args:
        y: has to be a numpy.array, a vector.
        y_hat: has to be a numpy.array, a vector.
        Returns:
        J_elem: numpy.array, a vector of dimension (number of training examples, 1).
        None if there is a dimension matching problem between X, Y, or theta.
        None if any argument is not of the expected type.
        """
        if len(y.shape) == 1:
            y = y.reshape((-1, 1))
        if len(y_hat.shape) == 1:
            y_hat = y_hat.reshape((-1, 1))
        if y.shape != y_hat.shape:
            return None
        return (y_hat - y) ** 2 

        
if __name__=='__main__':
    x = np.array([[12.4956442], [21.5007972], [31.5527382], [48.9145838], [57.5088733]])
    y = np.array([[37.4013816], [36.1473236], [45.7655287], [46.6793434], [59.5585554]])
    lr1 = MyLinearRegression(np.array([[2], [0.7]]))
    # Example 0.0:
    y_hat = lr1.predict_(x)
    # print(y_hat)
    # Output:
    np.array([[10.74695094],
    [17.05055804],
    [24.08691674],
    [36.24020866],
    [42.25621131]])

    # Example 0.1:
    print(lr1.loss_elem_(y, y_hat))
    # Output:
    np.array([[710.45867381],
    [364.68645485],
    [469.96221651],
    [108.97553412],
    [299.37111101]])

    # Example 0.2:
    # print(lr1.loss_(y, y_hat))
    # Output:
    195.34539903032385

    # Example 1.0:
    lr2 = MyLinearRegression(np.array([[1], [1]]), 5e-8, 1500000)
    lr2.fit_(x, y)
    # print(lr2.thetas)
    # Output:
    np.array([[1.40709365], [1.1150909 ]])

    # Example 1.1:
    y_hat = lr2.predict_(x)
    print(y_hat)
    # Output:
    np.array([[15.3408728 ],
    [25.38243697],
    [36.59126492],
    [55.95130097],
    [65.53471499]])

    # Example 1.2:
    lr2.loss_elem_(y, y_hat)
    print(lr2.loss_elem_(y, y_hat))
    # Output:
    np.array([[486.66604863],
    [115.88278416],
    [ 84.16711596],
    [ 85.96919719],
    [ 35.71448348]])

    # Example 1.3:
    print(lr2.loss_(y, y_hat))
    # Output:
    80.83996294128525

    