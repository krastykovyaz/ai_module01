import numpy as np
def fit_(x, y, theta, alpha, max_iter):
    """
    Description:
    Fits the model to the training dataset contained in x and y.
    Args:
    x: has to be a numpy.ndarray, a vector of dimension m * 1: (number of training examples, 1).
    y: has to be a numpy.ndarray, a vector of dimension m * 1: (number of training examples, 1).
    theta: has to be a numpy.ndarray, a vector of dimension 2 * 1.
    alpha: has to be a float, the learning rate
    max_iter: has to be an int, the number of iterations done during the gradient descent
    Returns:
    new_theta: numpy.ndarray, a vector of dimension 2 * 1.
    None if there is a matching dimension problem.
    Raises:
    This function should not raise any Exception.
    """
    try:
        if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray) or not isinstance(theta, np.ndarray):
            return None
        if x.size == 0 or y.size == 0 or theta.size == 0:
            return None
        if x.shape != y.shape or theta.shape != (2, 1):
            return None
        def simple_grad(x,y,theta):
            m = len(x)
            grad = np.zeros(theta.shape) # 2x1
            for i in range(m):
                grad[0,0] = (theta[1,0] * x[i,0] - theta[0,0] * y[i,0]) 
                grad[1,0] = ((theta[1,0] * x[i,0] - theta[0,0] * y[i,0]) * x[i, 0]) 
            return grad / m # 2x1
        theta = theta.astype("float64")
        for _ in range(max_iter):
            grad = simple_grad(x,y,theta) # 2x1
            theta -= grad * alpha
        return theta
    except Exception as e:
        print(e)
        return None
    
def predict(x, theta):
    x_one = np.hstack((np.ones(x.shape), x)) # 5x2
    return x_one.dot(theta) # 5x1

if __name__=='__main__':
    x = np.array([[12.4956442], [21.5007972], [31.5527382], [48.9145838], [57.5088733]])
    y = np.array([[37.4013816], [36.1473236], [45.7655287], [46.6793434], [59.5585554]])
    theta= np.array([1, 1]).reshape((-1, 1))
    # Example 0:
    theta1 = fit_(x, y, theta, alpha=5e-8, max_iter=1500000)
    print(theta1)
    # Output:
    np.array([[1.40709365],[1.1150909 ]])
    # Example 1:
    print(predict(x.reshape(-1,1), theta1))
    # Output:
    np.array([[15.3408728 ],
        [25.38243697],
        [36.59126492],
        [55.95130097],
        [65.53471499]])