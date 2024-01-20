import numpy as np

def simple_gradient(x, y, theta):
    """Computes a gradient vector from three non-empty numpy.array, with a for-loop.
    The three arrays must have compatible shapes.
    Args:
    x: has to be an numpy.array, a vector of shape m * 1.
    y: has to be an numpy.array, a vector of shape m * 1.
    theta: has to be an numpy.array, a 2 * 1 vector.
    Return:
    The gradient as a numpy.array, a vector of shape 2 * 1.
    None if x, y, or theta are empty numpy.array.
    None if x, y and theta do not have compatible shapes.
    None if x, y or theta is not of the expected type.
    Raises:
    This function should not raise any Exception."""
    try:
        if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray) or not isinstance(theta, np.ndarray):
            return None
        if x.size == 0 or y.size == 0 or theta.size == 0:
            return None
        if x.shape != y.shape or theta.shape != (2, 1):
            return None
        gradient = np.zeros(theta.shape) # [2x1]
        for i in range(x.shape[0]): # 5
            gradient[0,0] += (theta[1,0] * x[i,0] + theta[0,0] - y[i,0]) 
            gradient[1,0] +=  (theta[1,0] * x[i,0] + theta[0,0] - y[i,0]) * x[i, 0] 
        gradient /=  x.shape[0]
        return gradient  # [2x1]
    except Exception as e:
        print(e)
        return None
    
if __name__=='__main__':
    x = np.array([12.4956442, 21.5007972, 31.5527382, 48.9145838, 57.5088733]).reshape((-1, 1)) # (5x1)
    y = np.array([37.4013816, 36.1473236, 45.7655287, 46.6793434, 59.5585554]).reshape((-1, 1)) # (5x1)
    # Example 0:
    theta1 = np.array([2, 0.7]).reshape((-1, 1)) # (2x1)
    print(simple_gradient(x,y,theta1))
    # Example 1:
    theta2 = np.array([1, -0.4]).reshape((-1, 1))
    print(simple_gradient(x, y, theta2))
    # Output:
    np.array([[-57.86823748], [-2230.12297889]])
