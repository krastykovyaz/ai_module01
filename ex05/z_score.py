import numpy as np

def zscore(x: np.ndarray):
    """Computes the normalized version of a non-empty numpy.ndarray using the z-score standardization.
    Args:
    x: has to be an numpy.ndarray, a vector.
    Returns:
    x’ as a numpy.ndarray.
    None if x is a non-empty numpy.ndarray or not a numpy.ndarray.
    Raises:
    This function shouldn’t raise any Exception.
    """
    try:
        if not isinstance(x, np.ndarray):
            return None
        if x.size==0:
            return None
        m = len(x)
        mean_x = np.mean(x)
        numerator = [x[i] - mean_x for i in range(len(x))]
        denominator = np.sqrt(np.sum([(item)**2 for item in numerator]) / (m))
        return (numerator/denominator).reshape((-1,))
    except Exception as e:
        print(e)
        return None


if __name__=='__main__':
    # Example 1:
    X = np.array([0, 15, -9, 7, 12, 3, -21])
    print(zscore(X))
    # Output:
    np.array([-0.08620324, 1.2068453 , -0.86203236, 0.51721942, 0.94823559,
    0.17240647, -1.89647119])
    # Example 2:
    Y = np.array([2, 14, -13, 5, 12, 4, -19]).reshape((-1, 1))
    # Output:
    y = np.array([ 0.11267619, 1.16432067, -1.20187941, 0.37558731, 0.98904659,
    0.28795027, -1.72770165])
    assert np.allclose(zscore(Y), y)
    
