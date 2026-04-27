import numpy as np

def bohachevsky(x):
    d = len(x)
    total = 0
    for i in range(d - 1):
        term1 = x[i]**2
        term2 = 2 * (x[i+1]**2)
        term3 = -0.3 * np.cos(3 * np.pi * x[i])
        term4 = -0.4 * np.cos(4 * np.pi * x[i+1])
        total += term1 + term2 + term3 + term4
    return total + 0.7

def ackley(x, a=20, b=0.2, c=2*np.pi):

    # Find out how many dimensions our particle is searching in
    d = len(x) 
    
    # Using np.sum and np.mean makes the array math incredibly fast
    sum_sq_term = -a * np.exp(-b * np.sqrt(np.sum(x**2) / d))
    cos_term = -np.exp(np.sum(np.cos(c * x)) / d)
    
    return sum_sq_term + cos_term + a + np.exp(1)

def power_sum(x):
    """
    Power Sum Function (2D)
    Global Minimum: 0 at x = [1, 2] (or [2, 1])
    Recommended Search Space: [0, 4]
    """
    # The standard 'b' constants for a 2-Dimensional Power Sum
    b = np.array([3, 5])
    d = len(x)
    total = 0
    
    for i in range(1, d + 1):
        # Calculate the sum of every coordinate raised to the power of i
        inner_sum = np.sum(x ** i)
        # Subtract the corresponding 'b' constant and square the result
        total += (inner_sum - b[i-1]) ** 2
        
    return total

def rastrigin(x):
    """
    Rastrigin Function (N-dimensional)
    Global Minimum: 0 at x = [0, ..., 0]
    Recommended Search Space: [-5.12, 5.12]
    """
    d = len(x)
    
    # The Rastrigin formula: 10 * d + sum(x_i^2 - 10 * cos(2 * pi * x_i))
    sum_term = np.sum(x**2 - 10 * np.cos(2 * np.pi * x))
    
    return 10 * d + sum_term
